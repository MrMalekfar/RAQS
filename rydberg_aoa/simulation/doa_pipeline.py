"""DOA-INTEGRATE-001 -- production DOA orchestration (method selection).

ONE production orchestration call (estimate_doa_from_results) per processed
simulation run dispatches the configured DOA method:

* registry default comes from config.doa_config (lazy, per call);
* geometry is built from the ACTUAL path positions (M inferred dynamically);
* complex IF snapshots are extracted EXACTLY ONCE from the SAME run and
  shared by every covariance / structured estimator;
* search-free estimators (esprit / root_music) never receive a scan grid;
* incompatible geometry raises IncompatibleGeometryError (no fallback);
* legacy estimated_aoa / angles / power keys are preserved.

The validated phase_beamforming production contract (true AoA -30 deg,
estimated AoA -30 deg, phase slope -22.5 deg/path on the frozen
six-path configuration) is preserved bit-exactly.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from rydberg_aoa.config import doa_config
from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import (
    ArrayGeometry,
    DOAResult,
    IncompatibleGeometryError,
    get_doa_estimator,
)
from rydberg_aoa.doa.snapshots import IFSnapshotData, extract_if_snapshots
from .time_domain import atomic_if_polarity

__all__ = [
    "build_array_geometry",
    "resolve_doa_method",
    "build_estimator_kwargs",
    "build_doa_context",
    "extract_snapshots_once",
    "estimate_doa_from_results",
    "merge_doa_into_results",
    "DOA_INTEGRATE",
]

DOA_INTEGRATE = "DOA-INTEGRATE-001"

# Estimator capability classes (single source of truth: the registry's
# metadata via the estimator's capability attributes).
_SEARCH_FREE_METHODS = ("esprit", "root_music")
_SNAPSHOT_METHODS = ("bartlett", "mvdr", "capon", "music", "esprit",
                     "root_music")
_LOADING_METHODS = ("mvdr", "capon")
_N_SOURCES_METHODS = ("music", "esprit", "root_music")


def build_array_geometry(array_positions) -> ArrayGeometry:
    """Build the geometry from the ACTUAL path-position array.

    Accepts the (M, 2) production path positions (x, y) in metres, or an
    (M, 3) array (x, y, z).  M is INFERRED dynamically -- the element
    count is never hard-coded and never read back from configuration.
    """
    arr = np.asarray(array_positions, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            "array positions must be a 2-D (M, 2) or (M, 3) array in "
            f"metres; got shape {arr.shape}"
        )
    if arr.shape[1] == 3:
        return ArrayGeometry(arr)
    if arr.shape[1] == 2:
        return ArrayGeometry.from_xy(arr)
    raise ValueError(
        "array positions must have 2 columns (x, y) or 3 columns "
        f"(x, y, z) in metres; got shape {arr.shape}"
    )


def resolve_doa_method(method: Optional[str] = None) -> str:
    """Resolve the DOA method for THIS call (explicit name wins).

    ``None`` defers to ``config.doa_config.DOA_METHOD`` read LAZILY at
    call time (never snapshotted at import time).  Unknown names raise
    ValueError.
    """
    selected = str(method).strip().lower() if method else doa_config.DOA_METHOD
    if selected not in _SNAPSHOT_METHODS and selected != "phase_beamforming":
        raise ValueError(f"Unknown DOA estimator {selected!r}")
    return selected


def build_estimator_kwargs(method: str, doa_options=None) -> Dict[str, Any]:
    """Estimator kwargs from doa_config, shaped per estimator class.

    Search-free estimators (esprit / root_music) accept NO scan grid, so
    the grid kwarg is withheld for them (they would reject it).
    """
    if doa_options is not None:
        from ..config.doa_options import ProfileDOAOptions
        from ..exceptions import PhysicsConfigurationError
        if not isinstance(doa_options, ProfileDOAOptions):
            raise PhysicsConfigurationError('DOA-PROFILE-001: explicit options must be ProfileDOAOptions.')
        return doa_options.estimator_kwargs(method)
    kwargs: Dict[str, Any] = {
        "elevation_deg": doa_config.DOA_ELEVATION_DEG,
    }
    if method not in _SEARCH_FREE_METHODS:
        kwargs["scan_azimuth_deg"] = doa_config.azimuth_scan_grid_deg()
    if method in _LOADING_METHODS:
        kwargs["loading_rel"] = doa_config.DOA_MVDR_LOADING_REL
    if method in _N_SOURCES_METHODS:
        kwargs["n_sources"] = doa_config.DOA_NUM_SOURCES
    return kwargs


def build_doa_context() -> Dict[str, Any]:
    """Shared estimation context (post-processing information only).

    IMPORTANT -- the spatial carrier is the RF SIGNAL carrier
    (parameters.RF_SIGNAL_FREQUENCY), NEVER the heterodyne IF frequency;
    the IF is used only for temporal snapshot extraction.  The remaining
    keys mirror the estimator configuration from doa_config so the
    context is self-describing for diagnostics; the effective values
    consumed by the estimator core come from the estimator kwargs built
    from the very same configuration fields (single source of truth:
    doa_config).
    """
    return {
        "frequency_hz": params.RF_SIGNAL_FREQUENCY,
        "elevation_deg": doa_config.DOA_ELEVATION_DEG,
        "n_sources": doa_config.DOA_NUM_SOURCES,
        "mvdr_loading_rel": doa_config.DOA_MVDR_LOADING_REL,
    }


def extract_snapshots_once(
    tlist_us: np.ndarray,
    time_traces: Sequence,
    cycles_per_snapshot: Optional[float] = None,
) -> IFSnapshotData:
    """Extract the complex IF snapshots from the production time traces.

    Uses the ACTUAL tlist / traces and the production IF frequency
    (parameters.IF_FREQ, MHz).  The spatial carrier never appears here.
    """
    cycles = (
        doa_config.DOA_SNAPSHOT_CYCLES
        if cycles_per_snapshot is None
        else cycles_per_snapshot
    )
    return extract_if_snapshots(
        tlist_us,
        time_traces,
        params.IF_FREQ,
        cycles_per_snapshot=cycles,
    )

def estimate_doa_from_results(
    array_positions,
    complex_responses: np.ndarray,
    tlist_us: np.ndarray,
    time_traces: Sequence,
    method: Optional[str] = None,
    if_snapshots: Optional[IFSnapshotData] = None,
    rf_orientation: Optional[Any] = None,
    doa_options: Optional[Any] = None,
) -> Tuple[DOAResult, Optional[IFSnapshotData]]:
    """Production DOA dispatch: ONE orchestration call per processed run.

    Args:
        array_positions: ACTUAL path-position array (M, 2) or (M, 3) used
            by the simulation.  M is inferred dynamically.
        complex_responses: legacy (M,) complex IF response vector (kept
            for the validated phase_beamforming contract).
        tlist_us: actual time axis of the time-domain output (us).
        time_traces: M per-path time traces from the same run.
        method: explicit estimator name; None selects
            config.doa_config.DOA_METHOD (lazy, per call).
        if_snapshots: optional pre-extracted snapshot block; when given it
            is never re-extracted. Raw projections receive the RF phase
            conversion once; already normalized outputs are reused unchanged
            for matching IF and atomic polarity. Conflicting context is
            rejected. The physical simulation is never re-run for comparison.
        rf_orientation: RF-DIR-001D - RFTransitionOrientation | None, the
            published RF-edge energy orientation.  For a downward edge
            the snapshot matrix X is conjugated ONCE here so every
            estimator receives the physical +Delta_phi polarity; None
            keeps the historical behaviour bit-identically.  The vector
            branch is normalized at its own production point in
            runner.process_simulation_results (same single rule,
            time_domain.atomic_if_polarity); estimators are never
            sign-patched individually.
        doa_options: immutable ProfileDOAOptions, or None for historical
            lazy doa_config defaults. Explicit options supply every applicable
            algorithm parameter and fixed zero elevation without global writes.

    Returns:
        (doa_result, if_snapshots) where doa_result is the selected
        estimator's DOAResult and if_snapshots is the snapshot block
        (always available, exactly once extracted) for storage in the
        processed results structure.

    Raises:
        IncompatibleGeometryError: when the selected algorithm cannot run
            on the production geometry (e.g. ESPRIT on a non-shift-
            invariant array).  No fallback, no warning-and-continue.
    """
    geometry = build_array_geometry(array_positions)
    selected = resolve_doa_method(method)
    kwargs = build_estimator_kwargs(selected, doa_options)
    estimator = get_doa_estimator(selected, **kwargs)
    context = build_doa_context()
    context.update({key: value for key, value in kwargs.items() if key != 'scan_azimuth_deg'})
    if 'loading_rel' in kwargs:
        context['mvdr_loading_rel'] = kwargs['loading_rel']

    # Snapshots are extracted EXACTLY ONCE per orchestration call so every
    # covariance/structured estimator -- now or in later comparisons -- can
    # consume the SAME X matrix.
    if if_snapshots is None:
        if_snapshots = extract_snapshots_once(tlist_us, time_traces)
    if not isinstance(if_snapshots, IFSnapshotData):
        raise ValueError("if_snapshots must be an IFSnapshotData block")
    if if_snapshots.if_frequency_mhz != params.IF_FREQ:
        raise ValueError(
            "Snapshot signed IF does not match the current run: "
            f"snapshot={if_snapshots.if_frequency_mhz!r} MHz, "
            f"run={params.IF_FREQ!r} MHz."
        )

    # RF-DIR-001D: ONE snapshot-branch conversion at the physics -> AoA
    # boundary (the vector branch was normalized once at its production
    # point in runner.process_simulation_results).  Same rule, single
    # source: time_domain.atomic_if_polarity(rf_orientation.direction).
    polarity = int(atomic_if_polarity(rf_orientation))
    if if_snapshots.normalization_polarity is None:
        # Keep the input block raw; the returned block carries provenance.
        if_snapshots = replace(
            if_snapshots,
            X=np.conj(if_snapshots.X) if polarity < 0 else if_snapshots.X,
            normalization_polarity=polarity,
        )
    elif if_snapshots.normalization_polarity != polarity:
        raise ValueError(
            "Snapshot RF normalization polarity conflicts with this run: "
            f"snapshot={if_snapshots.normalization_polarity}, run={polarity}. "
            "Use the raw snapshot with its original RF context."
        )

    if selected == "phase_beamforming":
        # Validated production contract: legacy (M,) response vector with
        # the RF spatial carrier; the frozen result (true AoA -30 deg,
        # estimated AoA -30 deg, phase slope -22.5 deg/path) is preserved
        # bit-exactly -- no numerical drift is introduced here.
        data: Any = np.asarray(complex_responses, dtype=complex)
    elif selected in _SNAPSHOT_METHODS:
        data = if_snapshots
    else:  # pragma: no cover - registry rejects unknown names earlier
        raise ValueError(f"Unknown DOA estimator {selected!r}")

    doa_result = estimator.estimate(data, geometry, context)
    settings = {key: (tuple(float(v) for v in value) if isinstance(value, np.ndarray) else value)
                for key, value in kwargs.items()}
    doa_result = replace(doa_result, diagnostics={**doa_result.diagnostics,
                         'effective_settings': settings,
                         'settings_source': 'profile' if doa_options is not None else 'doa_config'})
    return doa_result, if_snapshots


def merge_doa_into_results(
    results: Dict[str, Any],
    doa_result: DOAResult,
    if_snapshots: Optional[IFSnapshotData] = None,
) -> Dict[str, Any]:
    """Merge the DOA dispatch output into the processed results dict.

    Exposes the structured ``doa_result`` contract and preserves every
    legacy key required downstream:

        estimated_aoa = doa_result.azimuth_deg   (selected estimator)
        angles, power = scan grid / spectrum when the estimator provides
                        one; None for search-free methods (esprit,
                        root_music) -- no fabricated spectrum.

    Returns the SAME results dict, updated in place.
    """
    results["doa_result"] = doa_result
    if if_snapshots is not None:
        results["if_snapshots"] = if_snapshots
    results["estimated_aoa"] = float(doa_result.azimuth_deg)
    results["angles"] = (
        np.asarray(doa_result.scan_grid, dtype=float)
        if doa_result.scan_grid is not None
        else None
    )
    results["power"] = (
        np.asarray(doa_result.spectrum, dtype=float)
        if doa_result.spectrum is not None
        else None
    )
    return results

