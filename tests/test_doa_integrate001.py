"""DOA-INTEGRATE-001 — production DOA method selection (synthetic, fast).

Exercises the orchestration layer itself (rydberg_aoa.simulation.doa_pipeline)
with synthetic production-like data ONLY.  SimulationRunner is NEVER
instantiated; no physics, ARC, Doppler or multiprocessing is executed.

Synthetic contract mirrors the validated production polarity:
time traces s_m(t) = A cos(2*pi*f_IF*t + phi_m) with the arrival phases
phi_m taken at the RF SIGNAL carrier produce complex IF snapshots
X[:, k] ~ A * exp(+j*phi_m) — the same data polarity the production FFT
extractor feeds the estimators (frozen six-path result: estimated AoA
-30 deg, phase slope -22.5 deg/path).
"""

import inspect

import numpy as np
import pytest

from rydberg_aoa.config import doa_config
from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import (
    ArrayGeometry,
    BartlettEstimator,
    ESPRITEstimator,
    IncompatibleGeometryError,
    MUSICEstimator,
    MVDREstimator,
    PhaseBeamformingEstimator,
    RootMUSICEstimator,
    arrival_phases,
    get_doa_estimator,
)
from rydberg_aoa.doa.estimators import DOAResult
from rydberg_aoa.simulation import doa_pipeline
from rydberg_aoa.simulation.doa_pipeline import (
    build_array_geometry,
    build_doa_context,
    estimate_doa_from_results,
    merge_doa_into_results,
    resolve_doa_method,
)

RF_HZ = params.RF_SIGNAL_FREQUENCY          # spatial carrier (132.6491 GHz)
IF_MHZ = params.IF_FREQ                     # temporal IF (0.1 MHz)
IF_HZ = IF_MHZ * 1.0e6
TRUE_AOA_DEG = params.TARGET_AOA_DEG        # -30.0
SPACING_M = params.SPACING                  # lambda_RF / 8

ALL_METHODS = (
    "phase_beamforming", "bartlett", "mvdr", "capon",
    "music", "esprit", "root_music",
)
SPECTRUM_METHODS = ("phase_beamforming", "bartlett", "mvdr", "capon", "music")
SEARCH_FREE_METHODS = ("esprit", "root_music")


# --- synthetic production-like data -----------------------------------------
def _production_positions(m, nonuniform=False):
    """Path positions exactly like the production runner builds them."""
    if not nonuniform:
        pos = np.zeros((m, 2))
        pos[:, 0] = np.arange(m) * SPACING_M
        return pos
    # Deterministic NON-uniform monotone layout (not a ULA, not
    # shift-invariant): strictly varying consecutive gaps.
    gaps = SPACING_M * (0.6 + 0.9 * np.linspace(0.0, 1.0, m - 1))
    pos = np.zeros((m, 2))
    pos[1:, 0] = np.cumsum(gaps)
    return pos


def _synthetic_run(positions_xy):
    """Synthetic per-path heterodyne traces carrying the RF arrival phase."""
    geometry = ArrayGeometry.from_xy(positions_xy)
    phases = arrival_phases(geometry, RF_HZ, TRUE_AOA_DEG)
    t_us = np.arange(0.0, 120.0 + 1.0e-9, 0.05)   # 12 IF cycles @ 0.1 MHz
    traces = [
        np.cos(2.0 * np.pi * IF_MHZ * t_us + phi) for phi in phases
    ]
    complex_responses = np.exp(1j * phases)
    return geometry, t_us, traces, complex_responses


def _run_dispatch(method, m=6, nonuniform=False, **kwargs):
    positions = _production_positions(m, nonuniform=nonuniform)
    geometry, t_us, traces, responses = _synthetic_run(positions)
    doa_result, if_snapshots = estimate_doa_from_results(
        array_positions=positions,
        complex_responses=responses,
        tlist_us=t_us,
        time_traces=traces,
        method=method,
        **kwargs,
    )
    return doa_result, if_snapshots, geometry


# --- A. registry default comes from doa_config (lazy) -----------------------
def test_a_registry_default_is_phase_beamforming():
    assert doa_config.DOA_METHOD == "phase_beamforming"
    estimator = get_doa_estimator()               # lazy config read
    assert isinstance(estimator, PhaseBeamformingEstimator)
    assert estimator.algorithm == "phase_beamforming"


def test_a_registry_reads_doa_config_lazily(monkeypatch):
    monkeypatch.setattr(doa_config, "DOA_METHOD", "music")
    estimator = get_doa_estimator()               # no import-time snapshot
    assert isinstance(estimator, MUSICEstimator)
    monkeypatch.setattr(doa_config, "DOA_METHOD", "root_music")
    assert isinstance(get_doa_estimator(), RootMUSICEstimator)


# --- B. explicit estimator overrides config ---------------------------------
def test_b_explicit_method_overrides_config():
    assert doa_config.DOA_METHOD == "phase_beamforming"
    estimator = get_doa_estimator("bartlett")
    assert isinstance(estimator, BartlettEstimator)
    assert resolve_doa_method("esprit") == "esprit"


# --- C. all seven names dispatch correctly ----------------------------------
@pytest.mark.parametrize("name,cls", [
    ("phase_beamforming", PhaseBeamformingEstimator),
    ("bartlett", BartlettEstimator),
    ("mvdr", MVDREstimator),
    ("capon", MVDREstimator),
    ("music", MUSICEstimator),
    ("esprit", ESPRITEstimator),
    ("root_music", RootMUSICEstimator),
])
def test_c_all_names_dispatch(name, cls):
    estimator = get_doa_estimator(name)
    assert isinstance(estimator, cls)
    assert resolve_doa_method(name) == name
    with pytest.raises(ValueError):
        resolve_doa_method("definitely_not_a_doa_method")

# --- D. same synthetic six-path input: every compatible method hits -30 -----
@pytest.mark.parametrize("method", ALL_METHODS)
def test_d_all_compatible_methods_estimate_minus_30(method):
    positions = _production_positions(6)
    geometry, t_us, traces, responses = _synthetic_run(positions)
    doa_result, if_snapshots = estimate_doa_from_results(
        array_positions=positions,
        complex_responses=responses,
        tlist_us=t_us,
        time_traces=traces,
        method=method,
    )
    assert doa_result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=1.0e-6)
    assert doa_result.elevation_deg == pytest.approx(
        doa_config.DOA_ELEVATION_DEG)
    # All covariance/structured estimators consumed the SAME X matrix.
    assert if_snapshots.n_paths == 6


# --- E. M is dynamic (never hard-coded) --------------------------------------
@pytest.mark.parametrize("m", [4, 6, 10])
def test_e_geometry_m_is_inferred_dynamically(m):
    positions = _production_positions(m)
    geometry = build_array_geometry(positions)
    assert geometry.m == m                       # inferred, not hard-coded
    doa_result, if_snapshots, _ = _run_dispatch("bartlett", m=m)
    assert doa_result.diagnostics["n_elements"] == m
    assert if_snapshots.X.shape[0] == m
    assert doa_result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=1.0e-6)
    doa_result_pf, _, _ = _run_dispatch("phase_beamforming", m=m)
    assert doa_result_pf.azimuth_deg == pytest.approx(TRUE_AOA_DEG)


# --- F. MUSIC works on nonuniform geometry -----------------------------------
def test_f_music_on_nonuniform_geometry_works():
    doa_result, _, geometry = _run_dispatch("music", m=6, nonuniform=True)
    assert not geometry.capabilities["ula"]
    assert doa_result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=1.0e-6)


# --- G. ESPRIT fails loudly on non-shift-invariant geometry ------------------
def test_g_esprit_on_nonuniform_geometry_fails_loudly():
    with pytest.raises(IncompatibleGeometryError):
        _run_dispatch("esprit", m=6, nonuniform=True)


# --- H. Root-MUSIC fails loudly on non-ULA geometry --------------------------
def test_h_root_music_on_non_ula_fails_loudly():
    with pytest.raises(IncompatibleGeometryError):
        _run_dispatch("root_music", m=6, nonuniform=True)

# --- I. spatial carrier is the RF carrier, never the IF ----------------------
def test_i_orchestration_uses_rf_signal_carrier():
    context = build_doa_context()
    assert context["frequency_hz"] == RF_HZ
    assert context["frequency_hz"] != IF_HZ
    for method in ("phase_beamforming", "bartlett", "music", "esprit"):
        doa_result, _, _ = _run_dispatch(method)
        assert doa_result.diagnostics["frequency_hz"] == RF_HZ


def test_i_if_carrier_is_demonstrably_wrong():
    positions = _production_positions(6)
    geometry, _t, _tr, responses = _synthetic_run(positions)
    estimator = PhaseBeamformingEstimator()
    result_if = estimator.estimate(
        responses, geometry, {"frequency_hz": IF_HZ})
    # The IF beat note carries no resolvable spatial phase: steering with
    # it does NOT recover the true arrival (demonstrably wrong estimate).
    assert abs(result_if.azimuth_deg - TRUE_AOA_DEG) > 10.0


# --- J. snapshot extractor invoked exactly once per orchestration call -------
@pytest.mark.parametrize("method", ["phase_beamforming", "bartlett"])
def test_j_extract_if_snapshots_called_once_per_dispatch(
        monkeypatch, method):
    calls = {"count": 0}
    real_extractor = doa_pipeline.extract_if_snapshots

    def counting_extractor(*args, **kwargs):
        calls["count"] += 1
        return real_extractor(*args, **kwargs)

    monkeypatch.setattr(doa_pipeline, "extract_if_snapshots",
                        counting_extractor)
    positions = _production_positions(6)
    geometry, t_us, traces, responses = _synthetic_run(positions)
    estimate_doa_from_results(
        array_positions=positions,
        complex_responses=responses,
        tlist_us=t_us,
        time_traces=traces,
        method=method,
    )
    assert calls["count"] == 1

    # Pre-extracted snapshots are reused as-is (never re-extracted).
    snapshots = real_extractor(t_us, traces, IF_MHZ, start_us=0.0)
    calls["count"] = 0
    estimate_doa_from_results(
        array_positions=positions,
        complex_responses=responses,
        tlist_us=t_us,
        time_traces=traces,
        method=method,
        if_snapshots=snapshots,
    )
    assert calls["count"] == 0


# --- K. legacy estimated_aoa key contract ------------------------------------
def test_k_legacy_estimated_aoa_key_remains_available():
    doa_result, _, _ = _run_dispatch("phase_beamforming")
    results = {}
    merge_doa_into_results(results, doa_result)
    assert results["estimated_aoa"] == float(doa_result.azimuth_deg)
# --- L. search-free methods never fabricate a spectrum -----------------------
@pytest.mark.parametrize("method", SEARCH_FREE_METHODS)
def test_l_search_free_methods_have_no_spectrum(method):
    doa_result, _, _ = _run_dispatch(method)
    assert doa_result.spectrum is None
    assert doa_result.scan_grid is None


# --- M. spectrum methods expose spectrum and scan grid -----------------------
@pytest.mark.parametrize("method", SPECTRUM_METHODS)
def test_m_spectrum_methods_expose_spectrum_and_grid(method):
    doa_result, _, _ = _run_dispatch(method)
    assert doa_result.spectrum is not None
    assert doa_result.scan_grid is not None
    assert doa_result.spectrum.shape == doa_result.scan_grid.shape


# --- N. no hard-coded element count in the integration code ------------------
def test_n_no_hard_coded_element_count_in_pipeline():
    source = inspect.getsource(doa_pipeline)
    assert "N_ELEMENTS" not in source
    # Behavior proof: M=4 and M=10 flow through the same code path.
    for m in (4, 10):
        doa_result, _, geometry = _run_dispatch("music", m=m)
        assert geometry.m == m
        assert doa_result.diagnostics["n_elements"] == m


def test_n_geometry_from_real_positions_not_from_config():
    # A (M, 3) position array passes through unchanged; (M, 2) gains z = 0.
    pos2 = _production_positions(5)
    geometry = build_array_geometry(np.column_stack([pos2, np.zeros(5)]))
    assert geometry.m == 5
    np.testing.assert_array_equal(geometry.positions_m[:, :2], pos2)
    with pytest.raises(ValueError):
        build_array_geometry(np.zeros((5, 4)))


def test_k_merge_preserves_true_aoa_and_snapshots():
    doa_result, if_snapshots, _ = _run_dispatch("bartlett")
    results = {"true_aoa": TRUE_AOA_DEG}
    merge_doa_into_results(results, doa_result, if_snapshots)
    assert results["true_aoa"] == TRUE_AOA_DEG
    assert results["if_snapshots"] is if_snapshots
    assert results["estimated_aoa"] == pytest.approx(TRUE_AOA_DEG, abs=1e-6)
