"""Complex IF snapshot data model for narrowband DOA processing (DOA-DATA-001).

Converts per-path heterodyne time-domain traces into complex narrowband
snapshots at the known intermediate frequency (IF) so that subspace DOA
estimators (MUSIC / MVDR / ESPRIT) can operate on a single
common matrix.

Units and phase convention (traced from the production chain):
  * time   : microseconds (us)            -- tlist from the time-domain worker
  * f_IF   : signed MHz                   -- params.IF_FREQ (== cycles/us)
  * phase  : 2*pi * f_IF_MHz * t_us       -- NO extra mega factor (no 1e6 scale)
  * sign   : exp(-j * 2*pi * f_IF * t)    -- identical to the numpy FFT
    forward transform. The production extractor now fits the exact IF with
    phase relative to its first retained sample; snapshots project using
    absolute sample times. For a real observable s(t) = A*cos(2*pi*f*t + phi) the
    coherent projection returns Z = A * exp(+j*phi) over complete cycles.
    The production extractor's local phase origin adds a common phase
    factor exp(+j*2*pi*f*t_first). Rounded windows that do not span complete
    cycles can still bias this projection; its formula is unchanged here.

Window model:
  * consecutive, disjoint full windows of `cycles_per_snapshot` IF cycles;
    window lengths use abs(f_IF), while the projection keeps its sign
  * partial trailing windows are NEVER used (dropped and counted)
  * windows with too few samples are rejected with ValueError

The layer is deliberately DOA-agnostic: no array geometry, no steering
grid, no MUSIC/MVDR/ESPRIT knowledge.  It only converts per-path time data
into snapshots.  M (number of paths) is always inferred from the input and
is never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

__all__ = ["IFSnapshotData", "extract_if_snapshots", "sample_covariance"]

_TWO_PI = 2.0 * np.pi

ArrayLike1D = Union[np.ndarray, Sequence[float], Sequence[complex]]


def _finite_real_scalar(value, name, requirement="finite") -> float:
    """Validate a Python/NumPy real scalar before any coercion."""
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a real numeric scalar; got {value!r}")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be {requirement}; got {value!r}")
    return result


def _positive_integer(value, name) -> int:
    """Reject fractional counts, bools and conversion-only objects."""
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1; got {result!r}")
    return result


def _finite_numeric_array(value, name, *, real=False) -> np.ndarray:
    """Preserve complex data, reject nonnumeric/complex-time coercions."""
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    supported = (np.issubdtype(arr.dtype, np.integer) or
                 np.issubdtype(arr.dtype, np.floating) or
                 np.issubdtype(arr.dtype, np.complexfloating))
    if not supported or (real and np.iscomplexobj(arr)):
        kind = "real numeric" if real else "numeric"
        raise ValueError(f"{name} must contain {kind} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    try:
        with np.errstate(over="raise", invalid="raise"):
            arr = arr.astype(float if real else complex, copy=False)
    except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} cannot be represented as finite numeric values") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite converted values")
    return arr


def _production_transient_delay_us() -> float:
    """Production transient cut (ATOMIC_TRANSIENT_DELAY_US), resolved lazily.

    Kept behind a function so this module has no import-time configuration
    dependency (same pattern as doa/registry.py).
    """
    from ..config.parameters import ATOMIC_TRANSIENT_DELAY_US

    return ATOMIC_TRANSIENT_DELAY_US


@dataclass(frozen=True)
class IFSnapshotData:
    """Immutable complex IF snapshot block.

    Canonical orientation is X.shape == (M, K) with:
        M = number of laser paths (rows)
        K = number of temporal snapshots (columns)

    Attributes:
        X: complex snapshot matrix, shape (M, K).
        time_centers_us: mean sample time of each window, shape (K,), in us.
        if_frequency_mhz: Signed nonzero IF used for the projection, in MHz.
        n_paths: M == X.shape[0].
        window_duration_us: span between the first and the last sample of a
            window: (samples_per_window - 1) * dt_us.
        samples_per_window: integer sample count per window.
        diagnostics: extraction metadata (selection window, grid step,
            windows used / dropped).
        normalization_polarity: None for raw IF projection. After the RF-to-AoA
            boundary, records the applied atomic polarity (-1 or +1) so the
            same data can be reused without applying that conversion twice.
            This is provenance, not a choice of RF energy direction.
    """

    X: np.ndarray
    time_centers_us: np.ndarray
    if_frequency_mhz: float
    n_paths: int
    window_duration_us: float
    samples_per_window: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    normalization_polarity: Optional[int] = None

    def __post_init__(self) -> None:
        x = _finite_numeric_array(self.X, "X")
        if x.ndim != 2:
            raise ValueError(
                "X must be a 2-D complex array of shape (M, K); "
                f"got shape {x.shape}"
            )
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("X must contain at least one path and one snapshot")
        centers = _finite_numeric_array(self.time_centers_us, "time_centers_us", real=True)
        if centers.ndim != 1 or centers.shape[0] != x.shape[1]:
            raise ValueError(
                "time_centers_us must be a 1-D array with one center per "
                f"snapshot (K={x.shape[1]}); got shape {centers.shape}"
            )
        freq = _finite_real_scalar(self.if_frequency_mhz, "if_frequency_mhz", "nonzero finite")
        if freq == 0.0:
            raise ValueError(
                f"if_frequency_mhz must be a nonzero finite value; got {freq!r}"
            )
        duration = _finite_real_scalar(self.window_duration_us, "window_duration_us")
        if duration <= 0.0:
            raise ValueError(
                "window_duration_us must be a positive finite value; "
                f"got {duration!r}"
            )
        spw = _positive_integer(self.samples_per_window, "samples_per_window")
        if _positive_integer(self.n_paths, "n_paths") != x.shape[0]:
            raise ValueError(
                f"n_paths must equal X.shape[0] ({x.shape[0]}); got {self.n_paths!r}"
            )
        polarity = self.normalization_polarity
        if polarity is not None:
            if (isinstance(polarity, (bool, np.bool_)) or
                    not isinstance(polarity, (int, np.integer)) or
                    polarity not in (-1, 1)):
                raise ValueError("normalization_polarity must be None, -1 or +1")
            polarity = int(polarity)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "time_centers_us", centers)
        object.__setattr__(self, "if_frequency_mhz", freq)
        object.__setattr__(self, "n_paths", int(x.shape[0]))
        object.__setattr__(self, "window_duration_us", duration)
        object.__setattr__(self, "samples_per_window", spw)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "normalization_polarity", polarity)

    @property
    def n_snapshots(self) -> int:
        """K == X.shape[1]: number of temporal snapshots (columns)."""
        return int(self.X.shape[1])


def extract_if_snapshots(
    t_us: ArrayLike1D,
    time_traces: Sequence[ArrayLike1D],
    if_frequency_mhz: float,
    *,
    start_us: Optional[float] = None,
    stop_us: Optional[float] = None,
    cycles_per_snapshot: float = 2.0,
    remove_dc: bool = True,
    min_samples: int = 8,
) -> IFSnapshotData:
    """Extract complex IF snapshots from per-path heterodyne time traces.

    Each snapshot is the coherent signed f_IF projection of one consecutive time
    window:

        z_m,k = (2/N) * sum_n [s_m(t_n) - mean_window(s_m)]
                       * exp(-j * 2*pi * f_IF_MHz * t_n_us)

    This uses the Fourier sign convention and is exact for windows spanning
    an integer number of cycles. Rounded window lengths can introduce
    projection bias; this function does not fit or correct that bias.

    Args:
        t_us: 1-D uniform sample times in microseconds (strictly increasing).
        time_traces: sequence of M 1-D traces (one per laser path), each the
            same length as t_us.  Real traces are the production case
            (Im<sigma21>(t)); complex traces are accepted unchanged.
        if_frequency_mhz: Signed nonzero finite IF in MHz (== cycles/us).
            The sign is retained in the projection and output metadata; only
            window lengths use its magnitude. Zero IF requires a separate DC
            measurement contract and is rejected.
        start_us: first sample time to use; None reuses the production
            transient cut (ATOMIC_TRANSIENT_DELAY_US = 40.0 us).
        stop_us: last sample time to use (inclusive); None uses the final
            sample of t_us.  Partial trailing windows are dropped, never
            used.
        cycles_per_snapshot: IF cycles per snapshot window (default 2).
        remove_dc: subtract each window mean before projecting (True).
        min_samples: minimum samples per window; anything shorter raises.

    Returns:
        IFSnapshotData with X of shape (M, K), K >= 1.

    Raises:
        ValueError: for non-finite / non-uniform / non-increasing time
            grids, length-mismatched traces, non-finite traces, invalid
            scalar parameters, IF outside Nyquist, projection overflow, or
            records too short to form one full window.
    """
    # --- time grid ---------------------------------------------------------
    t = _finite_numeric_array(t_us, "t_us", real=True)
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError(
            "t_us must be a 1-D array of at least two sample times (us); "
            f"got shape {np.shape(t_us)}"
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            step = np.diff(t)
    except FloatingPointError as exc:
        raise ValueError("t_us must have finite sample steps") from exc
    if not np.all(np.isfinite(step)):
        raise ValueError("t_us must have finite sample steps")
    if np.any(step <= 0.0):
        raise ValueError("t_us must be strictly increasing (microseconds)")
    dt = float(step[0])
    if not np.allclose(step, dt, rtol=1.0e-7, atol=0.0):
        raise ValueError("t_us must be uniformly sampled (constant dt)")

    # --- traces ------------------------------------------------------------
    traces_list = list(time_traces)
    if len(traces_list) < 1:
        raise ValueError("time_traces must contain at least one path trace")
    m = len(traces_list)
    n_grid = int(t.shape[0])
    s = np.empty((m, n_grid), dtype=complex)
    for row, trace in enumerate(traces_list):
        arr = _finite_numeric_array(trace, f"trace {row}")
        if arr.ndim != 1 or arr.shape[0] != n_grid:
            raise ValueError(
                f"trace {row} must be a 1-D array with the same length as "
                f"t_us ({n_grid}); got shape {arr.shape}"
            )
        s[row] = arr

    # --- scalar parameters --------------------------------------------------
    f_if = _finite_real_scalar(if_frequency_mhz, "if_frequency_mhz", "nonzero finite")
    if f_if == 0.0:
        raise ValueError(
            f"if_frequency_mhz must be a nonzero finite value; got {f_if!r}"
        )
    if abs(f_if) >= 0.5 / dt:
        raise ValueError("if_frequency_mhz must lie strictly inside the Nyquist interval")
    cycles = _finite_real_scalar(cycles_per_snapshot, "cycles_per_snapshot")
    if cycles <= 0.0:
        raise ValueError(
            f"cycles_per_snapshot must be a positive finite value; got {cycles!r}"
        )
    min_spw = _positive_integer(min_samples, "min_samples")
    if not isinstance(remove_dc, (bool, np.bool_)):
        raise ValueError("remove_dc must be a boolean")

    # --- selection window ---------------------------------------------------
    if start_us is None:
        start_us = _production_transient_delay_us()
    start = _finite_real_scalar(start_us, "start_us")
    stop = float(t[-1]) if stop_us is None else _finite_real_scalar(stop_us, "stop_us")
    if stop < start:
        raise ValueError(f"stop_us ({stop:g}) must be >= start_us ({start:g})")

    n0 = int(np.searchsorted(t, start, side="left"))
    n1 = int(np.searchsorted(t, stop, side="right"))
    n_avail = n1 - n0
    if n_avail < 1:
        raise ValueError(
            f"No samples in selected window [{start:g}, {stop:g}] us on the "
            "given time grid"
        )
    # --- window geometry -----------------------------------------------------
    # Signed f in MHz == cycles/us; the positive period is 1/abs(f_if) us.
    period_us = 1.0 / abs(f_if)
    samples_per_cycle = period_us / dt
    spw_float = cycles * samples_per_cycle
    if not np.isfinite(samples_per_cycle) or not np.isfinite(spw_float):
        raise ValueError("Snapshot window geometry must be finite")
    if spw_float < min_spw:
        raise ValueError(
            "Snapshot windows too short: at this sampling density "
            f"({samples_per_cycle:.3f} samples per IF cycle) a "
            f"{cycles:g}-cycle window has only {spw_float:.3f} samples; "
            f"minimum is {min_spw}. Reduce cycles_per_snapshot or increase "
            "the sampling rate."
        )
    spw = int(np.floor(spw_float + 0.5))  # nearest integer sample count
    if spw < min_spw:
        raise ValueError(
            f"Snapshot window would have {spw} samples (< min_samples={min_spw})"
        )

    n_windows = n_avail // spw
    if n_windows < 1:
        raise ValueError(
            f"Too few samples in the selected window ({n_avail}) to form one "
            f"full {spw}-sample snapshot window; increase the record length "
            "or reduce cycles_per_snapshot."
        )
    n_used = n_windows * spw
    dropped = n_avail - n_used

    # --- coherent projection --------------------------------------------------
    block = s[:, n0 : n0 + n_used]                      # (M, K*W)
    t_block = t[n0 : n0 + n_used]
    windows_t = t_block.reshape(n_windows, spw)         # (K, W)
    s_win = block.reshape(m, n_windows, spw)            # (M, K, W)
    try:
        with np.errstate(over="raise", invalid="raise"):
            centers = windows_t.mean(axis=1)           # (K,)
            if remove_dc:
                s_win = s_win - s_win.mean(axis=2, keepdims=True)
            osc = np.exp(-1j * _TWO_PI * f_if * windows_t)  # (K, W)
            x_snap = (2.0 / spw) * (s_win * osc[None, :, :]).sum(axis=2)  # (M, K)
    except FloatingPointError as exc:
        raise ValueError("IF snapshot projection overflow or invalid arithmetic") from exc
    if not np.all(np.isfinite(x_snap)) or not np.all(np.isfinite(centers)):
        raise ValueError("IF snapshot projection and time centers must be finite")

    diagnostics = {
        "start_us": start,
        "stop_us": stop,
        "dt_us": dt,
        "n_samples_selected": n_avail,
        "n_samples_used": n_used,
        "trailing_samples_dropped": dropped,
        "cycles_per_snapshot": cycles,
        "samples_per_if_cycle": samples_per_cycle,
        "remove_dc": bool(remove_dc),
    }
    return IFSnapshotData(
        X=x_snap,
        time_centers_us=centers,
        if_frequency_mhz=f_if,
        n_paths=m,
        window_duration_us=float(spw - 1) * dt,
        samples_per_window=spw,
        diagnostics=diagnostics,
    )


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Unregularized sample covariance of the snapshot matrix.

    For X of shape (M, K):

        Rxx = X @ X.conj().T / K

    Returns a complex (M, M) array that is Hermitian to numerical precision
    with finite entries.  No diagonal loading and no regularization are
    applied here; consumers may whiten/load later without changing this
    contract.

    Args:
        X: complex snapshot matrix, shape (M, K), K >= 1.

    Returns:
        sample covariance, shape (M, M).

    Raises:
        ValueError: when X is not a finite numeric non-empty 2-D array, or
            covariance arithmetic overflows or produces non-finite values.
    """
    arr = _finite_numeric_array(X, "X")
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            "X must be a non-empty 2-D complex array of shape (M, K); "
            f"got shape {arr.shape}"
        )
    k = arr.shape[1]
    try:
        with np.errstate(over="raise", invalid="raise"):
            covariance = arr @ arr.conj().T / k
    except FloatingPointError as exc:
        raise ValueError("Covariance overflow or invalid arithmetic") from exc
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance output must be finite")
    return covariance
