"""Deterministic NumPy covariance DOA estimators (DOA-ALG-001).

Three selectable narrowband estimators on the validated ArrayGeometry /
steering_matrix / IFSnapshotData X(M, K) stack:

    BartlettEstimator  P_B(theta)     = m(theta)^H Rxx m(theta)
    MVDREstimator      P_MVDR(theta)  = 1 / (m(theta)^H R_loaded^-1 m(theta))
    MUSICEstimator     P_MUSIC(theta) = 1 / (m(theta)^H En En^H m(theta))

Sign / steering convention (no second steering source is introduced):

    * the spatial carrier is the RF signal frequency (context frequency_hz);
    * steering rows come only from doa.steering.steering_matrix;
    * the estimator manifold is the CONJUGATE steering row,
      m(theta) = conj(s(theta)) with s(theta) = exp(-j phi(theta)).

The conjugate manifold is required for parity with the validated
production data polarity x = exp(+j phi): the covariance estimators use
the Hermitian form m^H R m, while the legacy delay-and-sum beamformer
uses the transpose form |s^T x|^2.  For x = exp(+j phi) both forms peak
at the true arrival azimuth; using s directly inside m^H R m would mirror
the covariance spectra to the wrong sign.  A read-only probe verified
phase_beamforming / bartlett / MUSIC / MVDR agreement at -30 deg.

Geometry capability is arbitrary for all three estimators (no ULA
assumption, no hard-coded element count).  No printing occurs inside the
estimator cores; per-estimate metadata is stored in DOAResult.diagnostics.
Only NumPy is used (no ARLpy production backend, no randomness, no
multiprocessing, no physics changes).
"""
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
from .estimators import (
    DEFAULT_AZIMUTH_SCAN_DEG,
    DOAError,
    DOAEstimator,
    DOAResult,
    InvalidContextError,
)
from .geometry import ArrayGeometry
from .snapshots import IFSnapshotData, sample_covariance
from .steering import steering_matrix
__all__ = [
    "SnapshotDOAEstimator",
    "BartlettEstimator",
    "MVDREstimator",
    "MUSICEstimator",
]


def _resolve_frequency_hz(context: Optional[Mapping[str, Any]]) -> float:
    """Extract the RF spatial-carrier frequency from the estimate context."""
    ctx: Dict[str, Any] = dict(context) if context else {}
    frequency_hz = ctx.get("frequency_hz")
    if frequency_hz is None:
        raise InvalidContextError(
            "context must provide 'frequency_hz' (RF signal carrier, Hz). "
            "The heterodyne IF frequency is not a spatial steering frequency."
        )
    freq = float(frequency_hz)
    if not np.isfinite(freq) or freq <= 0.0:
        raise InvalidContextError(
            "context 'frequency_hz' must be a positive finite value; "
            f"got {frequency_hz!r}"
        )
    return freq


def _resolve_scan_grid(scan_azimuth_deg) -> np.ndarray:
    """Validate the shared azimuth scan contract (1-D grid, configurable)."""
    if scan_azimuth_deg is None:
        return np.array(DEFAULT_AZIMUTH_SCAN_DEG, dtype=float, copy=True)
    grid = np.asarray(scan_azimuth_deg, dtype=float)
    if grid.ndim != 1 or grid.shape[0] < 1:
        raise ValueError("scan_azimuth_deg must be a non-empty 1-D grid")
    if not np.all(np.isfinite(grid)):
        raise ValueError("scan_azimuth_deg contains non-finite values")
    return grid.copy()


def _as_snapshot_matrix(data) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Coerce estimator input to the canonical (M, K) snapshot matrix."""
    if isinstance(data, IFSnapshotData):
        mat = np.asarray(data.X, dtype=complex)
        meta: Dict[str, Any] = {
            "input_kind": "IFSnapshotData",
            "if_frequency_mhz": float(data.if_frequency_mhz),
        }
        return mat, meta
    arr = np.asarray(data, dtype=complex)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1), {"input_kind": "vector_1d"}
    if arr.ndim == 2:
        return arr, {"input_kind": "snapshot_matrix"}
    raise DOAError(
        "covariance DOA estimators require snapshot data of shape (M, K) "
        f"(or a single (M,) response vector); got shape {arr.shape}"
    )


def _manifold_matrix(positions_m, frequency_hz, grid, elevation_deg):
    """Return the (G, M) conjugate-manifold matrix conj(steering_matrix)."""
    steer = steering_matrix(positions_m, frequency_hz, grid, elevation_deg)
    return np.conjugate(steer)


def _normalize_for_display(raw: np.ndarray) -> np.ndarray:
    """Scale a non-negative spectrum by its peak for presentation only."""
    shown = np.asarray(raw, dtype=float)
    if shown.size == 0:
        return shown.copy()
    peak = float(np.max(shown))
    if not np.isfinite(peak) or peak <= 0.0:
        return shown.copy()
    return shown / peak


def _covariance_diagnostics(rxx: np.ndarray) -> Dict[str, Any]:
    """Deterministic rank / conditioning summary of a covariance matrix."""
    rank = int(np.linalg.matrix_rank(rxx))
    try:
        cond = float(np.linalg.cond(rxx))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return {"covariance_rank": rank, "covariance_condition_number": cond}

class SnapshotDOAEstimator(DOAEstimator):
    """Shared input contract for covariance (snapshot-matrix) estimators.

    Accepts IFSnapshotData or a complex (M, K) snapshot matrix (a single
    (M,) response vector is treated as K == 1).  M is always inferred from
    the data and checked against geometry.m; no element count is fixed
    here.  The azimuth grid defaults to the shared production scan but any
    1-D grid resolution may be supplied.
    """

    algorithm = "snapshot_base"
    required_capabilities = frozenset({"arbitrary"})

    def __init__(self, scan_azimuth_deg=None, elevation_deg: float = 0.0):
        self.scan_azimuth_deg = _resolve_scan_grid(scan_azimuth_deg)
        self.elevation_deg = float(elevation_deg)
        if not np.isfinite(self.elevation_deg):
            raise ValueError(f"elevation_deg must be finite; got {elevation_deg!r}")

    def supports(self, geometry, data) -> bool:
        if not isinstance(geometry, ArrayGeometry):
            return False
        try:
            mat, _ = _as_snapshot_matrix(data)
        except (DOAError, ValueError, TypeError):
            return False
        if mat.shape[0] != geometry.m or mat.shape[0] < 1 or mat.shape[1] < 1:
            return False
        if not np.all(np.isfinite(mat)):
            return False
        caps = geometry.capabilities
        return all(caps.get(key, False) for key in self.required_capabilities)

    def _require_supported(self, geometry, data):
        if not isinstance(geometry, ArrayGeometry):
            raise DOAError(
                f"estimator {self.algorithm!r} requires an ArrayGeometry; "
                f"got {type(geometry).__name__}"
            )
        try:
            mat, _ = _as_snapshot_matrix(data)
        except DOAError:
            raise
        except (ValueError, TypeError) as exc:
            raise DOAError(
                f"estimator {self.algorithm!r} requires snapshot data of shape "
                f"(M, K) with M == geometry.m ({geometry.m}); got {exc}"
            ) from exc
        if mat.ndim != 2 or mat.shape[0] != geometry.m:
            raise DOAError(
                f"estimator {self.algorithm!r} requires snapshot data with "
                f"{geometry.m} rows (geometry.m); got shape {mat.shape}"
            )
        if mat.shape[1] < 1:
            raise DOAError("snapshot data must contain at least one snapshot (K >= 1)")
        if not np.all(np.isfinite(mat)):
            raise DOAError("snapshot data contains non-finite values")
        from .estimators import IncompatibleGeometryError
        missing = [
            key for key in self.required_capabilities
            if not geometry.capabilities.get(key, False)
        ]
        if missing:
            raise IncompatibleGeometryError(
                self.algorithm, self.required_capabilities, missing)
        return np.asarray(mat, dtype=complex)

    def estimate(self, data, geometry, context=None):
        mat = self._require_supported(geometry, data)
        return self._estimate_impl(mat, geometry, context)

    def _estimate_impl(self, data, geometry, context=None):
        raise NotImplementedError

class BartlettEstimator(SnapshotDOAEstimator):
    """Conventional (Bartlett) beamformer on the sample covariance.

    P_B(theta) = m(theta)^H Rxx m(theta), evaluated vectorially over the
    scan grid.  The plotted spectrum is normalized by its peak purely for
    presentation; the raw power is retained in diagnostics.
    """

    algorithm = "bartlett"

    def _estimate_impl(self, data, geometry, context=None):
        frequency_hz = _resolve_frequency_hz(context)
        mat = np.asarray(data, dtype=complex)
        n_elem, n_snap = int(mat.shape[0]), int(mat.shape[1])
        rxx = sample_covariance(mat)
        grid = self.scan_azimuth_deg
        manifold = _manifold_matrix(
            geometry.positions_m, frequency_hz, grid, self.elevation_deg)
        raw = np.einsum("gm,mn,gn->g", manifold.conj(), rxx, manifold).real
        raw = np.maximum(raw, 0.0)
        shown = _normalize_for_display(raw)
        peak_index = int(np.argmax(shown))
        diag = _covariance_diagnostics(rxx)
        diag.update({
            "frequency_hz": frequency_hz,
            "elevation_deg": self.elevation_deg,
            "n_elements": n_elem,
            "n_snapshots": n_snap,
            "peak_index": peak_index,
            "raw_spectrum": np.asarray(raw, dtype=float),
            "capabilities": dict(geometry.capabilities),
        })
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(grid[peak_index]),
            elevation_deg=self.elevation_deg,
            spectrum=shown,
            scan_grid=np.array(grid, dtype=float, copy=True),
            diagnostics=diag,
        )
class MUSICEstimator(SnapshotDOAEstimator):
    """MUSIC pseudo-spectrum with deterministic Hermitian eigendecomposition.

    P_MUSIC(theta) = 1 / (m(theta)^H En En^H m(theta)), where En holds the
    noise-subspace eigenvectors of Rxx.  No random covariance perturbation
    and no synthetic noise injection are applied; only a deterministic
    machine-precision floor guards the divide-by-zero.
    """

    algorithm = "music"

    def __init__(self, scan_azimuth_deg=None, elevation_deg: float = 0.0,
                 n_sources: int = 1):
        super().__init__(scan_azimuth_deg=scan_azimuth_deg,
                         elevation_deg=elevation_deg)
        self.n_sources = self._validate_n_sources(n_sources)

    @staticmethod
    def _validate_n_sources(n_sources) -> int:
        try:
            as_float = float(n_sources)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "n_sources must be an integer with 1 <= n_sources < M; "
                f"got {n_sources!r}"
            ) from exc
        n = int(as_float)
        if not np.isfinite(as_float) or float(n) != as_float:
            raise ValueError(
                "n_sources must be an integer with 1 <= n_sources < M; "
                f"got {n_sources!r}"
            )
        if n < 1:
            raise ValueError(
                f"n_sources must satisfy 1 <= n_sources < M; got {n_sources!r}"
            )
        return n

    def _estimate_impl(self, data, geometry, context=None):
        frequency_hz = _resolve_frequency_hz(context)
        mat = np.asarray(data, dtype=complex)
        n_elem, n_snap = int(mat.shape[0]), int(mat.shape[1])
        n_src = int(self.n_sources)
        if not 1 <= n_src < n_elem:
            raise ValueError(
                f"n_sources must satisfy 1 <= n_sources < M (M={n_elem}); "
                f"got {n_src}"
            )
        rxx = sample_covariance(mat)
        eigenvals, eigenvecs = np.linalg.eigh(rxx)
        order = np.argsort(eigenvals)[::-1]
        eigenvals = np.asarray(eigenvals[order], dtype=float)
        eigenvecs = np.asarray(eigenvecs[:, order])
        noise_space = eigenvecs[:, n_src:]
        projector = noise_space @ noise_space.conj().T
        grid = self.scan_azimuth_deg
        manifold = _manifold_matrix(
            geometry.positions_m, frequency_hz, grid, self.elevation_deg)
        denom = np.einsum(
            "gm,mn,gn->g", manifold.conj(), projector, manifold).real
        neg_tol = 1.0e-12 * max(
            1.0, float(np.max(np.abs(denom))) if denom.size else 1.0)
        if np.any(denom < -neg_tol):
            raise DOAError(
                "MUSIC noise-projection denominator has significant negative "
                f"values (min {float(np.min(denom)):.3e}); failing loudly "
                "instead of returning a mirrored spectrum."
            )
        denom = np.maximum(denom, 0.0)
        floor = float(np.finfo(float).eps) * max(float(np.max(denom)), 1.0)
        raw = 1.0 / np.maximum(denom, floor)
        shown = _normalize_for_display(raw)
        peak_index = int(np.argmax(shown))
        diag = _covariance_diagnostics(rxx)
        diag.update({
            "frequency_hz": frequency_hz,
            "elevation_deg": self.elevation_deg,
            "n_elements": n_elem,
            "n_snapshots": n_snap,
            "n_sources": n_src,
            "eigenvalues": np.asarray(eigenvals, dtype=float),
            "noise_floor": floor,
            "peak_index": peak_index,
            "raw_spectrum": np.asarray(raw, dtype=float),
            "capabilities": dict(geometry.capabilities),
        })
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(grid[peak_index]),
            elevation_deg=self.elevation_deg,
            spectrum=shown,
            scan_grid=np.array(grid, dtype=float, copy=True),
            diagnostics=diag,
        )

class MVDREstimator(SnapshotDOAEstimator):
    """MVDR (Capon) estimator with deterministic diagonal loading.

    P_MVDR(theta) = 1 / (m(theta)^H R_loaded^-1 m(theta)) with
        R_loaded = Rxx + loading_rel * (real(trace(Rxx)) / M) * I.

    The noiseless single-source covariance is singular by construction, so
    the raw inverse is never used blindly.  loading_rel is an algorithmic
    numerical-regularization parameter, NOT physical noise; no random
    jitter is ever added.  Zero / non-finite trace fails loudly.
    """

    algorithm = "mvdr"

    def __init__(self, scan_azimuth_deg=None, elevation_deg: float = 0.0,
                 loading_rel: float = 10.0 ** (-1.0 - 1.0 - 1.0 - 1.0 - 1.0 - 1.0)):
        super().__init__(scan_azimuth_deg=scan_azimuth_deg,
                         elevation_deg=elevation_deg)
        self.loading_rel = self._validate_loading(loading_rel)

    @staticmethod
    def _validate_loading(loading_rel) -> float:
        try:
            value = float(loading_rel)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"loading_rel must be a positive finite value; got {loading_rel!r}"
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"loading_rel must be a positive finite value; got {loading_rel!r}"
            )
        return value

    def _estimate_impl(self, data, geometry, context=None):
        frequency_hz = _resolve_frequency_hz(context)
        mat = np.asarray(data, dtype=complex)
        n_elem, n_snap = int(mat.shape[0]), int(mat.shape[1])
        rxx = sample_covariance(mat)
        rank_before = int(np.linalg.matrix_rank(rxx))
        try:
            cond_before = float(np.linalg.cond(rxx))
        except np.linalg.LinAlgError:
            cond_before = float("inf")
        trace_real = float(np.real(np.trace(rxx)))
        if not np.isfinite(trace_real) or trace_real <= 0.0:
            raise DOAError(
                "MVDR diagonal loading requires real(trace(Rxx)) to be a "
                f"positive finite value; got {trace_real!r}. Failing loudly "
                "instead of inverting a degenerate covariance."
            )
        scale = trace_real / n_elem
        loading_abs = float(self.loading_rel) * scale
        loaded = rxx + loading_abs * np.eye(n_elem, dtype=complex)
        try:
            cond_after = float(np.linalg.cond(loaded))
        except np.linalg.LinAlgError:
            cond_after = float("inf")
        grid = self.scan_azimuth_deg
        manifold = _manifold_matrix(
            geometry.positions_m, frequency_hz, grid, self.elevation_deg)
        # Solve R_loaded Y = M^T once (columns are m(theta)); the MVDR
        # denominator is then m(theta)^H y(theta) for every grid point.
        solved = np.linalg.solve(loaded, manifold.T)
        denom = np.einsum("gm,mg->g", manifold.conj(), solved).real
        pos_tol = 1.0e-9 * max(1.0, float(np.max(np.abs(denom))) if denom.size else 1.0)
        if np.any(denom <= pos_tol):
            raise DOAError(
                "MVDR denominator is non-positive within tolerance "
                f"(min {float(np.min(denom)):.3e}); failing loudly instead "
                "of returning an unphysical spectrum."
            )
        raw = 1.0 / denom
        shown = _normalize_for_display(raw)
        peak_index = int(np.argmax(shown))
        diag = {
            "frequency_hz": frequency_hz,
            "elevation_deg": self.elevation_deg,
            "n_elements": n_elem,
            "n_snapshots": n_snap,
            "covariance_rank": rank_before,
            "rank_before_loading": rank_before,
            "covariance_condition_number": cond_before,
            "condition_before_loading": cond_before,
            "condition_after_loading": cond_after,
            "loading_rel": float(self.loading_rel),
            "loading_absolute": float(loading_abs),
            "loading_scale": float(scale),
            "peak_index": peak_index,
            "raw_spectrum": np.asarray(raw, dtype=float),
            "capabilities": dict(geometry.capabilities),
        }
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(grid[peak_index]),
            elevation_deg=self.elevation_deg,
            spectrum=shown,
            scan_grid=np.array(grid, dtype=float, copy=True),
            diagnostics=diag,
        )


