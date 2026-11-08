"""Search-free high-resolution DOA estimators for structured arrays (DOA-ALG-002).

Deterministic NumPy-only ESPRIT and Root-MUSIC on the validated
ArrayGeometry / steering / IFSnapshotData X(M, K) stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .covariance_estimators import (
    SnapshotDOAEstimator,
    _covariance_diagnostics,
    _resolve_frequency_hz,
)
from .estimators import DOAError, DOAResult, IncompatibleGeometryError
from .geometry import ArrayGeometry
from .snapshots import sample_covariance
from .steering import direction_unit_vector, wave_number

__all__ = [
    "ESPRITEstimator",
    "RootMUSICEstimator",
    "phase_to_azimuth_fixed_elevation",
    "root_music_polynomial",
]

_ESPRIT_PINV_RCOND = 1.0e-12
_AMBIG_TOL_DEG = 1.0e-6


def _validate_n_sources(n_sources, *, label: str) -> int:
    try:
        as_float = float(n_sources)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} n_sources must be an integer with 1 <= n_sources; "
            f"got {n_sources!r}"
        ) from exc
    n = int(as_float)
    if not np.isfinite(as_float) or float(n) != as_float:
        raise ValueError(
            f"{label} n_sources must be an integer with 1 <= n_sources; "
            f"got {n_sources!r}"
        )
    if n < 1:
        raise ValueError(
            f"{label} n_sources must satisfy 1 <= n_sources; got {n_sources!r}"
        )
    return n


def _require_pair(geometry: ArrayGeometry, algorithm: str):
    pair = geometry.shift_invariant_pair
    if pair is None:
        raise IncompatibleGeometryError(
            algorithm, {"shift_invariant"}, {"shift_invariant"}
        )
    return pair


def _require_ula(geometry: ArrayGeometry, algorithm: str):
    missing = []
    if not geometry.capabilities.get("ula", False):
        missing.append("ula")
    pair = geometry.shift_invariant_pair
    if pair is None and "shift_invariant" not in missing:
        missing.append("shift_invariant")
    if missing:
        raise IncompatibleGeometryError(
            algorithm, {"ula", "shift_invariant"}, missing
        )
    params = geometry.ula_parameters
    if params is None:
        raise IncompatibleGeometryError(
            algorithm, {"ula", "shift_invariant"}, ["ula_parameters"]
        )
    return pair, params


def _check_elevation(elevation_deg) -> float:
    elev = float(elevation_deg)
    if not np.isfinite(elev):
        raise ValueError(f"elevation_deg must be finite; got {elevation_deg!r}")
    return elev


def phase_to_azimuth_fixed_elevation(
    spatial_phase_rad: float,
    displacement_m,
    frequency_hz: float,
    elevation_deg: float = 0.0,
):
    """Invert rotation phase psi to azimuth at fixed elevation.

    Project steering: phi_m = k_vec.r_m, s_m = exp(-j phi_m).  Between two
    subarrays separated by displacement d the rotation is exp(+j k u.d),
    so psi = k*(u(az,el).d) + 2*pi*n.  Solved for az in [-90,+90] deg with
    explicit integer-wrap enumeration; ambiguous mappings fail loudly.
    """
    psi = float(spatial_phase_rad)
    if not np.isfinite(psi):
        raise DOAError("spatial phase must be finite; failing loudly.")
    disp = np.asarray(displacement_m, dtype=float).reshape(3,)
    if disp.shape != (3,) or not np.all(np.isfinite(disp)):
        raise DOAError("displacement_m must be a finite 3-vector.")
    norm_d = float(np.linalg.norm(disp))
    if not np.isfinite(norm_d) or norm_d <= 0.0:
        raise DOAError("displacement must have positive finite length.")
    freq = float(frequency_hz)
    if not np.isfinite(freq) or freq <= 0.0:
        raise DOAError("frequency_hz must be positive finite.")
    elev = _check_elevation(elevation_deg)
    k = float(wave_number(freq))
    kd = k * norm_d
    wrapped = float((psi + np.pi) % (2.0 * np.pi) - np.pi)
    el = np.deg2rad(elev)
    cos_el = float(np.cos(el))
    ax = cos_el * float(disp[0])
    ay = cos_el * float(disp[1])
    az_const = float(np.sin(el)) * float(disp[2])
    amp = float(np.hypot(ax, ay))
    info: Dict[str, Any] = {
        "wrapped_phase_rad": wrapped,
        "k_times_displacement": kd,
        "subarray_displacement_m": disp.copy(),
        "aliased": False,
        "ambiguity": None,
    }
    if amp <= 0.0:
        raise DOAError(
            "displacement orthogonal to azimuth search direction; "
            "azimuth unobservable. Failing loudly."
        )
    lo = int(np.floor((k * (az_const - amp) - psi) / (2.0 * np.pi))) - 1
    hi = int(np.ceil((k * (az_const + amp) - psi) / (2.0 * np.pi))) + 1
    candidates: List[float] = []
    for n in range(lo, hi + 1):
        target = (psi + 2.0 * np.pi * n) / k - az_const
        if abs(target) > amp * (1.0 + 1.0e-12):
            continue
        ratio = float(np.clip(target / amp, -1.0, 1.0))
        base = np.arctan2(ax, ay)
        gamma = float(np.arccos(ratio))
        for az_rad in (base + gamma, base - gamma):
            az_deg = float(np.rad2deg(az_rad))
            az_fold = float((az_deg + 180.0) % 360.0 - 180.0)
            if az_fold < -90.0 - 1.0e-9 or az_fold > 90.0 + 1.0e-9:
                continue
            az_fold = float(np.clip(az_fold, -90.0, 90.0))
            u = direction_unit_vector(az_fold, elev)
            pred = float(k * np.dot(u, disp))
            resid = abs(float((pred - psi + np.pi) % (2.0 * np.pi) - np.pi))
            if resid > 1.0e-6:
                continue
            candidates.append(az_fold)
    unique: List[float] = []
    for cand in candidates:
        if not any(abs(cand - keep) <= _AMBIG_TOL_DEG for keep in unique):
            unique.append(cand)
    unique.sort()
    info["n_integer_wraps_considered"] = [lo, hi]
    info["n_candidate_azimuths"] = len(unique)
    if len(unique) != 1:
        info["aliased"] = True
        info["ambiguity"] = (
            f"non-unique azimuth inversion: {len(unique)} candidates "
            f"{unique} in [-90,+90] deg for phase {wrapped:.6f} rad "
            f"(k*|d|={kd:.6f}). Failing loudly."
        )
        raise DOAError(info["ambiguity"])
    if kd > np.pi + 1.0e-12:
        info["ambiguity"] = (
            f"k*|d|={kd:.6f} > pi; inversion unique only inside the "
            "fixed-elevation visible domain."
        )
    return float(unique[0]), info



def root_music_polynomial(noise_projector: np.ndarray) -> np.ndarray:
    """Build Root-MUSIC polynomial from G = En @ En^H (descending order)."""
    g = np.asarray(noise_projector, dtype=complex)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] < 2:
        raise DOAError(
            "noise projector must be square (M,M) with M>=2; "
            f"got shape {g.shape}"
        )
    if not np.all(np.isfinite(g)):
        raise DOAError("noise projector non-finite; refusing to root it.")
    m = int(g.shape[0])
    # Manifold on the covariance side is m(az) = conj(s(az)) = exp(+j phi_m),
    # so m(az) = v(z) with v(z)_i = z^i, z = exp(+j*psi), psi = k*u(az,el).d.
    # D(z) = v(z)^H G v(z) = sum_{i,j} G_ij z^{j-i}; the coefficient of z^l
    # is the plain diagonal sum trace(G, offset=l).  No conjugation: the
    # Hermitian pair g_{-l} = conj(g_l) is already carried by the polynomial.
    diag_sum = np.array(
        [np.trace(g, offset=int(l)) for l in range(-(m - 1), m)],
        dtype=complex,
    )
    # diag_sum[i] == g_{i-(m-1)}; np.roots wants descending powers, so the
    # coefficient of z^k (k = 0..2m-2 ascending) is diag_sum[k].
    ascending = diag_sum.copy()
    if not np.all(np.isfinite(ascending)):
        raise DOAError("Root-MUSIC coefficients non-finite; failing loudly.")
    scale = float(np.max(np.abs(ascending)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise DOAError("Root-MUSIC polynomial identically zero.")
    ascending = ascending / scale
    top = len(ascending) - 1
    while top > 0 and abs(ascending[top]) <= 1.0e-12:
        top -= 1
    ascending = ascending[: top + 1]
    if ascending.shape[0] < 3:
        raise DOAError(
            "Root-MUSIC polynomial degree-deficient "
            f"(length {ascending.shape[0]})."
        )
    return ascending[::-1].copy()


def _select_root_music_roots(roots: np.ndarray, n_sources: int):
    flat = np.asarray(roots, dtype=complex).reshape(-1)
    if flat.shape[0] < n_sources:
        raise DOAError(
            f"Root-MUSIC found {flat.shape[0]} roots, needs {n_sources}."
        )
    if not np.all(np.isfinite(flat)):
        raise DOAError("Root-MUSIC roots non-finite; failing loudly.")
    order = sorted(
        range(flat.shape[0]),
        key=lambda i: (
            abs(abs(flat[i]) - 1.0),
            float(np.angle(flat[i])),
            float(flat[i].real),
            float(flat[i].imag),
        ),
    )
    selected: List[int] = []
    for idx in order:
        if len(selected) >= n_sources:
            break
        z = complex(flat[idx])
        if abs(z) <= 1.0e-12:
            continue
        skip = False
        for prev in selected:
            w = complex(flat[prev])
            if abs(w) <= 1.0e-12:
                continue
            if abs(z - 1.0 / np.conj(w)) <= 1.0e-6 * max(1.0, abs(z)):
                skip = True
                break
        if skip:
            continue
        selected.append(idx)
    if len(selected) < n_sources:
        raise DOAError(
            "pathological Root-MUSIC root set after reciprocal-pair "
            f"filtering (need {n_sources})."
        )
    chosen = np.asarray([flat[i] for i in selected], dtype=complex)
    phase_order = np.argsort(
        [float(np.angle(z)) for z in chosen], kind="stable"
    )
    chosen = chosen[phase_order]
    info = {
        "selected_indices": [int(selected[i]) for i in phase_order],
        "root_magnitudes": [float(abs(z)) for z in chosen],
        "root_phases": [float(np.angle(z)) for z in chosen],
    }
    return chosen, info


class _StructuredSnapshotEstimator(SnapshotDOAEstimator):
    """Snapshot contract shared by search-free structured estimators."""

    def __init__(self, elevation_deg: float = 0.0, n_sources: int = 1):
        self.elevation_deg = float(_check_elevation(elevation_deg))
        self.n_sources = _validate_n_sources(n_sources, label=self.algorithm)

    @property
    def scan_azimuth_deg(self):
        return None

    @scan_azimuth_deg.setter
    def scan_azimuth_deg(self, value) -> None:
        if value is not None:
            raise ValueError("search-free estimator accepts no scan grid.")

    def _evd_covariance(self, mat):
        rxx = sample_covariance(mat)
        vals, vecs = np.linalg.eigh(rxx)
        order = np.argsort(vals)[::-1]
        out_v = np.asarray(vecs[:, order])
        out_w = np.asarray(vals[order], dtype=float)
        return rxx, out_w, out_v




class ESPRITEstimator(_StructuredSnapshotEstimator):
    """Deterministic least-squares ESPRIT (shift-invariant geometry)."""

    algorithm = "esprit"
    required_capabilities = frozenset({"shift_invariant"})

    def _estimate_impl(self, data, geometry, context=None):
        freq = _resolve_frequency_hz(context)
        mat = np.asarray(data, dtype=complex)
        n_elem = int(mat.shape[0])
        n_snap = int(mat.shape[1])
        pair = _require_pair(geometry, self.algorithm)
        if n_elem < 3:
            raise IncompatibleGeometryError(
                self.algorithm, {"shift_invariant", "M>=3"}, {"M>=3"}
            )
        sub = len(pair.indices_1)
        n_src = int(self.n_sources)
        if not 1 <= n_src < sub:
            raise ValueError(
                "esprit needs 1 <= n_sources < subarray size "
                f"(size={sub}); got {n_src}"
            )
        rxx, eigvals, eigvecs = self._evd_covariance(mat)
        sig = eigvecs[:, :n_src]
        i1 = np.asarray(pair.indices_1, dtype=int)
        i2 = np.asarray(pair.indices_2, dtype=int)
        psi = np.linalg.pinv(sig[i1, :], rcond=_ESPRIT_PINV_RCOND) @ sig[i2, :]
        lam = np.asarray(np.linalg.eigvals(psi), dtype=complex)
        if not np.all(np.isfinite(lam)):
            raise DOAError("ESPRIT Psi eigenvalues non-finite.")
        ph = np.array([float(np.angle(v)) for v in lam], dtype=float)
        azs: List[float] = []
        conv: List[Dict[str, Any]] = []
        for p in ph:
            a, info = phase_to_azimuth_fixed_elevation(
                p, pair.displacement_m, freq, self.elevation_deg
            )
            azs.append(float(a))
            conv.append(info)
        order = np.argsort(np.asarray(azs), kind="stable")
        srt = [float(azs[i]) for i in order]
        diag = _covariance_diagnostics(rxx)
        diag.update({
            "frequency_hz": freq,
            "elevation_deg": self.elevation_deg,
            "n_elements": n_elem,
            "n_snapshots": n_snap,
            "n_sources": n_src,
            "eigenvalues_Rxx": np.asarray(eigvals, dtype=float),
            "signal_subspace_rank": int(n_src),
            "psi_eigenvalues": np.asarray(lam, dtype=complex),
            "spatial_phase_shifts": np.asarray(ph, dtype=float),
            "subarray_displacement_m": np.asarray(
                pair.displacement_m, dtype=float).copy(),
            "subarray_indices_1": tuple(pair.indices_1),
            "subarray_indices_2": tuple(pair.indices_2),
            "all_azimuths_deg": np.asarray(srt, dtype=float),
            "phase_conversions": conv,
            "capabilities": dict(geometry.capabilities),
        })
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(srt[0]),
            elevation_deg=self.elevation_deg,
            spectrum=None,
            scan_grid=None,
            diagnostics=diag,
        )


class RootMUSICEstimator(_StructuredSnapshotEstimator):
    """Deterministic Root-MUSIC polynomial estimator (ULA ONLY)."""

    algorithm = "root_music"
    required_capabilities = frozenset({"ula", "shift_invariant"})

    def _estimate_impl(self, data, geometry, context=None):
        freq = _resolve_frequency_hz(context)
        mat = np.asarray(data, dtype=complex)
        n_elem = int(mat.shape[0])
        n_snap = int(mat.shape[1])
        _, ula = _require_ula(geometry, self.algorithm)
        if n_elem < 3:
            raise IncompatibleGeometryError(
                self.algorithm, {"ula", "shift_invariant", "M>=3"}, {"M>=3"}
            )
        n_src = int(self.n_sources)
        if not 1 <= n_src < n_elem:
            raise ValueError(
                f"root_music needs 1 <= n_sources < M (M={n_elem}); "
                f"got {n_src}"
            )
        rxx, eigvals, eigvecs = self._evd_covariance(mat)
        proj = eigvecs[:, n_src:] @ eigvecs[:, n_src:].conj().T
        coeffs = root_music_polynomial(proj)
        roots = np.asarray(np.roots(coeffs), dtype=complex)
        if roots.size == 0 or not np.all(np.isfinite(roots)):
            raise DOAError("Root-MUSIC roots empty/non-finite.")
        pick, sel = _select_root_music_roots(roots, n_src)
        disp = np.asarray(ula["displacement_m"], dtype=float)
        azs: List[float] = []
        conv: List[Dict[str, Any]] = []
        for z in pick:
            a, info = phase_to_azimuth_fixed_elevation(
                float(np.angle(z)), disp, freq, self.elevation_deg
            )
            azs.append(float(a))
            conv.append(info)
        order = np.argsort(np.asarray(azs), kind="stable")
        srt = [float(azs[i]) for i in order]
        diag = _covariance_diagnostics(rxx)
        diag.update({
            "frequency_hz": freq,
            "elevation_deg": self.elevation_deg,
            "n_elements": n_elem,
            "n_snapshots": n_snap,
            "n_sources": n_src,
            "eigenvalues_Rxx": np.asarray(eigvals, dtype=float),
            "polynomial_coefficients": np.asarray(coeffs, dtype=complex),
            "all_roots": np.asarray(roots, dtype=complex),
            "selected_roots": np.asarray(pick, dtype=complex),
            "root_magnitudes": list(sel["root_magnitudes"]),
            "root_phases": list(sel["root_phases"]),
            "ula_spacing_m": float(ula["spacing_m"]),
            "ula_axis": np.asarray(ula["axis"], dtype=float).copy(),
            "subarray_displacement_m": disp.copy(),
            "all_azimuths_deg": np.asarray(srt, dtype=float),
            "phase_conversions": conv,
            "capabilities": dict(geometry.capabilities),
        })
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(srt[0]),
            elevation_deg=self.elevation_deg,
            spectrum=None,
            scan_grid=None,
            diagnostics=diag,
        )

