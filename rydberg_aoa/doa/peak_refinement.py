"""Sub-grid refinement of a scan-spectrum peak.

Every grid estimator in this package optimizes a Hermitian quadratic form
of the array manifold,

    f(theta) = m(theta)^H A m(theta),      m(theta) = exp(+j phi(theta)),

    phase beamforming / Bartlett : A = x x^H or Rxx      (maximize)
    MUSIC                        : A = En En^H            (minimize)
    MVDR / Capon                 : A = R_loaded^-1        (minimize)

The scan grid only locates the peak to the grid spacing (one degree for the
default grid), which quantizes the reported angle: an arrival at -30.4 deg
was reported as -30.0 deg.  This module finds the stationary point of the
SAME quadratic form between the two grid neighbours of the grid peak, using
the analytic derivative

    f'(theta) = 2 Re( m'(theta)^H A m(theta) ),   m' = j phi'(theta) * m,

and a bracketed root solve.  No interpolation model, no new steering
convention and no randomness is introduced; phi(theta) is
``steering.arrival_phases``.  If the neighbours do not bracket a stationary
point of the expected kind, or the refined point is not at least as good as
the grid peak, the grid angle is returned unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .steering import _coerce_positions, wave_number

__all__ = ["refine_quadratic_form_peak"]

# Angular distance [deg] below which a refined root is the grid angle itself.
_GRID_COINCIDENCE_DEG = 1e-9


def _manifold_and_derivative(positions, k, azimuth_deg, elevation_deg):
    """m(theta) = exp(+j phi) and dm/dtheta (per radian of azimuth)."""
    az = np.deg2rad(azimuth_deg)
    cos_el = np.cos(np.deg2rad(elevation_deg))
    sin_el = np.sin(np.deg2rad(elevation_deg))
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    phase = k * (cos_el * (x * np.sin(az) + y * np.cos(az)) + sin_el * z)
    dphase = k * cos_el * (x * np.cos(az) - y * np.sin(az))
    m = np.exp(1j * phase)
    return m, 1j * dphase * m


def refine_quadratic_form_peak(positions_m, frequency_hz, elevation_deg,
                               matrix, grid_deg, peak_index, maximize):
    """Refine ``grid_deg[peak_index]`` to the stationary point of m^H A m.

    Returns ``(azimuth_deg, refined)`` where ``refined`` is False when the
    grid angle was kept (edge of the grid without a bracket, flat or
    non-finite derivative, or a refined point that is not an improvement).
    """
    grid = np.asarray(grid_deg, dtype=float)
    index = int(peak_index)
    grid_angle = float(grid[index])
    if grid.shape[0] < 2:
        return grid_angle, False
    lo = float(grid[index - 1]) if index > 0 else grid_angle
    hi = float(grid[index + 1]) if index + 1 < grid.shape[0] else grid_angle
    if not lo < hi:
        return grid_angle, False

    positions = _coerce_positions(positions_m)
    k = wave_number(frequency_hz)
    a = np.asarray(matrix, dtype=complex)
    sign = 1.0 if maximize else -1.0

    def value(theta):
        m, _ = _manifold_and_derivative(positions, k, theta, elevation_deg)
        return sign * float(np.real(np.vdot(m, a @ m)))

    def slope(theta):
        m, dm = _manifold_and_derivative(positions, k, theta, elevation_deg)
        return sign * 2.0 * float(np.real(np.vdot(dm, a @ m)))

    try:
        with np.errstate(over="raise", invalid="raise"):
            slope_lo, slope_hi = slope(lo), slope(hi)
            if not (np.isfinite(slope_lo) and np.isfinite(slope_hi)):
                return grid_angle, False
            # A maximum of sign*f between lo and hi: rising at lo, falling at hi.
            if not (slope_lo > 0.0 and slope_hi < 0.0):
                return grid_angle, False
            root = float(brentq(slope, lo, hi, xtol=1e-13, rtol=8.9e-16, maxiter=200))
            if not np.isfinite(root) or not lo <= root <= hi:
                return grid_angle, False
            if value(root) < value(grid_angle):
                return grid_angle, False
    except (FloatingPointError, ValueError, RuntimeError):
        return grid_angle, False
    # A stationary point that coincides with the grid angle to within the
    # root solver's own resolution IS the grid angle (ideal on-grid data).
    if abs(root - grid_angle) <= _GRID_COINCIDENCE_DEG:
        return grid_angle, True
    return root, True
