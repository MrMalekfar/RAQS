"""Project-owned plane-wave steering functions (DOA-ARCH-001).

Convention preserved from the legacy implementation
(rydberg_aoa.signal_processing.array_geometry.get_array_phases):

    arrival phase of element i      phi_i = k_vec . r_i
    with                           k_vec = k * u(azimuth, elevation)
    and                            k     = 2 * pi * f_RF / c

where u(azimuth, 0) == (sin az, cos az, 0): azimuth is measured from the
+y (broadside) axis, positive toward +x, and elevation tilts the arrival
direction toward +z.

The steering reference is the CONJUGATE phase ramp

    s(az) = exp(-j phi(az))

so the beam power P(az) = |s(az)^H x|^2 peaks at the arrival azimuth.  For
the frozen six-path ULA this reproduces the production result exactly
(estimated AoA == true AoA, phase progression -22.5 deg per path).

IMPORTANT — the spatial steering frequency is the RF SIGNAL carrier
(RF_SIGNAL_FREQUENCY), NOT the heterodyne intermediate frequency (IF_FREQ);
the IF beat note carries no resolvable differential phase across the array
and must never be used for spatial steering.
"""

from __future__ import annotations

import numpy as np

from .geometry import ArrayGeometry

SPEED_OF_LIGHT_M_S = 299792458.0


def _coerce_positions(positions_m):
    """Accept ArrayGeometry or a raw ndarray; normalize to (M, 3) metres."""
    if isinstance(positions_m, ArrayGeometry):
        return positions_m.positions_m
    arr = np.asarray(positions_m, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            "positions must have shape (M, 2) or (M, 3) in metres; "
            f"got {arr.shape}"
        )
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(arr.shape[0], dtype=float)])
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions contain non-finite values")
    return arr


def direction_unit_vector(azimuth_deg, elevation_deg=0.0):
    """Arrival-direction unit vector u = (cos el sin az, cos el cos az, sin el)."""
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    return np.array([
        float(np.cos(el) * np.sin(az)),
        float(np.cos(el) * np.cos(az)),
        float(np.sin(el)),
    ])


def wave_number(frequency_hz, speed_of_light=SPEED_OF_LIGHT_M_S):
    """Spatial angular wave number k = 2 * pi * f / c [rad/m].

    The frequency argument is the RF signal carrier in Hz.  Passing the
    heterodyne IF frequency here is a physical error: the IF beat note
    carries no differential spatial phase across the array.
    """
    f = float(frequency_hz)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(
            f"frequency_hz must be a positive finite value; got {frequency_hz!r}")
    c = float(speed_of_light)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(
            f"speed_of_light must be a positive finite value; got {speed_of_light!r}")
    return 2.0 * np.pi * f / c


def arrival_phases(positions_m, frequency_hz, azimuth_deg, elevation_deg=0.0):
    """phi_i = k_vec . r_i, shape (M,).

    Bit-identical to the legacy get_array_phases for 2-D positions at
    elevation 0.
    """
    pos = _coerce_positions(positions_m)
    k = wave_number(frequency_hz)
    u = direction_unit_vector(azimuth_deg, elevation_deg)
    kx, ky, kz = k * u
    return kx * pos[:, 0] + ky * pos[:, 1] + kz * pos[:, 2]


def steering_vector(positions_m, frequency_hz, azimuth_deg, elevation_deg=0.0):
    """s(az) = exp(-j phi(az)), shape (M,)."""
    return np.exp(-1j * arrival_phases(
        positions_m, frequency_hz, azimuth_deg, elevation_deg))


def steering_matrix(positions_m, frequency_hz, azimuths_deg, elevation_deg=0.0):
    """Stack steering vectors for a scan grid; shape (G, M)."""
    pos = _coerce_positions(positions_m)
    az = np.asarray(azimuths_deg, dtype=float)
    if az.ndim != 1:
        raise ValueError(f"azimuths_deg must be a 1-D scan grid; got shape {az.shape}")
    rows = [steering_vector(pos, frequency_hz, float(a), elevation_deg) for a in az]
    return np.vstack(rows) if rows else np.empty((0, pos.shape[0]), dtype=complex)