"""
Parameter-derived Doppler resonance HINTS for the adaptive Maxwell integrator.

DOP-001B established that plain adaptive quadrature can miss a globally
narrow EIT/AT resonance if no initial sample lands near it.  These helpers
compute physics-informed breakpoint hints from the CURRENT parameters.
They are SAMPLING AIDS only: `quad_vec` remains responsible for the
integral, and the hints never truncate the integration domain.

For the ladder scheme the cumulative two-photon detuning is

    D(v) = Delta_p(v) + Delta_c(v),

with

    Delta_p(v) = delta_p_center + scan_offset + delta_p_shift(v),
    Delta_c(v) = delta_c_center            + delta_c_shift(v).

Doppler shifts (counter-propagating geometry):

    delta_p_shift(v) = -k_p v
    delta_c_shift(v) = +k_c v   (s_c = -1)

so

    D(v) = D0 + k_eff * v  (rad/us),
    k_eff = (k_c - k_p) * PER_S_TO_PER_US.

Resonance conditions
--------------------
RF OFF: the EIT dark state requires D(v) = 0.

RF ON : the upper {|3>,|4>} dressed block

        [ -D                  -Omega_RF/2 ]
        [ -Omega_RF/2   -(D + Delta_RF)   ]

has a dressed eigenvalue crossing zero when

        D (D + Delta_RF) - Omega_RF^2 / 4 = 0

with roots

        D_plus/minus = (-Delta_RF +/- sqrt(Delta_RF^2 + Omega_RF^2)) / 2.

Each root translates to a velocity hint v = (D_target - D0) / k_eff.
Around each center a stencil of breakpoints is added with a velocity scale

        dv_hint ~ gamma_scale / |k_eff|,

where gamma_scale is a homogeneous coherence scale (transition width +
dephasing + effective transit rate) expressed in rad/us.
"""

from __future__ import annotations

import math


def two_photon_slope_rad_per_us_per_ms(
    lambda_probe_m: float,
    lambda_coupling_m: float,
    counter_propagating: bool,
    per_s_to_per_us: float = 1e-6,
) -> float:
    """dD/dv [rad/us per m/s] for the ladder two-photon detuning."""
    k_p = 2.0 * math.pi / lambda_probe_m
    k_c = 2.0 * math.pi / lambda_coupling_m
    s_c = -1.0 if counter_propagating else 1.0
    # delta_p_shift = -k_p v; delta_c_shift = -k_c s_c v; D = shift_p + shift_c
    return (-k_p + -k_c * s_c) * per_s_to_per_us


def _v_from_D(D_target_rad, D0_rad, k_eff):
    if abs(k_eff) < 1e-30:
        return None
    return (D_target_rad - D0_rad) / k_eff


def resonance_velocity_hints(
    delta_p_center_rad,
    delta_c_center_rad,
    scan_offset_rad,
    delta_rf_rad,
    omega_rf_rad,
    lambda_probe_m,
    lambda_coupling_m,
    counter_propagating: bool = True,
    per_s_to_per_us: float = 1e-6,
):
    """Compute velocity breakpoint hints for one scan point.

    Returns list[float] of velocity hints (m/s) sorted/deduplicated.
    """
    k_eff = two_photon_slope_rad_per_us_per_ms(
        lambda_probe_m, lambda_coupling_m, counter_propagating,
        per_s_to_per_us)
    D0 = delta_p_center_rad + scan_offset_rad + delta_c_center_rad

    targets = [0.0]  # RF-off dark state
    if omega_rf_rad is not None and abs(omega_rf_rad) > 0.0:
        disc = math.sqrt(delta_rf_rad ** 2 + omega_rf_rad ** 2)
        targets.append((-delta_rf_rad + disc) / 2.0)
        targets.append((-delta_rf_rad - disc) / 2.0)

    hints = []
    for t in targets:
        v = _v_from_D(t, D0, k_eff)
        if v is not None:
            hints.append(v)

    # Deduplicate/nearby merge and sort.
    hints = sorted(set(round(h, 6) for h in hints))
    merged = []
    for h in hints:
        if not merged or abs(h - merged[-1]) > 1e-3:
            merged.append(h)
    return merged


def velocity_scale_hint(
    gamma_scale_rad_per_us,
    lambda_probe_m,
    lambda_coupling_m,
    counter_propagating: bool = True,
    per_s_to_per_us: float = 1e-6,
) -> float:
    """dv ~ gamma_scale / |dD/dv| — velocity width of the narrow feature."""
    k_eff = two_photon_slope_rad_per_us_per_ms(
        lambda_probe_m, lambda_coupling_m, counter_propagating,
        per_s_to_per_us)
    return gamma_scale_rad_per_us / abs(k_eff)


def breakpoint_stencil(
    v_center,
    dv_scale,
    factors=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
):
    """Around a resonance center build a breakpoint stencil.

    The stencil uses an O(1)-O(10) spread of dv_scale to improve initial
    sampling.  Breakpoints are only initial-grid points; they do not bound
    the integration.
    """
    return [v_center + f * dv_scale for f in factors]