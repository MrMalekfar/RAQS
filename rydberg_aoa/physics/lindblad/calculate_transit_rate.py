"""
Effective Markovian transit rate for the four-level vapor-cell model.

Physical picture
----------------
Atoms crossing the finite probe beam spend a finite interaction time in the
illuminated region and are then replaced by fresh atoms from the reservoir.
This is a LOSS/REPOPULATION process, not phase diffusion, and is represented
in the master equation by reset collapse operators (see
``create_transit_reset_collapse_ops``), not by pure-dephasing projectors.

Effective model (DPH-002B, documented approximation)
----------------------------------------------------
Interaction region : circular hard boundary with radius R equal to the probe
                     1/e² intensity radius w0.
Velocity           : 2-D thermal mean transverse speed
                     v_perp_mean = sqrt(pi·k_B·T / (2·m)).
Mean chord         : d_bar = pi·R/2 (mean chord of a circular region).

    Gamma_t = v_perp_mean / d_bar = (2/pi)·v_perp_mean / w0
            = sqrt(2/pi)·sqrt(k_B·T/m) / w0            [1/μs]

This is an EFFECTIVE MARKOVIAN TRANSIT RATE — it is NOT the exact
finite-Gaussian-beam transit solution (the true residence-time distribution
is non-exponential and the exact response is non-Lorentzian).
"""

import numpy as np

from ...config import constants as const
from ...config import parameters as params
from ...config import atomic_config as acfg
from ...config.laser_rf_config import LASER_PARAMS


def calculate_transit_rate_per_us(
    temperature_k=None,
    beam_waist_mm=None,
    atom_mass_kg=None,
):
    """
    Effective Markovian transit (loss/repopulation) rate [1/μs].

    Gamma_t = sqrt(2/pi) · sqrt(k_B·T/m) / w0

    with the interaction boundary R = probe 1/e² intensity radius w0 and the
    2-D mean transverse speed v_perp_mean = sqrt(pi·k_B·T/(2·m)) over the
    mean chord d_bar = pi·R/2.

    RF-002-style injection contract: every implicit config input can be
    supplied explicitly; omitted arguments fall back lazily to the identical
    configuration source.

    Args:
        temperature_k: vapour-cell temperature [K] (default params.VAPOR_CELL_TEMPERATURE_K —
                       the production owner, shared with the vapor density and the
                       Doppler velocity distribution; NOT the duplicated
                       constants.VAPOR_CELL_TEMPERATURE_K)
        beam_waist_mm: probe 1/e² beam radius [mm] (default LASER_PARAMS['probe']['beam_waist_mm'])
        atom_mass_kg: atom mass [kg] (default acfg.ATOM.mass)

    Returns:
        float: effective transit rate Gamma_t [1/μs]
    """
    if temperature_k is None:
        # DPH-002C: resolve the production owner (parameters), the same source
        # used by the vapor density and the Doppler velocity classes, so a
        # future temperature sweep moves all three mechanisms together.
        temperature_k = params.VAPOR_CELL_TEMPERATURE_K
    if beam_waist_mm is None:
        beam_waist_mm = LASER_PARAMS['probe']['beam_waist_mm']
    if atom_mass_kg is None:
        atom_mass_kg = acfg.ATOM.mass

    w0 = beam_waist_mm * 1e-3                       # 1/e² radius [m]
    v_perp_mean = np.sqrt(np.pi * const.k_B * temperature_k / (2.0 * atom_mass_kg))
    d_bar = np.pi * w0 / 2.0                        # mean chord through the region [m]
    # [m/s] / [m] = [1/s] -> [1/μs] (single inverse-time conversion, no 2π)
    return v_perp_mean / d_bar / const.US_TO_S_FACTOR


def resolve_effective_transit_rate(transit_rate_per_us=None, transit_enabled=None):
    """
    PHY-SW-001: resolve the transit rate that actually enters the model.

    Gate semantics (switch wins over any explicit rate):
      * ``transit_enabled=False`` → 0.0 (no reset collapse operators may be
        built downstream; Gamma_t itself remains computable via
        `calculate_transit_rate_per_us()` and no transit contribution may
        reappear as pure dephasing — DPH-002B already guarantees that).
      * ``transit_enabled=True`` and ``transit_rate_per_us=None`` → the
        configuration-derived effective Markovian rate (legacy behavior).
      * ``transit_enabled=True`` and an explicit rate → that rate unchanged.
      * ``transit_enabled=None`` → resolves from ``params.TRANSIT_ENABLED``
        (True in production → default-path invariance).

    Args:
        transit_rate_per_us: explicit effective transit rate [1/μs] or None
        transit_enabled: PHY-SW-001 gate or None (→ params.TRANSIT_ENABLED)

    Returns:
        float: effective transit rate entering create_sensor_model [1/μs]
    """
    if transit_enabled is None:
        transit_enabled = params.TRANSIT_ENABLED
    if not transit_enabled:
        return 0.0
    if transit_rate_per_us is None:
        return calculate_transit_rate_per_us()
    return transit_rate_per_us