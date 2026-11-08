"""
Doppler detuning shifts for moving atoms.

Physics
-------
In the rest frame of an atom moving with velocity component v along a beam's
wavevector k, the laser frequency is shifted to ω' = ω - k·v, so the detuning
becomes

    Δ_i(v) = Δ_i - k_i · s_i · v ,   k_i = 2π / λ_i,

where s_i = +1 for a field propagating along +ẑ (probe) and s_i = -1 for a
counter-propagating field (coupling), following the `factor_doppler_shift`
convention of MaxwellBloch (Ogden 2020).

For the counter-propagating pump-probe geometry the two-photon (Raman)
detuning shifts only by the *differential* amount

    δ_two-photon(v) = (-k_p + k_c)·v = -(2π/λ_p)(1 - λ_p/λ_c)·v,

which is much smaller than each single-photon shift — this is why EIT
survives Doppler broadening in counter-propagating cells. The RF transition
(λ ~ cm) is Doppler-free at the kHz level and is therefore not shifted.

Units: input velocity in m/s; returned shifts in rad/μs (project convention,
rad/μs = 2π × MHz).
"""

import numpy as np  # np.pi for wavenumber construction

# Seconds → microseconds divisor: angular shift [rad/s] → [rad/μs]
PER_S_TO_PER_US = 1e-6


def get_doppler_shifts(
    velocity_ms: float,
    lambda_probe_m: float,
    lambda_coupling_m: float,
    counter_propagating: bool = True,
) -> tuple[float, float]:
    """
    Per-velocity-class detuning shifts for the probe and coupling lasers.

    Δ_p(v) = -k_p·v            (probe taken as forward-propagating, +ẑ)
    Δ_c(v) = -k_c·s_c·v        (s_c = -1 if counter-propagating else +1)

    Parameters
    ----------
    velocity_ms : float
        Atom velocity component v along the probe propagation axis [m/s].
    lambda_probe_m : float
        Probe vacuum wavelength λ_p [m] (e.g. 780.24 nm for Rb D2).
    lambda_coupling_m : float
        Coupling vacuum wavelength λ_c [m] (e.g. 483.4 nm for Rydberg coupling).
    counter_propagating : bool
        True for the standard counter-propagating pump-probe vapour-cell
        geometry (coupling wavevector antiparallel to probe).

    Returns
    -------
    tuple[float, float]
        (delta_p_shift, delta_c_shift) in rad/μs, to be ADDED to the
        zero-velocity detunings Δ_p and Δ_c.
    """
    # Wavenumbers k = 2π/λ [rad/m]
    k_probe = 2.0 * np.pi / lambda_probe_m
    k_coupling = 2.0 * np.pi / lambda_coupling_m

    # Propagation sign factors: probe +ẑ, coupling −ẑ when counter-propagating
    s_coupling = -1.0 if counter_propagating else 1.0

    # Rest-frame detuning shifts Δ' = Δ − k·v; k already carries 2π [rad/s],
    # so only the s→μs divisor converts to project units [rad/μs]
    delta_p_shift = -k_probe * velocity_ms * PER_S_TO_PER_US
    delta_c_shift = -k_coupling * s_coupling * velocity_ms * PER_S_TO_PER_US

    return float(delta_p_shift), float(delta_c_shift)
