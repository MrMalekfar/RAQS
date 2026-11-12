"""
Absorption constant for the susceptibility-based transmission model.

Computes the scalar prefactors converting the optical coherence ρ₂₁ into cell
transmission T = exp(-2 · od_factor · Im{K_chi · ρ₂₁}) — Beer-Lambert law applied
to the imaginary part of the susceptibility χ = K_chi · ρ₂₁.
"""

import numpy as np

# Import our modules
from ...config import parameters as params, atomic_config as acfg

# Single SI_Constants instance (audit Issue #23): class attributes — including LAMBDA_P,
# which atomic_config overwrites with the ARC-derived probe wavelength — resolve
# dynamically through the instance, so the singleton always reflects current values.
_CONST_SI = params.SI_Constants()


def calculate_absorption_constant(
    mu_21_c_m=None,
    omega_p_rad_per_s=None,
    n0_m3=None,
    cell_length_m=None,
    lambda_p_m=None,
):
    """Compute the optical-depth factor and susceptibility prefactor.

    K_chi = -2 N₀ μ₂₁² / (ε₀ ħ ω_p) — linear susceptibility prefactor [s⁻¹]
        N₀: atomic number density [m⁻³] (ARC vapor-pressure model)
        μ₂₁: probe-transition dipole moment [C·m] (ARC, cached per λ_p/λ_c/f_RF)
        ω_p: probe angular frequency [rad/s]
    od_factor = π L / λ_p — resonant optical-depth scale [dimensionless]
        L: vapor-cell length [m]; λ_p: probe wavelength [m]

    RF-002 injection contract: each physics input may be supplied explicitly;
    omitted arguments fall back lazily to the identical configuration/cache
    source, keeping the default call bit-identical to pre-RF-002 behavior.
    An explicitly supplied μ₂₁ bypasses the dipole-cache lookup entirely
    (dependency isolation: injected μ drives only the quadratic K_chi term).

    Args:
        mu_21_c_m: probe dipole moment [C·m] (default: cached dipole matrix 'probe' entry)
        omega_p_rad_per_s: probe angular frequency [rad/s] (default: acfg.OMEGA_P × US_TO_S_FACTOR)
        n0_m3: number density [m⁻³] (default: _CONST_SI.N0)
        cell_length_m: vapor-cell length [m] (default: params.CELL_LENGTH)
        lambda_p_m: probe wavelength [m] (default: _CONST_SI.LAMBDA_P)

    Returns:
        tuple: (od_factor, K_chi)

    Raises:
        RuntimeError: If no dipole matrix is available for the current parameters.
    """
    if mu_21_c_m is None:
        dipole_matrix = acfg.get_cached_dipole_matrix()
        if not dipole_matrix or 'probe' not in dipole_matrix:
            # Explicit guard (Issue #4): fail loudly instead of TypeError on None subscript
            raise RuntimeError(
                "Dipole matrix unavailable for the current laser/RF parameters — "
                "atomic-level optimization did not populate the dipole cache. "
                "Check LASER_PARAMS/RF_PARAMS or re-run level optimization."
            )
        # μ₂₁ = ⟨2|d|1⟩ — probe transition dipole moment in SI units [C·m]
        mu_21_c_m = dipole_matrix['probe']['dipole_si']

    # Lazy fallbacks — identical legacy sources keep the default path bit-identical
    if omega_p_rad_per_s is None:
        # ω_p [rad/s] = OMEGA_P [rad/μs] × 10⁶ (internal unit: rad/μs, SI required here)
        omega_p_rad_per_s = acfg.OMEGA_P * params.US_TO_S_FACTOR
    if n0_m3 is None:
        n0_m3 = _CONST_SI.N0
        if n0_m3 is None:
            from ...exceptions import PhysicsConfigurationError
            raise PhysicsConfigurationError('ATOM-001: configure the atom before reading the default number density.')
    if cell_length_m is None:
        cell_length_m = params.CELL_LENGTH
    if lambda_p_m is None:
        lambda_p_m = _CONST_SI.LAMBDA_P

    # K_chi = -2 N₀ μ₂₁² / (ε₀ ħ ω_p); negative sign ⇒ Im{χ} > 0 gives absorption
    K_chi = (-2.0 * n0_m3 * mu_21_c_m**2) / (_CONST_SI.EPSILON_0 * _CONST_SI.HBAR * omega_p_rad_per_s)

    # od_factor = π L / λ_p — single-pass phase accumulation over the cell length
    od_factor = (np.pi * cell_length_m) / lambda_p_m

    return od_factor, K_chi