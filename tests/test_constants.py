#!/usr/bin/env python3
"""Test the SI constants to see if they are reasonable."""

import pytest

# Audit B.3: stale script-style test — SI_Constants.MU_12 was removed; μ₂₁ is now
# sourced from the ARC dipole cache (atomic_config.get_cached_dipole_matrix()).
pytest.skip(
    "Stale test: SI_Constants.MU_12 removed; dipole now comes from the ARC cache (audit B.3)",
    allow_module_level=True,
)

import numpy as np

# Manually import what we need
from rydberg_aoa.config import parameters as params

# SI Constants
const = params.SI_Constants()

print("=== SI Constants ===")
print(f"N0 (atomic density): {const.N0:.2e} m^-3")
print(f"MU_12 (dipole): {const.MU_12:.2e} C*m")
print(f"LAMBDA_P (wavelength): {const.LAMBDA_P:.2e} m")
print(f"EPSILON_0: {const.EPSILON_0:.2e}")
print(f"HBAR: {const.HBAR:.2e}")

print("\n=== Transmission Scaling Factors ===")
omega_p_si = 0.1 * params.US_TO_S_FACTOR  # Example: 0.1 rad/us = 1e7 rad/s
absorption_factor_const = (np.pi * params.CELL_LENGTH) / const.LAMBDA_P
K_chi = (-2.0 * const.N0 * const.MU_12**2) / (const.EPSILON_0 * const.HBAR * omega_p_si)

print(f"omega_p_si example: {omega_p_si:.2e} rad/s")
print(f"absorption_factor_const (π*L/λ_p): {absorption_factor_const:.2e}")
print(f"K_chi: {K_chi:.2e}")

print("\n=== Typical Susceptibility Calculation ===")
rho21 = 0.01  # Typical value
rho_21_normalized = rho21 / 0.1  # Example Omega_P = 0.1 rad/us
chi = K_chi * rho_21_normalized
absorption = 2.0 * absorption_factor_const * abs(np.imag(chi))
print(f"rho21: {rho21}")
print(f"rho_21_normalized: {rho_21_normalized}")
print(f"chi: {chi:.2e}")
print(f"absorption factor: {absorption:.2e}")
print(f"transmission: {np.exp(-absorption):.6f}")
