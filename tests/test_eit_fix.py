#!/usr/bin/env python3
"""Test script to verify the EIT transmission fix."""

import pytest

# Audit B.3 / Part B: rydberg_aoa.tools.simulation_core was removed from the project;
# this script cannot be imported without it. Skip cleanly instead of a collection error.
pytest.skip(
    "Stale test: rydberg_aoa.tools.simulation_core module removed (audit Part B)",
    allow_module_level=True,
)

import numpy as np
from rydberg_aoa.config import atomic_config as acfg, parameters as params
from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
from rydberg_aoa.tools.simulation_core import simulate_spectrum

print("=== Testing EIT Transmission Fix ===")

# Test transmission calculation directly with test values
print("\n--- Direct Transmission Test ---")
test_rho_values = [
    0.01 + 0.01j,   # Positive imaginary part (absorption)
    0.01 - 0.01j,   # Negative imaginary part (amplification)
    0.00 + 0.00j    # Zero (no effect)
]

for rho in test_rho_values:
    T = calculate_transmission_from_rho21(rho)
    print(f"rho_21 = {rho}, T = {T:.4f}")

# Test full spectrum simulation
print("\n--- Full Spectrum Simulation ---")
try:
    print("Testing spectrum simulation (no RF)...")
    dp_scan, trans_no_rf, trans_analytic = simulate_spectrum(0.0)

    print(f"Detuning range: {dp_scan[0]:.1f} to {dp_scan[-1]:.1f} MHz")
    print(f"Transmission range: {np.min(trans_no_rf):.4f} to {np.max(trans_no_rf):.4f}")
    print(f"EIT window depth change: {np.max(trans_no_rf) - np.min(trans_no_rf):.4f}")

    # Check for variation (no longer flat)
    if np.max(trans_no_rf) - np.min(trans_no_rf) > 0.001:
        print("✓ SUCCESS: Transmission varies with probe detuning!")
    else:
        print("✗ FAILURE: Transmission still flat")

    print("\nTesting spectrum simulation (with RF)...")
    dp_scan_rf, trans_with_rf, trans_with_rf_analytic = simulate_spectrum(acfg.OMEGA_RF_BASE)

    print(f"Transmission range (with RF): {np.min(trans_with_rf):.4f} to {np.max(trans_with_rf):.4f}")
    print(f"Autler-Townes splitting: {np.max(trans_with_rf) - np.min(trans_with_rf):.4f}")

    if np.max(trans_with_rf) - np.min(trans_with_rf) > 0.001:
        print("✓ SUCCESS: Autler-Townes splitting observed!")
    else:
        print("✗ FAILURE: No Autler-Townes splitting")

except Exception as e:
    print(f"ERROR in spectrum test: {e}")

print("\nEIT fix test completed.")
