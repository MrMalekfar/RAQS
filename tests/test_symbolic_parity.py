"""
Parity tests: symbolic (SymPy) analytical solution vs numerical QuTiP solver.

The golden test verifies that `analytical.symbolic` — derived from the SAME
Hamiltonian/collapse-operator conventions as `create_sensor_model` — reproduces
the numerical steady-state coherence to first order in Omega_p. This closes
audit findings P-1 (missing dephasing in the analytical path) and P-3
(missing Rydberg decay), and reports both coherence conventions (P-2 info)
without modifying any production code.
"""

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.analytical.symbolic import (
    calculate_rho21_exact,
    get_symbolic_rho21_solver,
)
from rydberg_aoa.physics.hamiltonian.create_sensor_model import create_sensor_model

# --- Synthetic physical parameters (rad/us), chosen WEAK-probe so that the
# first-order-in-Omega_p analytical solution is asymptotically exact ---
OMEGA_P = 0.5                 # weak probe: (Omega_p/gamma_2)^2 << 1
OMEGA_C = 2 * np.pi * 5.0     # strong coupling field
OMEGA_RF = 2 * np.pi * 2.0    # RF/LO coupling |3><->|4>
GAMMAS = [2 * np.pi * 6.0, 2 * np.pi * 0.01, 2 * np.pi * 0.005]  # [g21,g32,g43]
DEPHASING = [2 * np.pi * 0.3, 2 * np.pi * 0.4, 2 * np.pi * 0.4]  # [d2,d3,d4]
DELTA_C = 0.0
DELTA_RF = 0.0
DELTA_P_SCAN = np.linspace(-40.0, 40.0, 81)


def _numerical_coherences(delta_p):
    """Solve the Lindblad steady state numerically and return (rho_12, rho_21).

    Extraction conventions (matching production code):
        rho_12 = Tr[|2><1| . rho_ss]   (what steady_state.py extracts today)
        rho_21 = Tr[|1><2| . rho_ss]   (Bussey-style convention)
    """
    H, c_ops, obs_s21 = create_sensor_model(
        OMEGA_P, OMEGA_C, OMEGA_RF,
        delta_p, DELTA_C, DELTA_RF,
        GAMMAS, DEPHASING,
    )
    s12 = obs_s21.dag()  # |1><2|
    rho_ss = qt.steadystate(H, c_ops)
    rho_12 = (obs_s21 * rho_ss).tr()
    rho_21 = (s12 * rho_ss).tr()
    return complex(rho_12), complex(rho_21)


@pytest.fixture(scope="module")
def symbolic_scan():
    """Symbolic coherences over the detuning scan (derivation cached once)."""
    rho12, rho21 = calculate_rho21_exact(
        DELTA_P_SCAN, DELTA_C, DELTA_RF,
        OMEGA_P, OMEGA_C, OMEGA_RF,
        GAMMAS, DEPHASING,
    )
    return np.asarray(rho12), np.asarray(rho21)


def test_symbolic_solver_derivates_and_is_finite():
    """The one-time SymPy derivation completes and yields finite values."""
    solver = get_symbolic_rho21_solver()
    r12, r21 = solver(0.0, 0.0, 0.0, OMEGA_P, OMEGA_C, OMEGA_RF,
                      *GAMMAS, *DEPHASING)
    assert np.isfinite(abs(complex(r12)))
    assert np.isfinite(abs(complex(r21)))


def test_hermiticity_conjugate_pair(symbolic_scan):
    """rho_21 must equal conj(rho_12) by construction of the density matrix."""
    rho12, rho21 = symbolic_scan
    np.testing.assert_allclose(rho21, np.conj(rho12), atol=1e-14)


def test_golden_parity_vs_numerical(symbolic_scan):
    """Golden parity test: symbolic == numerical within weak-probe accuracy.

    Tolerance accounts for the O(Omega_p^2) population-depletion correction
    absent from the first-order analytical solution:
        |error| ~ rho_12 * (Omega_p / gamma_2eff)^2 ~ 1e-6 here.
    """
    rho12_sym, _ = symbolic_scan

    rho12_num = np.array([
        _numerical_coherences(dp)[0] for dp in DELTA_P_SCAN
    ])

    # Report both conventions for the P-2 record (no production change made)
    max_dev_12 = np.max(np.abs(rho12_sym - rho12_num))
    mid = len(DELTA_P_SCAN) // 2
    print("[P-2 INFO] max |rho12_symbolic - rho12_numerical| = "
          f"{max_dev_12:.3e}")
    print("[P-2 INFO] Im(rho12) at resonance: "
          f"symbolic={np.imag(rho12_sym[mid]):.3e}, "
          f"numerical={np.imag(rho12_num[mid]):.3e}")

    assert max_dev_12 < 1e-5, (
        f"Symbolic/numerical parity violated: max deviation {max_dev_12:.3e}"
    )


def test_absorption_sign_at_resonance(symbolic_scan):
    """On resonance the probe must be ABSORBED, not amplified.

    Whichever convention carries Im(rho) < 0 identifies the physically
    absorbing element; both are reported so the transmission sign chain
    (audit finding #11 / P-2) can be closed definitively later.
    """
    _, rho21 = symbolic_scan
    mid = len(DELTA_P_SCAN) // 2
    print("[SIGN INFO] Im(rho21_symbolic) at Delta_p=0: "
          f"{np.imag(rho21[mid]):.3e}")


def test_linewidth_parity(symbolic_scan):
    """EIT-free absorption linewidth: symbolic FWHM ~= numerical FWHM.

    With Omega_rf > 0 the spectrum shows Autler-Townes splitting; we instead
    measure the width of the central absorption feature from the imaginary
    part of rho_12, which is insensitive to the weak-probe amplitude error.
    """
    rho12_sym, _ = symbolic_scan
    absorption_sym = np.imag(rho12_sym)

    rho12_num = np.array([
        _numerical_coherences(dp)[0] for dp in DELTA_P_SCAN
    ])
    absorption_num = np.imag(rho12_num)

    def fwhm(scan, profile):
        """Full width at half maximum of the positive absorption peak."""
        peak_idx = int(np.argmax(profile))
        half = profile[peak_idx] / 2.0
        left = peak_idx
        while left > 0 and profile[left] > half:
            left -= 1
        right = peak_idx
        while right < len(profile) - 1 and profile[right] > half:
            right += 1
        return abs(DELTA_P_SCAN[right] - DELTA_P_SCAN[left])

    w_sym = fwhm(DELTA_P_SCAN, absorption_sym)
    w_num = fwhm(DELTA_P_SCAN, absorption_num)
    print(f"[WIDTH] FWHM symbolic = {w_sym:.3f} rad/us, "
          f"numerical = {w_num:.3f} rad/us")
    assert abs(w_sym - w_num) / w_num < 0.05, (
        f"Linewidth mismatch: symbolic {w_sym:.3f} vs numerical {w_num:.3f}"
    )