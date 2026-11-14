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
import sympy as sp

from rydberg_aoa.analytical.symbolic import (
    calculate_rho21_exact,
    get_symbolic_rho21_solver,
)
from rydberg_aoa.physics.hamiltonian.create_sensor_model import create_sensor_model
from rydberg_aoa.utils.validation import validate_steady_state

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


def _numerical_coherences(delta_p, omega_p=OMEGA_P):
    """Solve the Lindblad steady state numerically and return (rho_12, rho_21).

    Extraction conventions (matching production code):
        rho_12 = Tr[|2><1| . rho_ss]   (what steady_state.py extracts today)
        rho_21 = Tr[|1><2| . rho_ss]   (Bussey-style convention)
    """
    H, c_ops, obs_s21 = create_sensor_model(
        omega_p, OMEGA_C, OMEGA_RF,
        delta_p, DELTA_C, DELTA_RF,
        GAMMAS, DEPHASING,
    )
    s12 = obs_s21.dag()  # |1><2|
    rho_ss = qt.steadystate(H, c_ops)
    validate_steady_state(rho_ss, qt.liouvillian(H, c_ops), context='symbolic parity',
                          backend='qutip')
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


def test_symbolic_projection_matches_full_first_order_lindblad():
    """Solve all six complex coherence equations in their real representation."""
    from rydberg_aoa.analytical.symbolic import _build_lindblad

    generator, _, symbols, coherences, populations = _build_lindblad()
    values = (3., -2., 5., OMEGA_P, OMEGA_C, OMEGA_RF, *GAMMAS, *DEPHASING)
    epsilon = sp.Symbol('epsilon', real=True)
    unknowns = [value for pair in coherences.values() for value in pair]
    substitutions = dict(zip(symbols.values(), values))
    substitutions[symbols['Op']] = epsilon * OMEGA_P
    substitutions.update(zip(populations, (1, 0, 0, 0)))
    substitutions.update((value, epsilon * value) for value in unknowns)
    equations = []
    for i, j in coherences:
        linear = generator[i - 1, j - 1].xreplace(substitutions).diff(epsilon).subs(epsilon, 0)
        equations.extend(linear.as_real_imag())
    matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
    x, y, *_ = np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(rhs, dtype=float)[:, 0])
    actual, _ = get_symbolic_rho21_solver()(*values)
    assert actual == pytest.approx(complex(x, y), rel=1e-12)


@pytest.fixture(scope="module")
def numerical_scan():
    return np.array([_numerical_coherences(dp)[0] for dp in DELTA_P_SCAN])


def test_golden_parity_vs_numerical(symbolic_scan, numerical_scan):
    """Compare the linear term at the frozen probe scale using both probe strengths."""
    rho12_sym, _ = symbolic_scan
    half_probe = np.array([_numerical_coherences(dp, OMEGA_P / 2)[0] for dp in DELTA_P_SCAN])
    full_error = np.max(np.abs(rho12_sym - numerical_scan))
    half_error = np.max(np.abs(rho12_sym / 2 - half_probe))
    assert half_error < full_error / 4
    # rho(Omega) = a*Omega + b*Omega**3 + O(Omega**5). Cancel the cubic term.
    linear_at_reference = (8 * half_probe - numerical_scan) / 3
    linear_error = np.max(np.abs(rho12_sym - linear_at_reference))
    print(f'Full probe error: {full_error:.3e}; half probe: {half_error:.3e}; linear term: {linear_error:.3e}')
    assert linear_error < 1e-5


def test_absorption_sign_at_resonance(symbolic_scan, numerical_scan):
    rho12, _ = symbolic_scan
    mid = len(DELTA_P_SCAN) // 2
    assert rho12[mid].imag < 0 and numerical_scan[mid].imag < 0


def test_linewidth_parity(symbolic_scan, numerical_scan):
    """Compare the positive absorption peak on the same detuning grid."""
    rho12_sym, _ = symbolic_scan

    def fwhm(scan, profile):
        peak = int(np.argmax(profile))
        half = profile[peak] / 2
        assert half > 0
        left = np.flatnonzero(profile[:peak] <= half)
        right = np.flatnonzero(profile[peak + 1:] <= half)
        assert left.size and right.size, 'scan does not bracket both half maxima'
        i, j = left[-1], peak + 1 + right[0]
        low = np.interp(half, profile[i:i + 2], scan[i:i + 2])
        high = np.interp(half, profile[j - 1:j + 1][::-1], scan[j - 1:j + 1][::-1])
        return high - low

    w_sym = fwhm(DELTA_P_SCAN, -rho12_sym.imag)
    w_num = fwhm(DELTA_P_SCAN, -numerical_scan.imag)
    assert w_sym > 0 and w_num > 0
    print(f'FWHM: symbolic={w_sym:.3f}, numerical={w_num:.3f} rad/us')
    assert abs(w_sym - w_num) / w_num < 0.05
