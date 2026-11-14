"""
Symbolic (SymPy) derivation of the exact steady-state probe coherence.

This module derives the weak-probe steady-state coherence of the 4-level
cascade Optical Bloch Equations (OBEs) directly from the SAME Hamiltonian and
collapse-operator conventions used by the numerical solver
(`physics.hamiltonian.create_sensor_model`). Because the derivation starts
from the same Lindblad structure, the result is the linear term in probe
Rabi frequency of the numerical steady-state coherence.

Physics model (rotating frame, hbar = 1, frequencies in rad/us, rates in 1/us):
    H = Delta_p|2><2| + (Delta_p+Delta_c)|3><3| + (Delta_p+Delta_c+Delta_rf)|4><4|
        - Omega_p/2 (|1><2| + |2><1|)
        - Omega_c/2 (|2><3| + |3><2|)
        - Omega_rf/2 (|3><4| + |4><3|)
    Collapse operators:
        sqrt(gamma_21)|1><2|, sqrt(gamma_32)|2><3|, sqrt(gamma_43)|3><4|,
        sqrt(deph_2)|2><2|, sqrt(deph_3)|3><3|, sqrt(deph_4)|4><4|

Steady state: the master equation d(rho)/dt = L(rho) = 0 is LINEAR in rho,
so the closed-form solution is rho_ss = M^{-1} b. We build M and b
symbolically, solve once, and lambdify the result into a fast NumPy function.

Reference: derived from the same conventions as Bussey et al. (2024),
"Numerical model of N-level cascade systems"; weak-probe reduction follows
the standard treatment of Mohapatra et al., Phys. Rev. A 75, 030901(R) (2007).
"""

import logging
import time

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: the symbolic derivation is expensive (seconds) and its
# structure never changes, so it is performed exactly once per process.
# ---------------------------------------------------------------------------
_SOLVER_CACHE = {}


def _build_lindblad():
    """Construct the symbolic Lindblad generator L(rho) for the 4-level cascade.

    Returns:
        tuple: (L, rho, symbols_dict, coherence_pairs, populations). L is a 4x4 SymPy Matrix whose
            (i, j) entry is d(rho_ij)/dt, rho is the symbolic density matrix,
            and symbols_dict maps parameter names to SymPy symbols.
    """
    # --- Frequencies [rad/us] and nonnegative event rates [1/us] ---
    # Delta_p = omega_probe - omega_21   — probe detuning
    # Delta_c = omega_coupling - omega_32 — coupling detuning
    # Delta_rf = omega_rf - omega_43     — RF detuning
    # Omega_p/c/rf                        — Rabi frequencies (Omega = mu*E/hbar)
    # gamma_21/32/43                      — Einstein-A spontaneous decay rates
    # deph_2/3/4                          — pure dephasing rates on levels |2>,|3>,|4|
    dp, dc, drf = sp.symbols('Delta_p Delta_c Delta_rf', real=True)
    Op, Oc, Orf = sp.symbols('Omega_p Omega_c Omega_rf', real=True)
    g21, g32, g43 = sp.symbols('gamma_21 gamma_32 gamma_43', nonnegative=True)
    d2, d3, d4 = sp.symbols('deph_2 deph_3 deph_4', nonnegative=True)
    sym = dict(dp=dp, dc=dc, drf=drf, Op=Op, Oc=Oc, Orf=Orf,
               g21=g21, g32=g32, g43=g43, d2=d2, d3=d3, d4=d4)

    I = sp.I

    # --- Hamiltonian (identical convention to create_sensor_model) ---
    # H_det = diag(0, Delta_p, Delta_p+Delta_c, Delta_p+Delta_c+Delta_rf)
    # H_int off-diagonals are -Omega/2 for each RWA-coupled pair
    H = sp.Matrix([
        [0,          -Op/2,                 0,                       0],
        [-Op/2,      dp,                    -Oc/2,                   0],
        [0,          -Oc/2,                 dp + dc,                 -Orf/2],
        [0,          0,                     -Orf/2,                  dp + dc + drf],
    ])

    # --- Density matrix with EXPLICIT real unknowns ---
    # Hermiticity: rho_ji = conj(rho_ij). Writing Re/Im parts explicitly keeps
    # the entire system REAL, avoiding SymPy conjugate-handling pitfalls.
    # Unknown populations: p1..p4 ; unknown coherences: (x_ij, y_ij) for i<j
    # with rho_ij = x_ij + I*y_ij.
    p1, p2, p3, p4 = sp.symbols('p1 p2 p3 p4', real=True)
    coh = {}
    for (i, j) in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        x, y = sp.symbols(f'x{i}{j} y{i}{j}', real=True)
        coh[(i, j)] = (x, y)

    def rho_entry(i, j):
        # rho_ij for i<j is x + I*y; for i>j the conjugate; diagonal is real
        if i == j:
            return {1: p1, 2: p2, 3: p3, 4: p4}[i]
        if i < j:
            x, y = coh[(i, j)]
            return x + I * y
        x, y = coh[(j, i)]
        return x - I * y

    rho = sp.Matrix(4, 4, lambda i, j: rho_entry(i + 1, j + 1))

    # --- Collapse operators C_k = sqrt(rate) * |a><b| ---
    # Spontaneous emission (Einstein A) and pure dephasing (population jumps)
    c_ops = [
        sp.sqrt(g21) * sp.Matrix([[0, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]),  # |1><2| : |2> -> |1>
        sp.sqrt(g32) * sp.Matrix([[0, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]),  # |2><3| : |3> -> |2>
        sp.sqrt(g43) * sp.Matrix([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 0]]),  # |3><4| : |4> -> |3>
        sp.sqrt(d2) * sp.Matrix([[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]),  # |2><2| dephasing
        sp.sqrt(d3) * sp.Matrix([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]]),  # |3><3| dephasing
        sp.sqrt(d4) * sp.Matrix([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 1]]),  # |4><4| dephasing
    ]

    # --- Lindblad generator: L = -i[H, rho] + sum_k D[C_k] rho ---
    # D[C]rho = C rho C^dag - 1/2 {C^dag C, rho}
    commutator = H * rho - rho * H
    dissipator = sp.zeros(4, 4)
    for C in c_ops:
        Cd = C.H
        dissipator += C * rho * Cd - sp.Rational(1, 2) * (Cd * C * rho + rho * Cd * C)
    L = -I * commutator + dissipator
    return L, rho, sym, coh, (p1, p2, p3, p4)


def _derive_weak_probe():
    """Derive rho_12 and rho_21 to first order in Omega_p (weak-probe limit).

    Weak-probe reduction (Mohapatra et al. 2007):
      - Populations pinned to their zeroth-order values: rho_11 = 1,
        rho_22 = rho_33 = rho_44 = 0 (probe too weak to deplete |1>).
      - All six coherences are O(Omega_p); any term containing BOTH Omega_p
        and a coherence is O(Omega_p^2) and is discarded.

    Returns:
        tuple: (rho12_expr, rho21_expr, arg_symbols)
    """
    t0 = time.perf_counter()
    L, _, sym, coh, pops = _build_lindblad()
    dp, dc, drf = sym['dp'], sym['dc'], sym['drf']
    Op, Oc, Orf = sym['Op'], sym['Oc'], sym['Orf']

    # At first order only ground-to-excited coherences are driven. Their
    # three rows couple to each other without conjugates or excited coherences.
    ground = dict(zip(pops, (1, 0, 0, 0)))
    no_coherence = dict.fromkeys((v for pair in coh.values() for v in pair), 0)
    z = sp.symbols('z12 z13 z14')
    projected = {**ground, **no_coherence, Op: 0}
    for j, coherence in enumerate(z, start=2):
        projected[coh[1, j][0]] = coherence
    equations = [L[0, j].xreplace(projected) + L[0, j].xreplace({**ground, **no_coherence})
                 for j in range(1, 4)]
    matrix, rhs = sp.linear_eq_to_matrix(equations, z)
    numerator = matrix.copy()
    numerator[:, 0] = rhs
    rho12 = numerator.det() / matrix.det()
    rho21 = sp.conjugate(rho12)

    arg_symbols = (dp, dc, drf, Op, Oc, Orf,
                   sym['g21'], sym['g32'], sym['g43'],
                   sym['d2'], sym['d3'], sym['d4'])
    logger.info("Symbolic weak-probe derivation completed in %.1f s",
                time.perf_counter() - t0)
    return rho12, rho21, arg_symbols


def get_symbolic_rho21_solver(mode='weak_probe'):
    """Return a fast NumPy-callable solver for the steady-state coherences.

    Args:
        mode: 'weak_probe' (first order in Omega_p, includes ALL decay and
              dephasing rates). The derivation runs once and is cached.

    Returns:
        callable: f(Delta_p, Delta_c, Delta_rf, Omega_p, Omega_c, Omega_rf,
                    gamma_21, gamma_32, gamma_43,
                    deph_2, deph_3, deph_4) -> (rho_12, rho_21)
            Accepts scalars or NumPy arrays (broadcasting supported).
            rho_12 = <1|rho|2> and rho_21 = <2|rho|1> = conj(rho_12).
    """
    if mode in _SOLVER_CACHE:
        return _SOLVER_CACHE[mode]

    if mode != 'weak_probe':
        raise ValueError(f"Unsupported symbolic mode: {mode!r}")

    rho12, rho21, args = _derive_weak_probe()
    solver = sp.lambdify(args, (rho12, rho21), modules='numpy')
    _SOLVER_CACHE[mode] = solver
    return solver


def _effective_delta_rf(delta_rf, rf_orientation=None):
    """RF-DIR-001C: the Delta_rf value fed to the cached legacy derivation.

    The derivation builds H44 = Delta_p + Delta_c + Delta_rf and the
    Delta_rf symbol appears nowhere else in the generator (populations and
    collapse structure are free of it - asserted at matrix level in
    tests/test_rfdir001c_hamiltonian_direction.py).  Therefore evaluating
    the same solver at -Delta_rf is EXACTLY the derivation for a downward
    RF edge, whose rotating frame carries the laboratory
    Delta_rf = w_rf - |nu| with the opposite sign (RF-DIR-001C).
    None/upward - and therefore every historical caller - pass through
    unchanged, bit-identically.
    """
    if rf_orientation is not None and rf_orientation.is_downward:
        return -delta_rf
    return delta_rf


def calculate_rho21_exact(delta_p, delta_c, delta_rf,
                          omega_p, omega_c, omega_rf,
                          gammas, dephasings, mode='weak_probe',
                          rf_orientation=None):
    """Evaluate the symbolic steady-state probe coherence.

    Args:
        delta_p, delta_c, delta_rf: Detunings [rad/us] (scalar or array).
        omega_p, omega_c, omega_rf: Rabi frequencies [rad/us].
        gammas: [gamma_21, gamma_32, gamma_43] nonnegative decay rates [1/us].
        dephasings: [deph_2, deph_3, deph_4] nonnegative dephasing rates [1/us].
        mode: Solver mode passed to get_symbolic_rho21_solver.
        rf_orientation: RF-DIR-001C - RFTransitionOrientation | None.  For a
            downward RF edge the cached +Delta_rf derivation is evaluated at
            -Delta_rf (exact; see _effective_delta_rf).  None and upward
            edges keep the historical evaluation bit-identically.

    Returns:
        tuple: (rho_12, rho_21) complex scalars or arrays. Both conventions
            are provided so callers can match whichever observable their
            downstream transmission formula expects.
    """
    solver = get_symbolic_rho21_solver(mode)
    drf_eval = _effective_delta_rf(delta_rf, rf_orientation)
    g21, g32, g43 = gammas
    d2, d3, d4 = dephasings
    rho12, rho21 = solver(delta_p, delta_c, drf_eval,
                          omega_p, omega_c, omega_rf,
                          g21, g32, g43, d2, d3, d4)
    return rho12 + 0j, rho21 + 0j
