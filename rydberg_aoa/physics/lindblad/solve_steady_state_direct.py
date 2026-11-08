"""
Exact direct steady-state solver for the Lindblad master equation (DOP-003S).

Replaces ``qutip.steadystate`` in the adaptive Doppler hot path ONLY.

DOP-003P established that the dominant per-solve cost in the adaptive
Maxwell velocity integration is QuTiP's per-call framework overhead
(Options/dtype machinery), not the numerical linear algebra itself.  For
the four-level sensor model the Liouvillian is 16x16 and an exact dense
trace-constrained solve is both cheaper (~36-40x) and mathematically
identical: this module solves EXACTLY the same master equation

    d rho / dt = L(rho) = -i[H, rho] + sum_k D[L_k] rho,
    L(rho) = 0,  Tr(rho) = 1,

constructed from the SAME Hamiltonian and collapse operators the caller
already holds.  No weak-probe approximation, no state elimination, no
analytical shortcut.

Vectorization convention (HARD GATE, pinned by test):
    QuTiP uses COLUMN stacking (Fortran order):

        vec(rho) = rho.flatten(order="F")
        rho      = vec.reshape(N, N, order="F")

    and the Liouvillian matrix acts as ``L @ vec(rho)`` in that basis
    (verified elementwise against the explicit Lindblad generator and
    against ``vector_to_operator(L * operator_to_vector(rho))``).

Trace constraint (DOP-003P lesson): the chosen row of the dense
Liouvillian matrix is FIRST zeroed completely and only then loaded with
the trace functional coefficients.  Overwriting just the diagonal of an
unmodified Liouvillian row mixes genuine population-rate equations into
the constraint and corrupts the solution.

Fail-loud contract: a singular (or otherwise unsolvable) constrained
system raises ``np.linalg.LinAlgError`` with diagnostics -- this helper
NEVER silently returns zeros or falls back to different physics.
"""

from __future__ import annotations

import numpy as np
import qutip

from ...utils.validation import validate_steady_state

__all__ = ["solve_steady_state_direct"]

# Documented trace-row weight conventions.
_TRACE_WEIGHT_SCALE = "scale"   # mean |L| of the dense Liouvillian (default)
_TRACE_WEIGHT_UNIT = "unit"     # plain unit weight


def _vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorization matching qutip.operator_to_vector."""
    return rho.flatten(order="F")


def _unvec(vec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_vec` (column stacking)."""
    return vec.reshape((n, n), order="F")


def _trace_functional(n: int) -> np.ndarray:
    """Row vector ``e`` with ``e @ vec(rho) == Tr(rho)`` in column-stacking.

    In column-stacking order the diagonal entries of ``rho`` live at flat
    indices ``i * n + i``; ``vec(identity)`` places exactly these entries,
    and since they are real units no conjugation is required.
    """
    e = np.zeros(n * n, dtype=complex)
    e[np.arange(n) * (n + 1)] = 1.0
    return e


def solve_steady_state_direct(H, c_ops, trace_weight: str = "scale",
                              return_diagnostics: bool = False):
    """Exact dense trace-constrained steady state of ``d rho/dt = L(rho)``.

    Solves the SAME Lindblad master equation ``qutip.steadystate(H, c_ops)``
    solves, by direct linear algebra on the dense Liouvillian.

    Parameters
    ----------
    H : qutip.Qobj
        Hermitian Hamiltonian (must satisfy ``H.isherm``; the Liouvillian
        column-stacking action is validated for Hermitian H).
    c_ops : list of qutip.Qobj
        Collapse operators, exactly as passed to ``qutip.steadystate``.
    trace_weight : str
        ``"scale"`` (default): scale the trace row by ``mean(|L|)`` of the
        dense Liouvillian -- a Liouvillian-scale weight comparable to the
        matrix rows, following the spirit of QuTiP's scale-aware
        formulation.  ``"unit"``: plain unit weight.  Both are
        machine-equivalent (validated); the choice must not be tuned per
        spectrum point.
    return_diagnostics : bool
        If True, additionally return a diagnostics dict
        ``(rho_qobj, diagnostics)``.  The dict is a validation/research
        aid and is NOT computed on the production hot path by default; the
        inexpensive NUM-SS-001 certification gate itself IS always executed
        (see below).

    Returns
    -------
    qutip.Qobj
        The steady-state density matrix (dense operator Qobj), directly
        usable in ``(obs * rho_ss).tr()`` coherence extraction.

    Raises
    ------
    ValueError
        If ``H`` is not Hermitian or ``trace_weight`` is unknown.
    np.linalg.LinAlgError
        If the trace-constrained system is singular or the solve fails.
        The exception carries diagnostics.  Never swallowed.
    rydberg_aoa.exceptions.PhysicsCalculationError
        NUM-SS-001: if the returned state is structurally invalid (shape,
        NaN/Inf), not Hermitian, not trace one, not positive, or not
        stationary for the ORIGINAL Liouvillian.  Never swallowed, never
        repaired; this is the same contract the ``"qutip"`` backend passes
        through in ``simulation.steady_state._steady_state_solve``.
    """
    if trace_weight not in ("scale", "unit"):
        raise ValueError(
            "trace_weight must be 'scale' or 'unit', got %r" % (trace_weight,)
        )
    if not H.isherm:
        raise ValueError(
            "solve_steady_state_direct requires a Hermitian Hamiltonian; "
            "the column-stacking Liouvillian action is validated for H = H+."
        )

    # The Liouvillian of the SAME (H, c_ops) pair -- no physics change.
    liou = qutip.liouvillian(H, list(c_ops))
    ldense = np.asarray(liou.full(), dtype=complex)
    n = H.shape[0]
    dim = n * n

    if ldense.shape != (dim, dim):
        raise np.linalg.LinAlgError(
            "Liouvillian shape %r does not match N^2 = %d for N=%d"
            % (ldense.shape, dim, n)
        )

    # Trace-constrained augmented system:  A @ vec(rho) = b, Tr(rho) = 1.
    # The LAST row is REPLACED: zero it completely first (DOP-003P lesson:
    # overwriting only the diagonal of an unmodified Liouvillian row mixes
    # population-rate equations into the constraint), then load the trace
    # functional scaled by the chosen weight.
    a = ldense.copy()
    b = np.zeros(dim, dtype=complex)
    if trace_weight == "scale":
        weight = float(np.mean(np.abs(ldense)))
    else:
        weight = 1.0
    trace_row = _trace_functional(n)
    a[-1, :] = 0.0
    a[-1, :] = weight * trace_row
    b[-1] = weight

    vec_rho = np.linalg.solve(a, b)
    rho = _unvec(vec_rho, n)

    # NUM-SS-001 fail-loud gate: certify the state BEFORE any consumer can
    # reach it.  ``ldense`` is the ORIGINAL Liouvillian captured before the
    # last row was replaced by the trace constraint, i.e. exactly the
    # generator whose stationary state was solved for.  A rejected state is
    # raised, never repaired, renormalized or replaced (F05).
    validate_steady_state(
        rho, ldense,
        context="solve_steady_state_direct", backend="direct",
    )

    if not return_diagnostics:
        return qutip.Qobj(rho)

    # ---- validation-only diagnostics (optional, off the hot path) ----
    # NUM-SS-001: the trace error is the COMPLEX modulus |Tr(rho) - 1|.
    # The old expression took np.real() first and therefore discarded any
    # imaginary part of the trace; report the full distance from 1.
    trace_err = float(abs(trace_row @ vec_rho - 1.0))
    herm_err = float(np.max(np.abs(rho - rho.conj().T)))
    residual = float(np.linalg.norm(ldense @ vec_rho))
    try:
        cond = float(np.linalg.cond(a))
    except np.linalg.LinAlgError:
        cond = float("inf")
    min_eval = float(np.min(np.abs(np.linalg.eigvals(ldense))))
    herm_part = 0.5 * (rho + rho.conj().T)
    min_eig_rho = float(np.min(np.linalg.eigvalsh(herm_part)))
    diagnostics = {
        "trace_error": trace_err,
        "hermiticity_error": herm_err,
        "liouvillian_residual": residual,
        "condition_number": cond,
        "min_abs_liouvillian_eigenvalue": min_eval,
        "min_rho_eigenvalue": min_eig_rho,
        "trace_weight": weight,
    }
    return qutip.Qobj(rho), diagnostics
