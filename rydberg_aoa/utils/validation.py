"""
Validation utilities for quantum simulations.
"""

import logging
import math

import numpy as np

from ..exceptions import PhysicsCalculationError

logger = logging.getLogger(__name__)


def check_conservation_laws(rho, tolerance=1e-10, label=""):
    """
    Check that the density matrix satisfies fundamental conservation laws.

    Args:
        rho: QuTiP density matrix (Qobj)
        tolerance: Numerical tolerance for checks
        label: Optional label for logging

    Returns:
        bool: True if conservation laws are satisfied

    Raises:
        AssertionError: If conservation laws are violated beyond tolerance
    """
    # Convert to numpy array for calculations
    rho_array = rho.full()

    # Check trace = 1
    trace = np.trace(rho_array)
    trace_error = abs(trace - 1.0)

    # Check sum of diagonal elements = 1
    populations = np.diag(rho_array)
    pop_sum = np.sum(populations)
    pop_error = abs(pop_sum - 1.0)

    # Log results
    if label:
        logger.debug(f"Conservation check for {label}:")
    logger.debug(f"  Trace = {trace:.12f} (error: {trace_error:.2e})")
    logger.debug(f"  Population sum = {pop_sum:.12f} (error: {pop_error:.2e})")

    # Check if within tolerance
    if trace_error > tolerance or pop_error > tolerance:
        error_msg = f"Conservation laws violated for {label}: Trace error: {trace_error:.2e}, Population sum error: {pop_error:.2e}"
        logger.error(error_msg)
        raise AssertionError(error_msg)

    return True


# ---------------------------------------------------------------------------
# NUM-SS-001 — fail-loud steady-state density-matrix contract
# ---------------------------------------------------------------------------
# ``qutip.steadystate`` and the dense trace-constrained solve implemented in
# ``rydberg_aoa/physics/lindblad/solve_steady_state_direct.py`` both *return
# a matrix*; a return value is not proof that the matrix is the stationary
# state of the Liouvillian the caller actually built.  A wrong trace-row
# construction (the DOP-003P lesson), a corrupted generator, a non-converged
# solve or a silent fallback can all yield a matrix that is Hermitian,
# trace-one and therefore perfectly plausible downstream, where the whole
# state finally collapses to a single complex number ``Tr[sigma21 . rho]``
# that nothing re-checks.  NUM-SS-001 inserts ONE shared, non-repairing gate
# at the solver seam, used by BOTH backends (``"direct"`` and ``"qutip"``).
#
# Checks, in this order
# ---------------------
# 1. structure     square 2-D numeric rho; L must be (N^2, N^2)
# 2. finiteness    no NaN/Inf in rho nor in L
# 3. Hermiticity   max|rho - rho^dagger|
# 4. trace         |Tr(rho) - 1| as a COMPLEX modulus (never its real part)
# 5. positivity    min eigenvalue of (rho + rho^dagger)/2
# 6. stationarity  ||L vec(rho)||_2, absolute AND relative to ||L||_F
#
# Norms and why BOTH stationarity forms are required
# --------------------------------------------------
# ``vec(rho) = rho.flatten(order="F")`` is QuTiP's column-stacking
# convention (pinned by test), so ``L @ vec(rho)`` vectorizes the
# Lindbladian image ``L(rho)`` and ``||L vec(rho)||_2`` equals its
# Frobenius norm; it therefore carries the units of the generator
# (rad/us, i.e. 1/us, for the sensor model).
#
# * the RELATIVE form ``||L vec(rho)||_F / ||L||_F`` is scale-free and
#   measures how well the linear system was solved however large the rates
#   are;
# * the ABSOLUTE form ``||L vec(rho)||_F`` is required because a large
#   ``||L||_F`` must not be allowed to loosen the physical guarantee
#   one-for-one — with only the relative test, growing rates hide a growing
#   absolute residual.
#
# Both must pass.
#
# Measured reference (NUM-SS-001 probe, four-level model built at
# production scale: Gamma_21 = 38.11 1/us, Omega_p = 15.76 rad/us,
# Omega_c = 15 rad/us, detunings of order 2*pi rad/us, giving
# ||L||_F = 86.0 and cond(A) = 155 for the trace-constrained matrix A):
#
#     valid, direct backend   |L vec(rho)|_F = 1.4e-15   rel = 1.6e-17
#     valid, qutip backend    |L vec(rho)|_F = 2.5e-15   rel = 2.9e-17
#     |Tr(rho) - 1|           <= 3.3e-16
#     Hermiticity error       <= 2.0e-16
#     absorbing projector     |L vec(rho)|_F = 0.0 exactly, min eig = 0
#     non-stationary rho      rel = 1.6e-01  (trace/Hermiticity already OK)
#     trace off by 1e-7       |Tr - 1| = 1.0e-07, |L vec(rho)|_F = 1.1e-06
#     non-Hermitian by 1e-6   error  = 1.0e-06, |L vec(rho)|_F = 2.3e-05
#
# The tolerances below therefore sit 5.6 ... 7.5 orders of magnitude above
# the honest round-off floor of the SYNTHETIC near-resonant probe model above
# while still rejecting corruptions at the 1e-6 ... 1e-7 level.  IMPORTANT
# (NUM-SS-001-R2): those numbers describe the SYNTHETIC model only
# (||L||_F = 86, near-resonant detunings).  The actual PRESCRIBED reference is
# strongly FAR-detuned (coupling detuning of order -1.3e6 rad/us) and carries a
# much larger ||L||_F, so its residual round-off floor ``eps*cond(A)*||L||_F``
# is correspondingly larger and the RELATIVE bound is the meaningful scale-free
# guard there.  We make NO claim that ||L||_F stays below any fixed value
# across operating points, and the ABSOLUTE 1e-8 bound is NOT asserted to hold
# universally — it is calibrated to the near-resonant probe model.
#
# No repair, no fallback (F05): a rejected state is never symmetrized,
# re-normalized, clipped, replaced or logged-and-continued.
# ---------------------------------------------------------------------------

STEADY_STATE_TRACE_TOLERANCE = 1e-9               # |Tr(rho) - 1|, dimensionless
STEADY_STATE_HERMITICITY_TOLERANCE = 1e-9         # max|rho - rho^dagger|
STEADY_STATE_POSITIVITY_TOLERANCE = 1e-9          # tolerated negative eigenvalue
STEADY_STATE_STATIONARITY_ABS_TOLERANCE = 1e-8    # 1/us for the sensor model
STEADY_STATE_STATIONARITY_REL_TOLERANCE = 1e-9    # ||L rho|| / ||L||

def validate_steady_state(rho, liouvillian, *, context, backend=None,
                          trace_tolerance=STEADY_STATE_TRACE_TOLERANCE,
                          hermiticity_tolerance=STEADY_STATE_HERMITICITY_TOLERANCE,
                          positivity_tolerance=STEADY_STATE_POSITIVITY_TOLERANCE,
                          stationarity_abs_tolerance=STEADY_STATE_STATIONARITY_ABS_TOLERANCE,
                          stationarity_rel_tolerance=STEADY_STATE_STATIONARITY_REL_TOLERANCE):
    """Certify that ``rho`` is a valid stationary density matrix of ``L``.

    This is the single NUM-SS-001 contract shared by the ``"direct"`` and
    ``"qutip"`` steady-state backends.  It only MEASURES and either
    certifies or rejects: it never repairs, symmetrizes, renormalizes,
    clips or replaces the value it was given (F05, no fallback).

    Parameters
    ----------
    rho : qutip.Qobj or array-like
        Candidate steady state, square ``(N, N)``.
    liouvillian : array-like or qutip.Qobj
        The ORIGINAL dense or SciPy sparse ``(N^2, N^2)`` Liouvillian built from the same
        ``H`` and ``c_ops`` the solver received, acting on the
        column-stacked ``vec(rho) = rho.flatten(order="F")``.  Callers must
        NOT hand in a trace-constrained, augmented or otherwise modified
        matrix: the stationarity metric is only meaningful for ``L`` itself.
    context : str
        Where the value came from; reproduced verbatim in failure messages.
    backend : str, optional
        Solver label (``"direct"`` / ``"qutip"``) for the diagnostics.

    Returns
    -------
    dict
        Measured metrics: ``trace_error``, ``hermiticity_error``,
        ``min_eigenvalue``, ``stationarity_residual`` (units of the
        generator, 1/us for the sensor), ``stationarity_residual_rel``,
        ``liouvillian_frobenius_norm``, ``dimension``, ``backend``.

    Raises
    ------
    rydberg_aoa.exceptions.PhysicsCalculationError
        On any structural or physical violation, or when the metrics
        themselves cannot be computed.  The message reports every metric,
        every tolerance and the violated checks, and chains the original
        exception when a metric computation failed (``raise ... from``).
    """
    # ---- validate the gate's OWN tolerance arguments BEFORE any check ----
    # (NUM-SS-001-R1).  A NaN/Inf/negative/non-real tolerance would silently
    # DISABLE its own comparison (``x > nan`` and ``x < -nan`` are both always
    # False), turning a mis-set tolerance into a free pass.  Zero is a
    # legitimate (strictest) tolerance; only negatives and non-finite/non-real
    # values are rejected.  F05: fail loudly, never coerce-and-continue, and
    # never accept conversion success as proof of a valid type.
    for _tol_name, _tol in (
        ("trace_tolerance", trace_tolerance),
        ("hermiticity_tolerance", hermiticity_tolerance),
        ("positivity_tolerance", positivity_tolerance),
        ("stationarity_abs_tolerance", stationarity_abs_tolerance),
        ("stationarity_rel_tolerance", stationarity_rel_tolerance),
    ):
        if isinstance(_tol, bool) or not isinstance(
                _tol, (int, float, np.integer, np.floating)):
            raise PhysicsCalculationError(
                f"{context}: NUM-SS-001 tolerance '{_tol_name}' must be a real "
                f"numeric scalar, got {_tol!r} "
                f"(type={type(_tol).__name__}); a non-numeric tolerance would "
                f"silently disable its check (F05).")
        try:
            _tol_f = float(_tol)
        except (OverflowError, ValueError, TypeError) as exc:
            raise PhysicsCalculationError(
                f"{context}: NUM-SS-001 tolerance '{_tol_name}' cannot be "
                f"converted to a real float, got {_tol!r} "
                f"(type={type(_tol).__name__}); {exc}. A tolerance that "
                f"overflows or cannot be converted would silently disable its "
                f"check (F05).") from exc
        if not math.isfinite(_tol_f):
            raise PhysicsCalculationError(
                f"{context}: NUM-SS-001 tolerance '{_tol_name}' must be finite, "
                f"got {_tol_f!r}; a NaN/Inf tolerance would silently disable "
                f"its check (F05).")
        if _tol_f < 0.0:
            raise PhysicsCalculationError(
                f"{context}: NUM-SS-001 tolerance '{_tol_name}' must be "
                f"nonnegative, got {_tol_f!r}.")

    if rho is None:
        raise PhysicsCalculationError(
            f"{context}: the steady-state solver returned None instead of a "
            f"density matrix.")

    mat = rho.full() if hasattr(rho, "full") and callable(rho.full) else rho
    try:
        mat = np.asarray(mat)
        if mat.ndim == 0:
            raise ValueError("0-dimensional steady-state candidate")
        mat = mat.astype(complex, copy=False)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"{context}: the steady-state candidate is not a numeric 2-D "
            f"matrix (type={type(rho).__name__}).") from exc

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 1:
        raise PhysicsCalculationError(
            f"{context}: the steady-state candidate has shape {mat.shape}; "
            f"a square (N, N) density matrix is required.")

    finite = np.isfinite(mat)
    if not finite.all():
        raise PhysicsCalculationError(
            f"{context}: the steady-state density matrix contains "
            f"{int(mat.size - np.count_nonzero(finite))} NaN/Inf "
            f"entry/entries.")

    n = int(mat.shape[0])
    L = liouvillian.full() if (hasattr(liouvillian, "full")
                               and callable(liouvillian.full)) else liouvillian
    from scipy import sparse
    sparse_l = sparse.issparse(L)
    if not sparse_l:
        L = np.asarray(L)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise PhysicsCalculationError(
            f"{context}: the Liouvillian has shape {L.shape}; a square "
            f"(N^2, N^2) matrix is required.")
    if L.shape != (n * n, n * n):
        raise PhysicsCalculationError(
            f"{context}: the Liouvillian shape {L.shape} does not match the "
            f"{n}x{n} density matrix (expected {(n * n, n * n)}).")
    try:
        L = L.astype(complex, copy=False)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"{context}: the Liouvillian is not a numeric matrix.") from exc
    if not np.all(np.isfinite(L.data if sparse_l else L)):
        raise PhysicsCalculationError(
            f"{context}: the Liouvillian contains NaN/Inf entries.")

    # ---- metrics, all computed together so a failure reports everything ----
    trace_error = float(abs(np.trace(mat) - 1.0))
    # hermiticity_error is measured first.  The positivity diagnostic (min
    # eigenvalue of the Hermitian part) is computed UNCONDITIONALLY so it can
    # be reported, but the positivity REJECTION below is applied ONLY when the
    # candidate is Hermitian within tolerance — a non-Hermitian matrix is
    # never accepted via its Hermitian part.  The returned state is NEVER
    # replaced by the Hermitian part (NUM-SS-001 / -R1: no repair).
    hermiticity_error = float(np.max(np.abs(mat - mat.conj().T)))
    hermitian_part = 0.5 * (mat + mat.conj().T)
    try:
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_part)))
    except np.linalg.LinAlgError as exc:
        raise PhysicsCalculationError(
            f"{context}: the eigendecomposition of the Hermitian part of the "
            f"steady-state candidate failed ({exc}); positivity cannot be "
            f"certified.") from exc
    residual = float(np.linalg.norm(L @ mat.flatten(order="F")))
    liou_norm = float(sparse.linalg.norm(L) if sparse_l else np.linalg.norm(L))
    residual_rel = (residual / liou_norm if liou_norm > 0.0
                    else (0.0 if residual == 0.0 else float("inf")))

    # Fail loudly if any COMPUTED diagnostic is non-finite (F05): a NaN metric
    # compared against a tolerance would otherwise silently PASS its check.
    # ``residual_rel`` is excluded: ``float("inf")`` is its documented sentinel
    # when ||L||=0 with a non-stationary rho, and it is caught by the relative
    # tolerance comparison below.
    for _metric_name, _metric in (
        ("trace_error", trace_error),
        ("hermiticity_error", hermiticity_error),
        ("min_eigenvalue", min_eigenvalue),
        ("stationarity_residual", residual),
        ("liouvillian_frobenius_norm", liou_norm),
    ):
        if not math.isfinite(_metric):
            raise PhysicsCalculationError(
                f"{context}: NUM-SS-001 computed a non-finite {_metric_name}="
                f"{_metric!r} (backend="
                f"{backend if backend is not None else '<unspecified>'}); the "
                f"steady-state candidate cannot be certified and the value was "
                f"NOT repaired or replaced (F05).")

    violated = []
    if trace_error > trace_tolerance:
        violated.append("|Tr(rho)-1| exceeds the trace tolerance")
    hermiticity_ok = hermiticity_error <= hermiticity_tolerance
    if not hermiticity_ok:
        violated.append("rho is not Hermitian")
    # Positivity is judged from the Hermitian part ONLY for a candidate that is
    # Hermitian within tolerance; a non-Hermitian matrix is already rejected
    # above, so its Hermitian-part eigenvalue is never used to accept it
    # (NUM-SS-001-R1).
    if hermiticity_ok and min_eigenvalue < -positivity_tolerance:
        violated.append("rho has a negative eigenvalue")
    if residual > stationarity_abs_tolerance:
        violated.append("|L[rho]| exceeds the absolute stationarity tolerance")
    if residual_rel > stationarity_rel_tolerance:
        violated.append("|L[rho]|/|L| exceeds the relative stationarity "
                        "tolerance")

    backend_label = backend if backend is not None else "<unspecified>"
    if violated:
        raise PhysicsCalculationError(
            f"{context}: NUM-SS-001 rejected the steady-state density matrix "
            f"(backend={backend_label}) because " + "; ".join(violated) +
            f". trace_error={trace_error:.3e}, hermiticity_error="
            f"{hermiticity_error:.3e}, min_eigenvalue={min_eigenvalue:.3e}, "
            f"stationarity_residual={residual:.3e} (relative "
            f"{residual_rel:.3e}), ||L||_F={liou_norm:.3e}; tolerances: "
            f"trace {trace_tolerance:.0e}, hermiticity "
            f"{hermiticity_tolerance:.0e}, positivity "
            f"{positivity_tolerance:.0e}, stationarity "
            f"{stationarity_abs_tolerance:.0e} absolute / "
            f"{stationarity_rel_tolerance:.0e} relative. The value was NOT "
            f"repaired, renormalized or replaced (F05).")

    metrics = {
        "trace_error": trace_error,
        "hermiticity_error": hermiticity_error,
        "min_eigenvalue": min_eigenvalue,
        "stationarity_residual": residual,
        "stationarity_residual_rel": residual_rel,
        "liouvillian_frobenius_norm": liou_norm,
        "dimension": n,
        "backend": backend_label,
    }
    logger.debug(
        "NUM-SS-001 accepted %s (backend=%s): trace=%.3e herm=%.3e "
        "min_eig=%.3e |L[rho]|=%.3e rel=%.3e ||L||_F=%.3e",
        context, backend_label, trace_error, hermiticity_error,
        min_eigenvalue, residual, residual_rel, liou_norm)
    return metrics


