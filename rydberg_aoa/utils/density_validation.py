"""NUM-TD-001 — fail-loud certification of temporal density matrices.

A time-dependent solver *returning* a trajectory (or expectation values) is
not proof that each sampled state is a valid density matrix.  A corrupted,
non-trace-one, non-Hermitian or non-positive intermediate state would
otherwise flow straight into ``qt.expect(obs, rho)`` -> the heterodyne
observable -> IF phase -> AoA, with nothing re-checking it.  This module
certifies each sampled temporal state BEFORE it becomes an observable.

It is the temporal sibling of ``rydberg_aoa.utils.validation.validate_steady_
state``: it enforces the SAME trace / Hermiticity / positivity tolerances and
the same non-mutating, fail-loud contract, but it deliberately does NOT test
stationarity (an evolving state is not a steady state) and needs no
Liouvillian.

Passing this gate proves the sampled density matrix is valid within the
documented tolerance.  It does NOT prove integrator convergence, EIT,
uniqueness, or AoA accuracy.
"""

import math

import numpy as np

from ..exceptions import PhysicsCalculationError
from .validation import (
    STEADY_STATE_HERMITICITY_TOLERANCE,
    STEADY_STATE_POSITIVITY_TOLERANCE,
    STEADY_STATE_TRACE_TOLERANCE,
)


def _validate_tolerance(name, value, context):
    """Validate a tolerance: real numeric scalar, finite, nonnegative.

    Rejects NaN/Inf/negative/non-real/non-scalar types and conversion
    overflow (F05): a mis-set tolerance must never silently disable its own
    comparison.  Zero is a legitimate strictest tolerance.
    """
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 tolerance '{name}' must be a real numeric "
            f"scalar, got {value!r} (type={type(value).__name__}).")
    try:
        tol = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 tolerance '{name}' cannot be converted to "
            f"a real float, got {value!r} ({exc}).") from exc
    if not math.isfinite(tol):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 tolerance '{name}' must be finite, got "
            f"{tol!r}.")
    if tol < 0.0:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 tolerance '{name}' must be nonnegative, "
            f"got {tol!r}.")
    return tol


def validate_density_matrix(rho, *, context, expected_dim=None,
                            trace_tolerance=STEADY_STATE_TRACE_TOLERANCE,
                            hermiticity_tolerance=STEADY_STATE_HERMITICITY_TOLERANCE,
                            positivity_tolerance=STEADY_STATE_POSITIVITY_TOLERANCE):
    """Certify that ``rho`` is a valid (temporal) density matrix.

    Non-mutating and fail-loud (F05): it only MEASURES trace, Hermiticity and
    positivity and either certifies or rejects; it never normalizes, clips,
    symmetrizes or replaces the value.  Stationarity is deliberately NOT
    tested (an evolving state is not a steady state), and no Liouvillian is
    required.

    Parameters
    ----------
    rho : qutip.Qobj or array-like
        Candidate density matrix, square ``(N, N)``.
    context : str
        Where the value came from; reproduced verbatim in failure messages.
    expected_dim : int, optional
        When given, the matrix must be exactly ``(expected_dim, expected_dim)``.
    trace/hermiticity/positivity tolerances : float
        Reused unchanged from the NUM-SS contract.

    Returns
    -------
    dict
        Measured metrics: ``trace_error``, ``hermiticity_error``,
        ``min_eigenvalue``, ``dimension``.
    """
    trace_tol = _validate_tolerance("trace_tolerance", trace_tolerance, context)
    herm_tol = _validate_tolerance("hermiticity_tolerance",
                                   hermiticity_tolerance, context)
    pos_tol = _validate_tolerance("positivity_tolerance",
                                  positivity_tolerance, context)

    if rho is None:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 received None instead of a density matrix.")

    mat = rho.full() if hasattr(rho, "full") and callable(rho.full) else rho
    try:
        arr = np.asarray(mat)
        if arr.ndim == 0:
            raise ValueError("0-dimensional density matrix")
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate is not a numeric 2-D matrix "
            f"(type={type(rho).__name__}).") from exc

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate has shape {arr.shape}; a square "
            f"(N, N) density matrix is required.")
    if expected_dim is not None and arr.shape != (expected_dim, expected_dim):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate has shape {arr.shape}; expected "
            f"({expected_dim}, {expected_dim}).")

    # reject bool/object/string BEFORE complex conversion can disguise type
    if arr.dtype == np.bool_ or arr.dtype == object \
            or arr.dtype.kind in ("U", "S", "V"):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate is non-numeric/boolean (dtype "
            f"{arr.dtype}); it is rejected before conversion.")
    if not np.issubdtype(arr.dtype, np.number):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate is non-numeric (dtype "
            f"{arr.dtype}).")
    try:
        arr = arr.astype(complex, copy=False)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate is not complex-convertible.") \
            from exc

    if not np.all(np.isfinite(arr)):
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 candidate contains NaN/Inf entries.")

    trace_error = float(abs(np.trace(arr) - 1.0))
    hermiticity_error = float(np.max(np.abs(arr - arr.conj().T)))
    hermitian_part = 0.5 * (arr + arr.conj().T)
    try:
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_part)))
    except np.linalg.LinAlgError as exc:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 eigendecomposition of the Hermitian part "
            f"failed ({exc}); positivity cannot be certified.") from exc

    for name, value in (("trace_error", trace_error),
                        ("hermiticity_error", hermiticity_error),
                        ("min_eigenvalue", min_eigenvalue)):
        if not math.isfinite(value):
            raise PhysicsCalculationError(
                f"{context}: NUM-TD-001 computed a non-finite {name}={value!r}; "
                f"the candidate cannot be certified (F05).")

    violated = []
    if trace_error > trace_tol:
        violated.append(
            f"|Tr(rho)-1|={trace_error:.3e} exceeds {trace_tol:.1e}")
    if hermiticity_error > herm_tol:
        violated.append(
            f"Hermiticity error={hermiticity_error:.3e} exceeds {herm_tol:.1e}")
    if min_eigenvalue < -pos_tol:
        violated.append(
            f"min eigenvalue={min_eigenvalue:.3e} below -{pos_tol:.1e}")
    if violated:
        raise PhysicsCalculationError(
            f"{context}: NUM-TD-001 rejected the temporal density matrix "
            f"because " + "; ".join(violated) +
            f". The value was NOT normalized, clipped, symmetrized or "
            f"replaced (F05).")

    return {
        "trace_error": trace_error,
        "hermiticity_error": hermiticity_error,
        "min_eigenvalue": min_eigenvalue,
        "dimension": int(arr.shape[0]),
    }

