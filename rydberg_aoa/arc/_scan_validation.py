"""Strict scalar boundaries for RF-first ARC selection (LAB-001)."""

import math
import numpy as np

from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


def real_scalar(value, name, *, error_type=PhysicsConfigurationError):
    """Accept finite Python/NumPy real scalars without conversion-only types."""
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating))):
        raise error_type(f"LAB-001: {name} must be a real numeric scalar; "
                         f"got {value!r}. F05: no coerced or substituted value.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(f"LAB-001: {name} cannot be represented as a finite "
                         "real scalar. F05: no substituted value.") from exc
    if not math.isfinite(result):
        raise error_type(f"LAB-001: {name} must be finite; got {value!r}. "
                         "F05: no substituted value.")
    return result


def integer_scalar(value, name, *, minimum=0):
    """Require integer domains; never truncate a principal quantum number."""
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise PhysicsConfigurationError(
            f"LAB-001: {name} must be an integer; got {value!r}. F05.")
    result = int(value)
    if result < minimum:
        raise PhysicsConfigurationError(
            f"LAB-001: {name} must be >= {minimum}; got {value!r}. F05.")
    return result


def arc_real_scalar(value, context, *, positive=False, nonzero=False):
    """Validate an ARC result before float(), with its transition context."""
    result = real_scalar(value, context, error_type=PhysicsCalculationError)
    if (positive and result <= 0) or (nonzero and result == 0):
        requirement = "positive" if positive else "nonzero"
        raise PhysicsCalculationError(
            f"LAB-001: {context} must be {requirement}; got {value!r}. "
            "F05: no substituted ARC result.")
    return result
