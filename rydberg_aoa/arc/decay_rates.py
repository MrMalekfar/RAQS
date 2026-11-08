"""
Calculate atomic decay rates using ARC database.

Provides functions for computing Einstein A coefficients for atomic transitions
using dynamic level configurations determined by the level finder.

RF-DIR-001B (energy-ordered RF edge)
------------------------------------
The ``|3> <-> |4>`` (R1 <-> R2) edge is indexed by DRIVING topology, not by
energy, so it is NOT evaluated in ladder order: the production pair
(Rb85 26D5/2 vs 27P3/2) is a DOWNWARD edge.  ``calculate_decay_rates``
therefore returns ``gamma_43`` as the spontaneous-emission rate of the
ENERGY-DOWNWARD channel, evaluated at ARC's default temperature (0.0 K, i.e.
no black-body absorption term), and ``resolve_rydberg_orientation`` exposes
the direction that ``create_sensor_model(rf_orientation=...)`` needs to build
the top-to-bottom jump operator.  All rates are event rates in 1/us
(s^-1 / 1e6); no 2*pi factor is applied anywhere in this module.

F05-DECAY-001-R1 (strict real-scalar rate contract)
--------------------------------------------------
Every ARC-derived transition rate must already BE a real numeric scalar
(Python ``int``/``float`` or the NumPy real scalar types ARC returns:
``np.integer``/``np.floating``) BEFORE any conversion is attempted.  A
successful ``float()`` is never treated as proof of a valid type: it accepts
``True``, ``"1.0"``, ``np.array([1.0])`` and — worst of all — silently discards
the imaginary part of ``np.complex128(1+2j)`` under a ComplexWarning.  Those
inputs are rejected with ``PhysicsCalculationError`` instead.  No epsilon,
clip, ``abs``, replacement rate or warning-and-continue is ever substituted
for a physical rate, and a valid zero / below-floor value is preserved.
"""

import math
from typing import Tuple

import numpy as np

from ..exceptions import PhysicsCalculationError, PhysicsError
from .rf_orientation import RFTransitionOrientation, resolve_rf_orientation
from .types import AtomicLevel


def _level_label(level):
    """Deterministic (n, l, j) label used only for error context."""
    return f"n={level.n}, l={level.l}, j={level.j}"


def _is_supported_real_scalar(value) -> bool:
    """Is ``value`` already a supported real numeric scalar? (F05-DECAY-001-R1)

    The type is checked BEFORE any ``float()`` conversion, because conversion
    success is not proof of a valid type.

    ACCEPTED — real numeric scalars, i.e. exactly what ARC returns:

    * Python ``int`` / ``float`` (note: ``bool`` is an ``int`` subclass and is
      therefore rejected explicitly below);
    * NumPy real scalar types ``np.integer`` / ``np.floating``
      (``np.int64``, ``np.float32``, ``np.float16``, …).  ``np.bool_`` is
      neither, so it is rejected here as well.

    REJECTED — everything else, including inputs that ``float()`` would happily
    coerce:

    * Python and NumPy booleans (``bool``, ``np.bool_``);
    * ``str`` / ``bytes`` (``float("1.0")`` would succeed);
    * complex scalars (``complex``, ``np.complexfloating``) — including
      ``np.complex128(1+0j)`` and ``np.complex128(1+2j)``, whose imaginary part
      ``float()`` silently discards under a ComplexWarning;
    * lists, tuples and NumPy arrays, including 0-d and one-element arrays;
    * arbitrary objects accepted only through ``__float__``.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.bool_, complex)):
        return False
    return isinstance(value, (np.integer, np.floating))


def _validated_rate_per_us(rate_s, transition):
    """Validate an ARC transition rate and convert s^-1 -> 1/us (F05-DECAY-001).

    The input type is validated BEFORE conversion (F05-DECAY-001-R1): a real
    numeric scalar must already be supplied, so a boolean, a numeric string, an
    array or a complex value whose imaginary part would be silently dropped can
    never masquerade as a rate.

    No floor is applied: a valid zero is returned exactly, and a valid positive
    value below the removed 1e-6 guard is preserved.  A non-scalar, non-finite
    or negative ARC result fails loudly with the transition and offending value;
    no decay rate is invented and no epsilon/clip/alternate is substituted.
    """
    if not _is_supported_real_scalar(rate_s):
        raise PhysicsCalculationError(
            f"ARC getTransitionRate returned an unsupported type for the "
            f"{transition} transition: {rate_s!r} "
            f"(type {type(rate_s).__name__}); a real numeric scalar is "
            f"required and no decay rate is invented (F05)."
        )
    try:
        value = float(rate_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicsCalculationError(
            f"ARC getTransitionRate returned an unconvertible value for the "
            f"{transition} transition: {rate_s!r}; "
            f"no decay rate is invented (F05)."
        ) from exc
    if not math.isfinite(value):
        raise PhysicsCalculationError(
            f"ARC getTransitionRate returned a non-finite value for the "
            f"{transition} transition: {value} s^-1; no decay rate is invented (F05)."
        )
    if value < 0.0:
        raise PhysicsCalculationError(
            f"ARC getTransitionRate returned a negative rate for the "
            f"{transition} transition: {value} s^-1; no decay rate is invented (F05)."
        )
    # event rate s^-1 -> 1/us (÷ 1e6; NO 2π — a decay rate, not an angular rate)
    return value / 1e6


def resolve_rydberg_orientation(atom, rydberg1_level, rydberg2_level) -> RFTransitionOrientation:
    """Resolve the energy orientation of the ``|3> <-> |4>`` RF edge.

    RF-DIR-001A/B single source of direction: the signed ARC transition
    frequency is taken in LADDER order, ``nu = (E_R2 - E_R1)/h``, exactly as
    ``atom.getTransitionFrequency(rydberg1..., rydberg2...)`` returns it, and
    handed to :func:`resolve_rf_orientation` — the direction is never inferred
    from the ladder index and never substituted (F05).

    Args:
        atom: ARC Rubidium85 atom object
        rydberg1_level: R1 dict or AtomicLevel (ladder |3>)
        rydberg2_level: R2 dict or AtomicLevel (ladder |4>)

    Returns:
        RFTransitionOrientation (signed frequency, direction in {-1, +1},
        carrier |nu|, upper/lower ladder indices).

    Raises:
        ValueError: from ``resolve_rf_orientation`` for a degenerate or
            non-finite carrier (zero, NaN, inf).
        Exception: ARC failures propagate unchanged (fail loudly; nothing is
            inferred on failure).
    """
    rydberg1 = AtomicLevel(**rydberg1_level) if isinstance(rydberg1_level, dict) else rydberg1_level
    rydberg2 = AtomicLevel(**rydberg2_level) if isinstance(rydberg2_level, dict) else rydberg2_level
    nu = atom.getTransitionFrequency(
        rydberg1.n, rydberg1.l, rydberg1.j,
        rydberg2.n, rydberg2.l, rydberg2.j,
    )
    return resolve_rf_orientation(nu)


def calculate_decay_rates(atom, ground_level, excited_level, rydberg1_level, rydberg2_level) -> Tuple[float, float, float]:
    """
    Calculate Einstein A coefficients using ARC for the 4-level EIT system.

    Uses dynamically determined atomic levels instead of hardcoded quantum numbers.
    Returns event rates in 1/μs (s^-1 / 1e6; no 2π — these are decay rates,
    not angular frequencies).

    RF-DIR-001B (energy-ordered RF edge): the two optical channels keep their
    ladder order (energy-ascending for this system), but the |3> <-> |4> RF
    edge is evaluated in ENERGY order — gamma_43 is the spontaneous-emission
    rate from the energetically HIGHER Rydberg state into the LOWER one (for
    the production pair: R1(26D5/2) -> R2(27P3/2)).  The direction comes from
    the signed ARC transition frequency through ``resolve_rf_orientation``
    (single source of truth; RF-DIR-001A).  ARC's default temperature (0.0 K)
    is used, so the RF-edge channel is purely spontaneous (no black-body
    absorption term).

    Args:
        atom: ARC Rubidium85 atom object
        ground_level: Ground state dict or AtomicLevel
        excited_level: Excited state dict or AtomicLevel
        rydberg1_level: First Rydberg state dict or AtomicLevel
        rydberg2_level: Second Rydberg state dict or AtomicLevel

    Returns:
        Tuple of (gamma_21, gamma_32, gamma_43) in 1/μs:
        - gamma_21: excited → ground decay rate
        - gamma_32: rydberg1 → excited decay rate
        - gamma_43: energy-downward RF-edge spontaneous rate (R1 → R2 for the
          production downward pair); no floor — a valid zero stays zero (F05-DECAY-001)

    Raises:
        PhysicsCalculationError: If an ARC call fails, or an ARC transition
            rate is not a real numeric scalar, non-finite or negative
            (F05-DECAY-001 / F05-DECAY-001-R1).  The failing operation, the
            transition states and the offending value are named, the original
            exception is chained via ``from exc``, and no decay rate is
            invented.
    """
    # Convert dictionary parameters to AtomicLevel objects if needed
    ground = AtomicLevel(**ground_level) if isinstance(ground_level, dict) else ground_level
    excited = AtomicLevel(**excited_level) if isinstance(excited_level, dict) else excited_level
    rydberg1 = AtomicLevel(**rydberg1_level) if isinstance(rydberg1_level, dict) else rydberg1_level
    rydberg2 = AtomicLevel(**rydberg2_level) if isinstance(rydberg2_level, dict) else rydberg2_level

    def _arc_rate(operation, upper_level, lower_level):
        """One ARC transition-rate call with its own failure context (F05).

        The failing operation and both states are named in the message, and the
        original exception is chained as ``__cause__`` so nothing is lost.  An
        already-fail-fast ``PhysicsError`` is propagated unchanged.
        """
        try:
            return atom.getTransitionRate(
                upper_level.n, upper_level.l, upper_level.j,
                lower_level.n, lower_level.l, lower_level.j,
            )
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError(
                f"ARC getTransitionRate failed for the {operation} transition "
                f"({_level_label(upper_level)} -> {_level_label(lower_level)}): "
                f"{exc!r}; no decay rate is invented (F05)."
            ) from exc

    # Γ₂₁: excited → ground transition rate (s^-1)
    A_21 = _arc_rate("excited -> ground", excited, ground)

    # Γ₃₂: rydberg1 → excited transition rate (s^-1)
    A_32 = _arc_rate("rydberg1 -> excited", rydberg1, excited)

    # Γ₄₃ (RF-DIR-001B): spontaneous emission of the ENERGY-DOWNWARD
    # |3> <-> |4> edge.  The direction is resolved from the signed ARC
    # transition frequency (ladder order) — never from the ladder index.
    try:
        orientation = resolve_rydberg_orientation(atom, rydberg1, rydberg2)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC getTransitionFrequency failed for the R1 <-> R2 orientation "
            f"({_level_label(rydberg1)} <-> {_level_label(rydberg2)}): "
            f"{exc!r}; the RF decay direction cannot be determined and no "
            f"decay rate is invented (F05)."
        ) from exc
    levels_by_index = {3: rydberg1, 4: rydberg2}
    upper = levels_by_index[orientation.upper_ladder_index]
    lower = levels_by_index[orientation.lower_ladder_index]
    # The spontaneous-emission call is ALWAYS energy-downward (upper -> lower);
    # the ladder orientation only decides which of R1/R2 plays which role.
    A_down = _arc_rate(
        "energy-downward RF-edge spontaneous emission", upper, lower
    )

    # F05-DECAY-001: validate each ARC-derived rate and convert s^-1 -> 1/μs
    # (÷ 1e6; NO 2π — a decay rate is not an angular frequency).  A valid zero
    # stays exactly zero; a valid value below the removed 1e-6 floor is kept.
    # Negative / NaN / Inf / non-scalar values fail loudly — no floor, no
    # substitution, no invented rate: numerical solvability is never obtained
    # by changing physics (the legacy ``max(gamma_43, 1e-6)`` guard is removed).
    gamma_21 = _validated_rate_per_us(
        A_21, f"excited → ground ({_level_label(excited)} → {_level_label(ground)})"
    )
    gamma_32 = _validated_rate_per_us(
        A_32, f"rydberg1 → excited ({_level_label(rydberg1)} → {_level_label(excited)})"
    )
    # The spontaneous-emission channel is ALWAYS energy-downward (upper ->
    # lower); the ladder orientation is reported separately so this label can
    # never suggest an "energy-upward" spontaneous channel (F05-DECAY-001-R1).
    _ladder = "ladder-upward" if orientation.is_upward else "ladder-downward"
    gamma_43 = _validated_rate_per_us(
        A_down,
        f"energy-downward RF-edge spontaneous emission "
        f"({_level_label(upper)} -> {_level_label(lower)}; {_ladder})",
    )

    return gamma_21, gamma_32, gamma_43
