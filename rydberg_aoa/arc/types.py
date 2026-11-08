"""
Type definitions and data classes for ARC Interface.

Contains all the data structures used for representing atomic levels,
laser parameters, RF parameters, and transitions.
"""

import math
from typing import Optional
from dataclasses import dataclass

from ..exceptions import PhysicsConfigurationError

# Numerical tolerance for the integer-step test of a magnetic quantum number
# (``j - m`` must be an integer; ``j`` may be integer or half-integer).
_QN_INT_STEP_TOL = 1e-9


def validate_magnetic_quantum_number(m_value, j_total, m_name: str,
                                     j_name: str) -> None:
    """Validate ONE magnetic quantum number against its total angular momentum.

    QN-001A physics contract
    ------------------------
    ``mj`` is the projection of the electronic total angular momentum ``j`` and
    ``mf`` is the projection of the hyperfine total angular momentum ``f``.
    They are DIFFERENT quantum numbers: neither may be inferred from the other,
    and neither receives a default value.

        |m| <= j                    (a projection cannot exceed its total)
        j - m must be an integer    (the step structure of angular momenta)

    Consequences enforced here (and pinned by the QN-001A tests):

        j = 1/2  ->  m = ±1/2                (m = 0 does NOT exist)
        j = 3/2  ->  m = ±1/2, ±3/2
        j = 2.5  ->  m = ±0.5, ±1.5, ±2.5
        f = 3.0  ->  m = 0, ±1, ±2, ±3       (integer F -> integer mF)

    Args:
        m_value: the magnetic quantum number (``mj`` or ``mf``).
        j_total: the corresponding total angular momentum (``j`` or ``f``).
        m_name: name used in the error message (``"mj"`` or ``"mf"``).
        j_name: name used in the error message (``"j"`` or ``"f"``).

    Raises:
        PhysicsConfigurationError: if either value is non-numeric/non-finite,
            if ``|m| > j``, or if ``j - m`` is not an integer.
    """
    try:
        m = float(m_value)
        j = float(j_total)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"{m_name}={m_value!r} / {j_name}={j_total!r} are not numeric: the "
            "quantum numbers must be real numbers. "
            "QN-001A: no implicit magnetic quantum number."
        ) from exc

    if not (math.isfinite(m) and math.isfinite(j)):
        raise PhysicsConfigurationError(
            f"{m_name}={m!r} with {j_name}={j!r} is not finite. "
            "QN-001A: no implicit magnetic quantum number."
        )

    if abs(m) > j + _QN_INT_STEP_TOL:
        raise PhysicsConfigurationError(
            f"|{m_name}|={abs(m)} exceeds {j_name}={j}: the projection of an "
            "angular momentum can never be larger than the angular momentum "
            "itself. QN-001A: no implicit magnetic quantum number."
        )

    if abs((j - m) - round(j - m)) > _QN_INT_STEP_TOL:
        step = ("half-integer" if abs(j - round(j)) > _QN_INT_STEP_TOL
                else "integer")
        raise PhysicsConfigurationError(
            f"{m_name}={m} is not a valid projection of {j_name}={j}: the "
            f"allowed values differ from {j_name} by an integer, so "
            f"{j_name}={j} admits only {step} values of {m_name}. "
            "QN-001A: never infer and never default a magnetic quantum number."
        )


@dataclass
class LaserParams:
    """Parameters for a laser field."""
    wavelength_nm: float
    power_mw: float
    beam_waist_mm: float
    detuning_mhz: float
    polarization: int


@dataclass
class RFParams:
    """Parameters for RF field."""
    frequency_hz: float
    electric_field_vm: float
    polarization: int


@dataclass
class AtomicLevel:
    """Represents an atomic energy level."""
    n: int
    l: int
    j: float
    f: Optional[float] = None
    mj: Optional[float] = None   # projection of j — valid only if supplied
    mf: Optional[float] = None   # projection of f — valid only if supplied
    energy: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        """Validate the supplied quantum numbers — never manufacture one.

        QN-001A (removed defect): the historical implementation forced

            mj := mf   (if mf was given)   else   mj := 0.0

        which conflated two different quantum numbers and injected the invalid
        sublevel ``mj = 0`` for every half-integer ``j`` (e.g. the configured
        ``5S1/2, j = 1/2`` ground level).  ARC's ``getRabiFrequency`` then
        returned exactly ``0.0`` for the driven probe transition.

        ``mj`` and ``mf`` are now independent OPTIONAL quantum numbers: they are
        validated when present and left as ``None`` when absent, so a
        fine-structure calculation can require an explicit ``mj`` (fail-fast)
        instead of silently receiving a fabricated one.
        """
        if self.mj is not None:
            validate_magnetic_quantum_number(self.mj, self.j, "mj", "j")

        if self.mf is not None:
            if self.f is None:
                raise PhysicsConfigurationError(
                    f"mf={self.mf!r} was supplied for level "
                    f"{self.label or f'{self.n}{self.l}{self.j}'} without f: mf "
                    "is the projection of the hyperfine quantum number F and "
                    "cannot be validated (or used) without it. "
                    "QN-001A: no implicit f, no implicit mj."
                )
            validate_magnetic_quantum_number(self.mf, self.f, "mf", "f")

@dataclass
class Transition:
    """Represents a transition between two levels."""
    level1: AtomicLevel
    level2: AtomicLevel
    wavelength_nm: float
    dipole_moment: float
    rabi_frequency: float
    detuning: float
    linewidth: float
    frequency_diff_hz: Optional[float] = 0.0
