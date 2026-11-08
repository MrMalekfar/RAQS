"""RF-DIR-001 — the energy direction of the Rydberg RF edge.

Why this module exists
----------------------
The four-level cascade is indexed by **driving topology**, not by energy:

    |1> --probe--> |2> --coupling--> |3> <--RF--> |4>

The project's own reconstruction (``docs/03_energy_levels_transitions_detuning.md``,
section 4) establishes that the production pair

    |1> 5S1/2  <  |2> 5P3/2  <  |4> 27P3/2  <  |3> 26D5/2

is a **downward** RF edge: R1 (26D5/2) lies *above* R2 (27P3/2) in energy, so
``state_index_increases`` does **not** imply ``energy_increases``.

Two facts follow, and both are silent failure modes if left implicit:

1. **Carrier.**  The physical carrier of the RF field is the *magnitude*
   ``|nu|`` of the signed ARC transition frequency ``nu = (E_R2 - E_R1)/h``.
   ``nu`` itself is negative for this pair.  A detuning must be taken against
   ``|nu|`` (magnitude against magnitude); using the signed ``nu`` would put
   the resonance at ``2|nu|``.

2. **Decay direction.**  Spontaneous emission on the RF edge always runs from
   the energetic **top** of the ladder to the **bottom**.  For an ascending
   ladder that is ``|4> -> |3>`` (jump operator ``|3><4|``); for this
   downward edge it is ``|3> -> |4>`` (jump operator ``|4><3|``).  Building
   the collapse operator by ladder index alone therefore *corrupts the
   Lindblad dissipator* for a downward edge.

MODE-001 section 15 (``config/atomic_config.py``) and ``docs/03`` section 4.3
deliberately deferred this — "establishing the true energy ordering is a
separate, explicitly documented follow-up rather than an implicit repair".
RF-DIR-001 is that follow-up.

Conventions (RF-DIR-001)
------------------------
``direction == +1``  ascending ladder, ``E(R2) > E(R1)`` (upward RF edge)
``direction == -1``  descending ladder, ``E(R2) < E(R1)`` (downward RF edge)

The signed frequency is *always* taken in ladder order,
``nu = (E_ladder4 - E_ladder3)/h`` — exactly what ARC's
``getTransitionFrequency(rydberg1..., rydberg2...)`` returns.

Evidence
--------
``phase0_baseline.json``:  E(rydberg1 = 26D5/2) = -0.02238334263680137 eV,
E(rydberg2 = 27P3/2) = -0.022931935266660975 eV  =>  nu < 0, direction = -1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

__all__ = [
    "RFTransitionOrientation",
    "resolve_rf_orientation",
    "UPWARD_RF_DIRECTION",
    "DOWNWARD_RF_DIRECTION",
]

UPWARD_RF_DIRECTION = 1
DOWNWARD_RF_DIRECTION = -1

#: Ladder indices of the two Rydberg manifolds (|3> = R1, |4> = R2).
_R1_LADDER_INDEX = 3
_R2_LADDER_INDEX = 4


@dataclass(frozen=True)
class RFTransitionOrientation:
    """Energy orientation of the ``|3> <-> |4>`` (R1 <-> R2) RF edge.

    Instances satisfy, by construction:

    * ``carrier_hz > 0``                          (a real field carrier)
    * ``direction in (-1, +1)``                   (never zero / never guessed)
    * ``carrier_hz == abs(signed_frequency_hz)``  (magnitude consistency)

    Attributes
    ----------
    signed_frequency_hz : float
        ``nu = (E_R2 - E_R1)/h`` in Hz — **signed**, taken in ladder order.
    direction : int
        ``+1`` when ``E(R2) > E(R1)`` (ascending ladder); ``-1`` when
        ``E(R2) < E(R1)`` (descending ladder).
    """

    signed_frequency_hz: float
    direction: int

    def __post_init__(self) -> None:
        nu = float(self.signed_frequency_hz)
        if not math.isfinite(nu):
            raise ValueError(
                f"RF edge signed frequency is not finite ({nu!r}). RF-DIR-001: "
                "a non-physical RF carrier is never substituted."
            )
        if nu == 0.0:
            raise ValueError(
                "RF edge signed frequency is exactly zero; the |3> <-> |4> "
                "transition would be degenerate and its energy direction is "
                "undefined. RF-DIR-001: no direction is inferred from a "
                "degenerate pair."
            )
        if self.direction not in (UPWARD_RF_DIRECTION, DOWNWARD_RF_DIRECTION):
            raise ValueError(
                f"RF edge energy direction must be -1 or +1 (got "
                f"{self.direction!r}). RF-DIR-001: the direction is derived "
                "from the signed ARC frequency and is never guessed."
            )
        expected = UPWARD_RF_DIRECTION if nu > 0.0 else DOWNWARD_RF_DIRECTION
        if self.direction != expected:
            raise ValueError(
                f"RF edge energy direction {self.direction!r} contradicts the "
                f"signed frequency {nu!r} Hz (expected {expected!r}). "
                "RF-DIR-001: direction and signed frequency must agree."
            )

    # -- carrier ---------------------------------------------------------
    @property
    def carrier_hz(self) -> float:
        """Positive RF carrier magnitude ``|nu|`` in Hz."""
        return abs(self.signed_frequency_hz)

    # -- geometry of the edge --------------------------------------------
    @property
    def is_upward(self) -> bool:
        """True when ``E(R2) > E(R1)`` (ascending ladder)."""
        return self.direction == UPWARD_RF_DIRECTION

    @property
    def is_downward(self) -> bool:
        """True when ``E(R2) < E(R1)`` (descending ladder)."""
        return self.direction == DOWNWARD_RF_DIRECTION

    @property
    def upper_ladder_index(self) -> int:
        """Ladder index (3 or 4) of the energetically HIGHER state."""
        return _R2_LADDER_INDEX if self.is_upward else _R1_LADDER_INDEX

    @property
    def lower_ladder_index(self) -> int:
        """Ladder index (3 or 4) of the energetically LOWER state."""
        return _R1_LADDER_INDEX if self.is_upward else _R2_LADDER_INDEX

    @property
    def upper_role(self) -> str:
        """``'R1'`` or ``'R2'`` — label of the energetically higher state."""
        return "R2" if self.is_upward else "R1"

    @property
    def lower_role(self) -> str:
        """``'R1'`` or ``'R2'`` — label of the energetically lower state."""
        return "R1" if self.is_upward else "R2"

    # -- decay channel ---------------------------------------------------
    @property
    def decay_operator_name(self) -> str:
        """RF spontaneous-emission jump operator, as named in
        ``create_sensor_model``.

        ``'s34'`` == ``|3><4|``  (ascending ladder — the historical form)
        ``'s43'`` == ``|4><3|``  (descending ladder)
        """
        return "s34" if self.is_upward else "s43"

    @property
    def decay_description(self) -> str:
        """Human-readable decay channel, e.g. ``'R1(3) -> R2(4)'``."""
        return (
            f"{self.upper_role}({self.upper_ladder_index}) -> "
            f"{self.lower_role}({self.lower_ladder_index})"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RFTransitionOrientation(signed={self.signed_frequency_hz:.6e} Hz, "
            f"carrier={self.carrier_hz:.6e} Hz, direction={self.direction:+d}, "
            f"decay={self.decay_description})"
        )


def resolve_rf_orientation(
    signed_frequency_hz: Union[float, int]
) -> RFTransitionOrientation:
    """Build the :class:`RFTransitionOrientation` from a signed ARC frequency.

    Parameters
    ----------
    signed_frequency_hz : float
        ``nu = (E_R2 - E_R1)/h`` in Hz, in LADDER order — exactly
        ``atom.getTransitionFrequency(rydberg1..., rydberg2...)``.

    Returns
    -------
    RFTransitionOrientation

    Raises
    ------
    ValueError
        If ``nu`` is non-finite or exactly zero (F05: no silent substitution
        and no inferred direction).
    """
    nu = float(signed_frequency_hz)
    direction = UPWARD_RF_DIRECTION if nu > 0.0 else DOWNWARD_RF_DIRECTION
    # The dataclass re-validates, so callers cannot bypass the invariants
    # through the resolver.
    return RFTransitionOrientation(
        signed_frequency_hz=nu, direction=direction
    )
