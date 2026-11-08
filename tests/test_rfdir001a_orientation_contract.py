"""RF-DIR-001A — the energy-orientation contract of the |3> <-> |4> RF edge.

Scope
-----
This suite covers ONLY the standalone orientation primitive
(:mod:`rydberg_aoa.arc.rf_orientation`):

* ``RFTransitionOrientation`` invariants (frozen, re-validated on construction);
* ``resolve_rf_orientation``: signed frequency -> direction / carrier;
* ``carrier_hz == abs(signed_frequency_hz)``;
* upward / downward direction and the upper / lower ladder roles;
* zero / non-finite rejection (F05: direction is never inferred or substituted);
* pickle safety for the multiprocessing worker path;
* the live ARC production pair (Rb85 26D5/2 -> 27P3/2) resolving DOWNWARD.

Deferred to the follow-up tickets (their production integration does not exist
yet; parked outside pytest collection in
``tests/manual_diagnostics/test_rfdir001bcd_energy_orientation_deferred.py``):

* RF-DIR-001B — energy-ordered decay / thermal BBR contract;
* RF-DIR-001C — rotating-frame Hamiltonian direction;
* RF-DIR-001D — signed IF / heterodyne / AoA phase convention.
"""

import pickle

import pytest

from rydberg_aoa.arc.rf_orientation import (
    DOWNWARD_RF_DIRECTION,
    RFTransitionOrientation,
    UPWARD_RF_DIRECTION,
    resolve_rf_orientation,
)

# ARC's SIGNED frequency for the production pair (Rb85 26D5/2 -> 27P3/2),
# taken in LADDER order.  Negative => E(rydberg2) < E(rydberg1): DOWNWARD.
NU_R1_TO_R2_HZ = -1.3264910772876007e11
NU_R2_TO_R1_HZ = +1.3264910772876007e11

DOWNWARD = resolve_rf_orientation(NU_R1_TO_R2_HZ)
UPWARD = resolve_rf_orientation(NU_R2_TO_R1_HZ)


def test_resolver_direction_carrier_and_roles():
    """Negative nu -> downward (|4><3|); positive nu -> upward (|3><4|)."""
    assert DOWNWARD.direction == DOWNWARD_RF_DIRECTION == -1
    assert DOWNWARD.is_downward and not DOWNWARD.is_upward
    assert DOWNWARD.carrier_hz == pytest.approx(abs(NU_R1_TO_R2_HZ))
    assert DOWNWARD.carrier_hz > 0.0
    assert DOWNWARD.decay_operator_name == "s43"
    # The energy-downward channel runs from the ladder |3> (R1) to the |4> (R2).
    assert (DOWNWARD.upper_ladder_index, DOWNWARD.lower_ladder_index) == (3, 4)
    assert DOWNWARD.upper_role == "R1"
    assert DOWNWARD.lower_role == "R2"
    assert DOWNWARD.decay_description == "R1(3) -> R2(4)"

    assert UPWARD.direction == UPWARD_RF_DIRECTION == +1
    assert UPWARD.is_upward and not UPWARD.is_downward
    assert UPWARD.carrier_hz == pytest.approx(abs(NU_R2_TO_R1_HZ))
    assert UPWARD.decay_operator_name == "s34"
    assert (UPWARD.upper_ladder_index, UPWARD.lower_ladder_index) == (4, 3)
    assert UPWARD.upper_role == "R2"
    assert UPWARD.lower_role == "R1"
    assert UPWARD.decay_description == "R2(4) -> R1(3)"


def test_carrier_magnitude_is_sign_independent():
    """The field carrier is |nu|; only its sign carries the direction."""
    assert DOWNWARD.carrier_hz == pytest.approx(UPWARD.carrier_hz)
    assert DOWNWARD.signed_frequency_hz == -UPWARD.signed_frequency_hz


@pytest.mark.parametrize("nu", [0.0, float("nan"), float("inf"), -float("inf")])
def test_resolver_rejects_degenerate_or_nonfinite(nu):
    """F05: a degenerate/non-physical carrier never yields an inferred direction."""
    with pytest.raises(ValueError):
        resolve_rf_orientation(nu)


def test_dataclass_rejects_degenerate_or_nonfinite_directly():
    """The dataclass re-validates, so no caller can bypass the resolver."""
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=float("nan"), direction=-1)
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=float("inf"), direction=+1)
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=0.0, direction=-1)


def test_dataclass_rejects_direction_sign_contradiction():
    """Direction and signed frequency must agree; direction is never guessed."""
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=NU_R1_TO_R2_HZ, direction=+1)
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=NU_R2_TO_R1_HZ, direction=-1)
    with pytest.raises(ValueError):
        RFTransitionOrientation(signed_frequency_hz=NU_R1_TO_R2_HZ, direction=0)


def test_resolver_admits_any_signed_real_and_is_picklable():
    """A resolved orientation survives a spawn (the time-domain worker path)."""
    tiny = resolve_rf_orientation(-1.0)
    assert tiny.is_downward and tiny.carrier_hz == pytest.approx(1.0)
    assert pickle.loads(pickle.dumps(DOWNWARD)) == DOWNWARD


def test_production_pair_is_genuinely_downward():
    """ARC regression guard for RF-DIR-001A's central physical claim.

    The production pair (Rb85 26D5/2 -> 27P3/2) must resolve to a DOWNWARD
    edge.  Reads ARC directly (no project configuration, no atomic
    initialization) and skips cleanly when the ARC database is unavailable.
    """
    arc = pytest.importorskip("arc")
    try:
        atom = arc.Rubidium85()
    except Exception as exc:  # environment-dependent (ARC data unavailable)
        pytest.skip(f"ARC atomic database unavailable: {exc}")

    nu = atom.getTransitionFrequency(26, 2, 2.5, 27, 1, 1.5)
    orientation = resolve_rf_orientation(nu)

    assert orientation.signed_frequency_hz < 0.0
    assert orientation.carrier_hz > 0.0
    assert orientation.direction == -1
    assert orientation.is_downward
    assert orientation.upper_role == "R1"
    assert orientation.lower_role == "R2"
    assert (orientation.upper_ladder_index,
            orientation.lower_ladder_index) == (3, 4)
    assert orientation.decay_operator_name == "s43"
    assert orientation.carrier_hz == pytest.approx(132.64910772876e9,
                                                   rel=1e-6)
