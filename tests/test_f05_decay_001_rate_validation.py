"""F05-DECAY-001 / F05-DECAY-001-R1 -- decay-rate validation contract.

Two guarantees are under test:

1. The legacy ``gamma_43 = max(gamma_43, 1e-6)`` floor is REMOVED: a valid ARC
   rate is returned exactly -- a genuine zero, and any positive value below the
   old floor.  No epsilon, clip, abs, alternate transition, alternate decay
   channel or solver workaround is ever substituted for a physical rate.
2. Every ARC-derived rate must already BE a real numeric scalar BEFORE any
   conversion is attempted (F05-DECAY-001-R1).  ``float()`` accepts ``True``,
   ``"1.0"`` and ``np.array([1.0])``, and silently discards the imaginary part
   of ``np.complex128(1+2j)`` under a ComplexWarning; those inputs are rejected
   with ``PhysicsCalculationError`` instead, and no ComplexWarning is emitted.

Controlled atom doubles supply boundary/invalid values; one live-ARC test
compares the current prescribed pair against values read from ARC itself.
No physical rate, solver behaviour or applied laser/RF input is modified.
"""

import copy
import warnings

import numpy as np
import pytest

from rydberg_aoa.arc.decay_rates import (
    _is_supported_real_scalar,
    calculate_decay_rates,
)
from rydberg_aoa.config import laser_rf_config
from rydberg_aoa.config import parameters as params
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsError

GROUND = {"n": 5, "l": 0, "j": 0.5}
EXCITED = {"n": 5, "l": 1, "j": 1.5}
R1 = {"n": 26, "l": 2, "j": 2.5}
R2 = {"n": 27, "l": 1, "j": 1.5}

# transition keys used by the atom doubles, identified by (n1, n2)
_OPTICAL_21 = (5, 5)      # excited -> ground
_OPTICAL_32 = (26, 5)     # rydberg1 -> excited
_RF_DOWN = (26, 27)       # R1 -> R2  (energy-downward for nu < 0)
_RF_UP = (27, 26)         # R2 -> R1  (energy-downward for nu > 0)

_GOOD = 10.0  # s^-1, comfortably above the removed 1e-6/us floor


class _ControlledAtom:
    """ARC double returning a per-(n1, n2) rate; records call order."""

    def __init__(self, nu, rates):
        self.nu = nu
        self.rates = dict(rates)
        self.calls = []

    def getTransitionFrequency(self, *args):
        return self.nu

    def getTransitionRate(self, n1, l1, j1, n2, l2, j2):
        self.calls.append((n1, n2))
        return self.rates.get((n1, n2), _GOOD)


# ---------------------------------------------------------------------------
# 1. The accepted real-scalar type contract
# ---------------------------------------------------------------------------

_ACCEPTED_VALUES = [
    10.0,
    0.0,
    5e-7,
    int(10),
    np.float64(10.0),
    np.float32(5.0),
    np.int64(10),
    np.int32(3),
]

_REJECTED_VALUES = [
    True,
    False,
    np.bool_(True),
    "1.0",
    b"1.0",
    1 + 2j,
    np.complex128(1 + 0j),
    np.complex128(1 + 2j),
    [1.0],
    [1.0, 2.0],
    np.array([1.0]),
    np.array(1.0),
    object(),
    None,
]


@pytest.mark.parametrize("value", _ACCEPTED_VALUES, ids=lambda v: repr(v))
def test_accepted_real_scalars(value):
    """Python and NumPy real numeric scalars are accepted by the type contract."""
    assert _is_supported_real_scalar(value) is True

# ---------------------------------------------------------------------------
# 2. Below-floor / zero / above-floor rates are preserved exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rf_rate_s, note",
    [
        (0.5, "below the removed 1e-6/us floor"),
        (0.0, "a genuine zero"),
        (5.0, "above the removed floor"),
        (47.944361878374565, "the current prescribed pair value"),
    ],
)
def test_rf_rate_is_preserved_without_flooring(rf_rate_s, note):
    """The RF-edge rate is returned exactly (s^-1 / 1e6); no floor is applied."""
    atom = _ControlledAtom(-5.0, {_RF_DOWN: rf_rate_s})
    _, _, g43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)

    expected = rf_rate_s / 1e6            # independent unit conversion
    assert g43 == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert atom.calls[-1] == _RF_DOWN     # energy-downward order for nu < 0
    if rf_rate_s < 1.0:
        assert g43 < 1e-6, note


@pytest.mark.parametrize("value", _ACCEPTED_VALUES, ids=lambda v: repr(v))
def test_accepted_scalars_are_converted_at_every_position(value):
    """A supported real scalar at any rate position converts to 1/us unchanged."""
    atom = _ControlledAtom(
        -5.0,
        {_OPTICAL_21: value, _OPTICAL_32: value, _RF_DOWN: value},
    )
    g21, g32, g43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)

    expected = float(value) / 1e6
    assert g21 == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert g32 == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert g43 == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_zero_is_exactly_zero_not_epsilon():
    """A valid zero is exactly 0.0 -- not 1e-6, not an epsilon, not a clip."""
    atom = _ControlledAtom(-5.0, {_RF_DOWN: 0.0})
    _, _, g43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert g43 == 0.0


# ---------------------------------------------------------------------------
# 3. Invalid outputs are rejected at each of the three rate positions
# ---------------------------------------------------------------------------

_BAD_VALUES = [
    None,
    "not-a-number",
    "1.0",
    b"1.0",
    [1.0],
    [1.0, 2.0],
    np.array([1.0]),
    np.array(1.0),
    True,
    False,
    np.bool_(True),
    1 + 2j,
    np.complex128(1 + 0j),
    np.complex128(1 + 2j),
    object(),
    float("nan"),
    float("inf"),
    -float("inf"),
    -5.0,
]

_POSITIONS = {
    "gamma_21": _OPTICAL_21,
    "gamma_32": _OPTICAL_32,
    "gamma_43": _RF_DOWN,
}


@pytest.mark.parametrize("bad", _BAD_VALUES, ids=lambda v: repr(v))
@pytest.mark.parametrize("position,key", list(_POSITIONS.items()))
def test_invalid_rate_at_each_position_fails_loudly(position, key, bad):
    """Any unsupported / non-finite / negative rate fails loudly, never floored."""
    atom = _ControlledAtom(-5.0, {key: bad})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert "no decay rate is invented" in str(excinfo.value)


@pytest.mark.parametrize("position,key", list(_POSITIONS.items()))
def test_huge_integer_overflows_before_conversion(position, key):
    """An unrepresentable magnitude overflows and fails loudly, with context."""
    atom = _ControlledAtom(-5.0, {key: 10 ** 400})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert "no decay rate is invented" in str(excinfo.value)
    assert "unconvertible" in str(excinfo.value)


def test_negative_rate_names_the_offending_value():
    atom = _ControlledAtom(-5.0, {_RF_DOWN: -3.0})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert "-3.0" in str(excinfo.value)
    assert "negative" in str(excinfo.value)


def test_nonfinite_rate_names_the_offending_value():
    atom = _ControlledAtom(-5.0, {_RF_DOWN: float("inf")})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert "non-finite" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad",
    [1 + 2j, np.complex128(1 + 0j), np.complex128(1 + 2j), np.array([1.0])],
    ids=lambda v: repr(v),
)
def test_rejection_happens_before_coercion_emits_no_warning(bad):
    """The type check runs first: no ComplexWarning/DeprecationWarning escapes.

    ``float(np.complex128(1+2j))`` would emit a ComplexWarning and silently
    discard the imaginary part; ``float(np.array([1.0]))`` would emit a
    DeprecationWarning.  Neither warning may be produced, because the value is
    rejected before any conversion is attempted.
    """
    atom = _ControlledAtom(-5.0, {_RF_DOWN: bad})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(PhysicsCalculationError):
            calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert not [w for w in caught if "Complex" in type(w.category).__name__]
    assert not [w for w in caught if "Deprecat" in type(w.category).__name__]


def test_rejected_value_is_named_in_the_message():
    """The offending value and its type are reported for diagnosis."""
    atom = _ControlledAtom(-5.0, {_RF_DOWN: True})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    message = str(excinfo.value)
    assert "True" in message
    assert "bool" in message



@pytest.mark.parametrize("value", _REJECTED_VALUES, ids=lambda v: repr(v))
def test_rejected_values_are_not_real_scalars(value):
    """Booleans, strings, complex, arrays and arbitrary objects are rejected."""
    assert _is_supported_real_scalar(value) is False


def test_bool_is_rejected_even_though_it_subclasses_int():
    """``bool`` is an ``int`` subclass, so it needs an explicit rejection."""
    assert isinstance(True, int)
    assert _is_supported_real_scalar(True) is False

# ---------------------------------------------------------------------------
# 4. ARC exceptions retain context and original cause
# ---------------------------------------------------------------------------

def test_arc_exception_is_wrapped_and_cause_preserved():
    class _BoomAtom(_ControlledAtom):
        def getTransitionRate(self, *args):
            raise ValueError("ARC internal failure")

    with pytest.raises(RuntimeError) as excinfo:
        calculate_decay_rates(_BoomAtom(-5.0, {}), GROUND, EXCITED, R1, R2)
    assert isinstance(excinfo.value, RuntimeError)
    assert "ARC internal failure" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_arc_failure_names_the_failing_operation_and_transition():
    """The wrapper identifies which ARC operation and which states failed."""

    class _BoomAtom(_ControlledAtom):
        def getTransitionRate(self, *args):
            raise ValueError("ARC internal failure")

    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(_BoomAtom(-5.0, {}), GROUND, EXCITED, R1, R2)
    message = str(excinfo.value)
    # the first ARC call is the excited -> ground rate
    assert "getTransitionRate" in message
    assert "excited -> ground" in message
    assert "n=5, l=1, j=1.5" in message
    assert "n=5, l=0, j=0.5" in message


def test_orientation_failure_is_attributed_to_getTransitionFrequency():
    """A frequency failure is attributed to the frequency call, not a rate."""

    class _FreqBoomAtom(_ControlledAtom):
        def getTransitionFrequency(self, *args):
            raise ValueError("frequency database corrupt")

    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(_FreqBoomAtom(-5.0, {}), GROUND, EXCITED, R1, R2)
    message = str(excinfo.value)
    assert "getTransitionFrequency" in message
    assert "orientation" in message
    assert "frequency database corrupt" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_degenerate_carrier_wraps_orientation_valueerror_with_cause():
    """nu = 0 -> resolve_rf_orientation ValueError, wrapped with cause intact."""
    with pytest.raises(RuntimeError) as excinfo:
        calculate_decay_rates(_ControlledAtom(0.0, {}), GROUND, EXCITED, R1, R2)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_physics_error_inside_is_not_remasked():
    """A PhysicsError raised inside propagates unchanged (not re-wrapped)."""

    class _PhysicsAtom(_ControlledAtom):
        def getTransitionRate(self, *args):
            raise PhysicsCalculationError("inner physics failure")

    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(_PhysicsAtom(-5.0, {}), GROUND, EXCITED, R1, R2)
    assert "inner physics failure" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. Energy-downward ARC call order for both orientations
# ---------------------------------------------------------------------------

def test_upward_and_downward_call_arc_in_energy_downward_order():
    down = _ControlledAtom(-5.0, {_RF_DOWN: 42.0})
    _, _, g43_down = calculate_decay_rates(down, GROUND, EXCITED, R1, R2)
    assert down.calls[-1] == _RF_DOWN
    assert g43_down == pytest.approx(42.0 / 1e6)

    up = _ControlledAtom(+5.0, {_RF_UP: 24.0})
    _, _, g43_up = calculate_decay_rates(up, GROUND, EXCITED, R1, R2)
    assert up.calls[-1] == _RF_UP
    assert g43_up == pytest.approx(24.0 / 1e6)


def test_emission_label_never_claims_an_energy_upward_channel():
    """The spontaneous-emission call is always energy-downward.

    For a ladder-upward pair the label must say the emission is
    energy-downward and merely note the ladder orientation; it must never
    read "energy-upward RF edge".
    """
    up = _ControlledAtom(+5.0, {_RF_UP: -3.0})
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_decay_rates(up, GROUND, EXCITED, R1, R2)
    message = str(excinfo.value)
    assert "energy-downward RF-edge spontaneous emission" in message
    assert "ladder-upward" in message
    assert "energy-upward" not in message


# ---------------------------------------------------------------------------
# 6. No applied laser/RF input or declared level is mutated
# ---------------------------------------------------------------------------

def _applied_config_snapshot():
    """Deep snapshot of every applied laser/RF input this module could touch."""
    return {
        "LASER_PARAMS": copy.deepcopy(laser_rf_config.LASER_PARAMS),
        "RF_PARAMS": copy.deepcopy(laser_rf_config.RF_PARAMS),
        "LO_RF_PARAMS": copy.deepcopy(laser_rf_config.LO_RF_PARAMS),
        "SIGNAL_RF_PARAMS": copy.deepcopy(laser_rf_config.SIGNAL_RF_PARAMS),
        "LAMBDA_P": params.SI_Constants.LAMBDA_P,
        "OMEGA_S_BASE": params.OMEGA_S_BASE,
        "OMEGA_LO_BASE": params.OMEGA_LO_BASE,
    }


def test_inputs_are_not_mutated():
    """Declared levels AND every applied laser/RF input survive the call."""
    ground, excited, r1, r2 = dict(GROUND), dict(EXCITED), dict(R1), dict(R2)
    before = _applied_config_snapshot()
    identities = (
        laser_rf_config.LASER_PARAMS,
        laser_rf_config.RF_PARAMS,
        laser_rf_config.LO_RF_PARAMS,
        laser_rf_config.SIGNAL_RF_PARAMS,
    )

    calculate_decay_rates(_ControlledAtom(-5.0, {}), ground, excited, r1, r2)

    after = _applied_config_snapshot()
    assert after == before
    assert identities == (
        laser_rf_config.LASER_PARAMS,
        laser_rf_config.RF_PARAMS,
        laser_rf_config.LO_RF_PARAMS,
        laser_rf_config.SIGNAL_RF_PARAMS,
    )
    assert ground == GROUND and excited == EXCITED
    assert r1 == R1 and r2 == R2


# ---------------------------------------------------------------------------
# 7. Live ARC: the current prescribed pair value is unchanged by this ticket
# ---------------------------------------------------------------------------

def test_live_current_pair_value_is_unchanged():
    """Live ARC: gamma_43 equals the energy-downward ARC rate divided by 1e6.

    ARC-derived values are the primary comparison; the historical constant is
    kept only as a secondary regression pin.  A missing dependency or a broken
    ARC database is a visible failure here, not a skip.
    """
    import arc

    atom = arc.Rubidium85()

    down_s = atom.getTransitionRate(26, 2, 2.5, 27, 1, 1.5)
    g21, g32, g43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)

    assert g43 == pytest.approx(down_s / 1e6, rel=1e-9)
    assert g43 == pytest.approx(47.944361878374565 / 1e6, rel=1e-6)
    assert g43 > 1e-6                 # genuinely above the removed floor
    assert g21 > 0.0 and g32 > 0.0
