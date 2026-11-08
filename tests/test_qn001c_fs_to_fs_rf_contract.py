"""QN-001C — explicit FS -> FS RF magnetic-channel contract.

Capability added (the QN-001C objective)
----------------------------------------
Before QN-001C the RF Rabi path delegated to the generic
``calculate_dipole_matrix_element`` helper, which SCANS mj1, mj2 and q and
keeps the LARGEST element: it silently SELECTED a magnetic channel instead
of computing the configured one, ignored ``rf_params.polarization``
entirely, and never required an explicit mJ on either Rydberg level.
QN-001C adds that explicit primitive:

    arc/dipole.py:
        calculate_fs_to_fs_dipole_matrix_element(atom, from_level,
                                                  to_level, q)
            -> SIGNED ARC dipole matrix element [e*a0] of EXACTLY the
               declared magnetic channel
        _validate_fs_to_fs_magnetic_channel(from_level, to_level, q)
            -> exact structural projection rule mj2 = mj1 + q (QN-001C.1),
               checked BEFORE ARC is called

    arc/rabi_frequency.py:
        calculate_rf_rabi_frequency(atom, level1, level2, rf_params)
            now routes through the primitive:
                Omega = |d| E_RF / hbar                    [rad/s]
            q comes from rf_params.polarization (dict or RFParams),
            required, never hard-coded, never defaulted, never scanned.

QN-001C computes a channel; it does NOT select one. Both levels must carry
an EXPLICIT mJ — the current Experimental optimizer output (26D5/2 and
27P3/2 without mJ) therefore fails fast here too, exactly as the coupling
path does at the HFS->FS boundary (QN-001B), and the production
``initialize_atomic_physics()`` seam stays BLOCKED until the level
declaration carries the Zeeman channel (ARCH-001).

Independent oracle
------------------
Signed values are cross-checked against real ARC 3.9.0 by a SymPy
angular-momentum formula that never calls ARC's element helpers:

    d = (-1)^(j1-mj1) * wigner3j(j1, 1, j2; -mj1, -q, mj2)
        * atom.getReducedMatrixElementJ(n1, l1, j1, n2, l2, j2)

This is ARC's own documented factorisation re-derived independently
(wigner3j from sympy.physics.wigner, radial part from ARC's radial database
only). Signs are compared exactly; a systematic sign disagreement would be
a STOP-the-task signal, never hidden with abs().

Scope note: the laser FS->FS branch of ``calculate_rabi_frequency``
(generic getRabiFrequency with explicit mJ) is unchanged (pinned by
QN-001B), and the level-selection dipole cache (``dipole_matrix``) is
intentionally NOT part of this contract.
"""
import numpy as np
import pytest
from scipy import constants as sc
from sympy.physics.wigner import wigner_3j

from rydberg_aoa.arc import rabi_frequency as rabi_frequency_module
from rydberg_aoa.arc.dipole import (
    calculate_fs_to_fs_dipole_matrix_element,
    EA0_TO_CM,
)
from rydberg_aoa.arc.rabi_frequency import calculate_rf_rabi_frequency
from rydberg_aoa.arc.types import AtomicLevel, RFParams
from rydberg_aoa.config.laser_rf_config import (
    LO_RF_PARAMS,
    SIGNAL_RF_PARAMS,
    calculate_rf_rabi_frequency as calculate_rf_rabi_frequency_config,
    get_signal_rf_rabi_frequency,
)
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)

HBAR = sc.hbar


# --- the configured RF operating point (phase0_baseline pair) ----------------
def _ryd1(mj=None):
    """26D5/2 lower level of the configured RF transition."""
    return AtomicLevel(n=26, l=2, j=2.5, mj=mj, label="26D5/2")


def _ryd2(mj=None):
    """27P3/2 upper level of the configured RF transition."""
    return AtomicLevel(n=27, l=1, j=1.5, mj=mj, label="27P3/2")


def _rf(polarization=0, e_field=0.5):
    return RFParams(frequency_hz=132.649e9, electric_field_vm=e_field,
                    polarization=polarization)


def _sympy_predicted_signed_dme_ea0(atom, j1, mj1, j2, mj2, q,
                                    n1, l1, n2, l2):
    """Independent SymPy oracle for ARC's FS dipole element [e*a0].

    Re-derives ARC's factorisation

        d = (-1)^(j1-mj1) wigner3j(j1,1,j2;-mj1,-q,mj2) * <n1 l1 j1||er||n2 l2 j2>

    without touching ARC's element helpers (radial part read directly).
    """
    reduced = atom.getReducedMatrixElementJ(n1, l1, j1, n2, l2, j2)
    return float((-1.0) ** (j1 - mj1)
                 * wigner_3j(j1, 1.0, j2, -mj1, -q, mj2)
                 * reduced)


# All 12 structurally admissible channels of 26D5/2 -> 27P3/2 (mj2 = mj1 + q,
# |mj1| <= 5/2, |mj2| <= 3/2); every one verified nonzero against ARC.
_ADMISSIBLE_RF_CHANNELS = [
    (-2.5, -1.5, 1), (-1.5, -0.5, 1), (-0.5, 0.5, 1), (0.5, 1.5, 1),
    (-1.5, -1.5, 0), (-0.5, -0.5, 0), (0.5, 0.5, 0), (1.5, 1.5, 0),
    (-0.5, -1.5, -1), (0.5, -0.5, -1), (1.5, 0.5, -1), (2.5, 1.5, -1),
]


@pytest.fixture(scope="module")
def atom():
    """One real ARC 3.9.0 Rubidium85 instance for the whole module."""
    from arc import Rubidium85
    return Rubidium85()


class _SpyFsAtom:
    """REAL ARC atom wrapper recording FS element calls; FORBIDS every
    other dipole path (the QN-001C one-call contract)."""

    def __init__(self, real):
        self._real = real
        self.fs_calls = []

    def getDipoleMatrixElement(self, *args, **kwargs):
        self.fs_calls.append((args, kwargs))
        return self._real.getDipoleMatrixElement(*args, **kwargs)

    def getDipoleMatrixElementHFS(self, *args, **kwargs):
        raise AssertionError("QN-001C: the FS->FS path called the HFS->HFS element")

    def getDipoleMatrixElementHFStoFS(self, *args, **kwargs):
        raise AssertionError("QN-001C: the FS->FS path called the HFS->FS element")

    def getRabiFrequency(self, *args, **kwargs):
        raise AssertionError("QN-001C: the FS->FS path called generic getRabiFrequency")

    def getReducedMatrixElementJ(self, *args, **kwargs):
        return self._real.getReducedMatrixElementJ(*args, **kwargs)


class _FailingFsAtom:
    """ARC stand-in whose FS element lookup blows up."""

    def getDipoleMatrixElement(self, *args, **kwargs):
        raise KeyError("element not in ARC database")


class _ZeroFsAtom:
    """ARC stand-in returning exactly zero for an ADMISSIBLE channel."""

    def getDipoleMatrixElement(self, *args, **kwargs):
        return 0.0


class _BadReturnAtom:
    """ARC stand-in returning a contract-violating value."""

    def __init__(self, value):
        self.value = value

    def getDipoleMatrixElement(self, *args, **kwargs):
        return self.value


# ---------------------------------------------------------------------------
# 1. primitive contract: fail-loud, one channel, one ARC call
# ---------------------------------------------------------------------------
def test_primitive_requires_an_arc_atom():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(None, _ryd1(0.5), _ryd2(0.5), 0)
    assert "F05" in str(excinfo.value)


def test_primitive_requires_explicit_mj_on_both_levels():
    """No mJ inference (QN-001C): the SOURCE without mJ fails ..."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            object(), _ryd1(None), _ryd2(0.5), 0)
    message = str(excinfo.value)
    assert "source" in message and "QN-001C" in message
    assert "no mJ is inferred" in message and "no mJ scan" in message
    # ... and so does the DESTINATION without mJ
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            object(), _ryd1(0.5), _ryd2(None), 0)
    message = str(excinfo.value)
    assert "destination" in message and "QN-001C" in message
    assert "no mJ is inferred" in message


def test_primitive_rejects_ambiguous_hfs_plus_fs_declarations():
    """A level carrying BOTH mJ and F/mF is refused on either side."""
    ambiguous = AtomicLevel(n=26, l=2, j=2.5, mj=0.5, f=3.0, mf=1.0,
                            label="26D5/2 (HFS+FS)")
    for src, dst in ((ambiguous, _ryd2(0.5)), (_ryd1(0.5), ambiguous)):
        with pytest.raises(PhysicsConfigurationError) as excinfo:
            calculate_fs_to_fs_dipole_matrix_element(object(), src, dst, 0)
        message = str(excinfo.value)
        assert "ambiguous" in message and "QN-001C" in message


def test_primitive_enforces_e1_selection_rules():
    # delta_l = 0 (26D -> 26D)
    same_l = AtomicLevel(n=27, l=2, j=2.5, mj=0.5, label="27D5/2")
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(object(), _ryd1(0.5), same_l, 0)
    assert "delta_l" in str(excinfo.value)
    # delta_j = 2 (5S1/2 -> 5P5/2 does not exist; use j=1/2 -> j=5/2 with dl=1)
    wide_j = AtomicLevel(n=7, l=1, j=2.5, mj=2.5, label="7P5/2")
    s_half = AtomicLevel(n=7, l=0, j=0.5, mj=0.5, label="7S1/2")
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(object(), s_half, wide_j, 1)
    assert "delta_j" in str(excinfo.value)
    # j = 0 -> j = 0 with delta_l = 1 (synthetic structural declaration;
    # the delta_l and delta_j rules pass, only the 0 -> 0 rule fires)
    j0a = AtomicLevel(n=6, l=0, j=0.0, mj=0.0, label="6S0")
    j0b = AtomicLevel(n=7, l=1, j=0.0, mj=0.0, label="7P0 (synthetic)")
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(object(), j0a, j0b, 0)
    assert "j = 0 -> j = 0" in str(excinfo.value)


@pytest.mark.parametrize("bad_q", [2, -2, 0.5, 1.5, None, "pi"])
def test_primitive_rejects_invalid_polarisation(bad_q):
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            object(), _ryd1(-1.5), _ryd2(-0.5), bad_q)
    assert "QN-001C" in str(excinfo.value)


def test_structural_channel_is_rejected_exactly_before_arc(atom):
    """mj2 - mj1 != q: rejected with ZERO ARC calls (QN-001C.1)."""
    spy = _SpyFsAtom(atom)
    # structurally impossible: mj2 = mj1 would need q = 0, but q = 1 declared
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(spy, _ryd1(1.5), _ryd2(1.5), 1)
    message = str(excinfo.value)
    assert "structurally forbidden" in message
    assert "mj2 = mj1 + q" in message
    assert "No alternate mJ or q" in message
    # the projection mismatch is caught BEFORE ARC evaluates anything
    assert spy.fs_calls == []


def test_admissible_channel_makes_exactly_one_arc_call(atom):
    """The QN-001C one-call contract with the exact positional arguments."""
    spy = _SpyFsAtom(atom)
    d = calculate_fs_to_fs_dipole_matrix_element(spy, _ryd1(-2.5), _ryd2(-1.5), 1)
    assert len(spy.fs_calls) == 1
    args, kwargs = spy.fs_calls[0]
    assert kwargs == {}                     # positional ARC contract, no drift
    assert args == (26, 2, 2.5, -2.5, 27, 1, 1.5, -1.5, 1)
    assert d > 0.0                          # signed identity, checked below


def test_arc_failure_is_not_swallowed():
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            _FailingFsAtom(), _ryd1(0.5), _ryd2(0.5), 0)
    assert isinstance(excinfo.value.__cause__, KeyError)
    assert "F05" in str(excinfo.value)


@pytest.mark.parametrize("bad_return", [None, np.nan, np.inf, -np.inf,
                                        "abc", {}])
def test_unusable_arc_returns_raise(bad_return):
    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            _BadReturnAtom(bad_return), _ryd1(0.5), _ryd2(0.5), 0)
    assert "F05" in str(excinfo.value)


def test_dark_channel_raises_instead_of_returning_a_zero():
    """An admissible-but-dark declared channel is a CONFIGURATION error:
    never a zero dipole, never a silent re-selection."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_fs_to_fs_dipole_matrix_element(
            _ZeroFsAtom(), _ryd1(0.5), _ryd2(0.5), 0)
    message = str(excinfo.value)
    assert "zero electric-dipole coupling" in message
    assert "No alternate mJ or q" in message


# ---------------------------------------------------------------------------
# 2. independent SymPy oracle: signed identity with real ARC 3.9.0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mj1,mj2,q", _ADMISSIBLE_RF_CHANNELS)
def test_signed_element_matches_independent_sympy_oracle(atom, mj1, mj2, q):
    """Exact sign agreement + relative agreement of the magnitude with a
    wigner3j-based re-derivation that never calls ARC's element helpers."""
    predicted = _sympy_predicted_signed_dme_ea0(
        atom, 2.5, mj1, 1.5, mj2, q, 26, 2, 27, 1)
    assert predicted != 0.0, "oracle must predict a nonzero admissible channel"
    d = calculate_fs_to_fs_dipole_matrix_element(
        atom, _ryd1(mj1), _ryd2(mj2), q)
    assert (d > 0) == (predicted > 0), (
        f"SIGN DISAGREEMENT for (mj1={mj1}, mj2={mj2}, q={q}): "
        f"ARC primitive {d!r} vs SymPy oracle {predicted!r}")
    assert d == pytest.approx(predicted, rel=1e-9)


def test_signed_sign_is_preserved_on_a_known_negative_channel(atom):
    """5S1/2 mj=-1/2 -> 5P3/2 mj=-3/2 (sigma-): the primitive returns the
    SIGNED element identical to the raw ARC value — no abs() inside."""
    lower = AtomicLevel(n=5, l=0, j=0.5, mj=-0.5, label="5S1/2 (FS)")
    upper = AtomicLevel(n=5, l=1, j=1.5, mj=-1.5, label="5P3/2 (FS)")
    raw = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 5, 1, 1.5, -1.5, -1)
    d = calculate_fs_to_fs_dipole_matrix_element(atom, lower, upper, -1)
    assert d == raw                      # bit-identical to the raw ARC call
    predicted = _sympy_predicted_signed_dme_ea0(
        atom, 0.5, -0.5, 1.5, -1.5, -1, 5, 0, 5, 1)
    assert (d > 0) == (predicted > 0)    # sign convention cross-validated
    assert d == pytest.approx(predicted, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. RF Rabi helper: ONE declared channel, configured q, C-001 units
# ---------------------------------------------------------------------------
def test_rf_rabi_matches_independent_oracle(atom):
    """Omega = |SymPy-oracle d| E / hbar — the production helper is never
    used to generate its own expected value (no 2pi, one e*a0 conversion)."""
    e_field = 0.5
    predicted_ea0 = _sympy_predicted_signed_dme_ea0(
        atom, 2.5, -2.5, 1.5, -1.5, 1, 26, 2, 27, 1)
    omega = calculate_rf_rabi_frequency(
        atom, _ryd1(-2.5), _ryd2(-1.5), _rf(polarization=1, e_field=e_field))
    expected = abs(predicted_ea0) * EA0_TO_CM * e_field / HBAR
    assert omega == pytest.approx(expected, rel=1e-9)


def test_rf_rabi_accepts_dict_and_object_levels_equally(atom):
    """Both supported input shapes run the SAME declared channel."""
    r1_obj, r2_obj = _ryd1(-2.5), _ryd2(-1.5)
    r1_dict = {"n": 26, "l": 2, "j": 2.5, "mj": -2.5, "label": "26D5/2"}
    r2_dict = {"n": 27, "l": 1, "j": 1.5, "mj": -1.5, "label": "27P3/2"}
    omega_obj = calculate_rf_rabi_frequency(atom, r1_obj, r2_obj, _rf(1))
    omega_dict = calculate_rf_rabi_frequency(atom, r1_dict, r2_dict, _rf(1))
    assert omega_dict == pytest.approx(omega_obj, rel=1e-15)


def test_rf_rabi_passes_the_configured_polarisation_to_arc(atom):
    """q comes from rf_params.polarization and reaches ARC verbatim —
    there is no hard-coded q and no q scan. Channel 2.5 -> 1.5 REQUIRES
    q = -1 (mj2 - mj1 = -1), so this also proves q and mJ are used
    together, never independently."""
    spy = _SpyFsAtom(atom)
    omega = calculate_rf_rabi_frequency(spy, _ryd1(2.5), _ryd2(1.5),
                                        _rf(polarization=-1, e_field=0.5))
    assert len(spy.fs_calls) == 1
    args, kwargs = spy.fs_calls[0]
    assert kwargs == {}
    assert args[-1] == -1               # the configured q, positional
    assert args == (26, 2, 2.5, 2.5, 27, 1, 1.5, 1.5, -1)
    assert omega > 0.0


def test_rf_rabi_requires_explicit_mj_on_both_levels(atom):
    """The documented production blocker: the current optimizer output
    declares no Rydberg mJ, so the RF Rabi fails fast instead of silently
    picking the strongest channel."""
    r1_no_mj = {"n": 26, "l": 2, "j": 2.5, "mj": None, "label": "26D2.5"}
    r2_no_mj = {"n": 27, "l": 1, "j": 1.5, "mj": None, "label": "27P1.5"}
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_rf_rabi_frequency(atom, r1_no_mj, r2_no_mj, _rf(0))
    message = str(excinfo.value)
    assert "QN-001C" in message
    assert "explicit fine-structure magnetic sublevel" in message
    assert "no mJ is inferred" in message
    # source-only and destination-only omissions fail too
    with pytest.raises(PhysicsConfigurationError):
        calculate_rf_rabi_frequency(atom, r1_no_mj, _ryd2(0.5), _rf(0))
    with pytest.raises(PhysicsConfigurationError):
        calculate_rf_rabi_frequency(atom, _ryd1(0.5), r2_no_mj, _rf(0))


def test_rf_rabi_requires_a_configured_polarisation(atom):
    """A missing/invalid q is a configuration error, never a default."""
    no_q = {"frequency_hz": 132.649e9, "electric_field_vm": 0.5}
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_rf_rabi_frequency(atom, _ryd1(0.5), _ryd2(0.5), no_q)
    assert "QN-001C" in str(excinfo.value)
    with pytest.raises(PhysicsConfigurationError):
        calculate_rf_rabi_frequency(
            atom, _ryd1(0.5), _ryd2(0.5),
            {"frequency_hz": 132.649e9, "electric_field_vm": 0.5,
             "polarization": 3})
    with pytest.raises(PhysicsConfigurationError):
        calculate_rf_rabi_frequency(
            atom, _ryd1(0.5), _ryd2(0.5),
            {"frequency_hz": 132.649e9, "electric_field_vm": 0.5})


def test_rf_rabi_requires_a_field(atom):
    with pytest.raises(PhysicsConfigurationError):
        calculate_rf_rabi_frequency(
            atom, _ryd1(0.5), _ryd2(0.5),
            {"frequency_hz": 132.649e9, "electric_field_vm": None,
             "polarization": 0})


def test_rf_rabi_never_uses_the_max_over_mj_helper(atom, monkeypatch):
    """The legacy scan helper must be unreachable from the RF path."""
    def _boom(*args, **kwargs):
        raise AssertionError(
            "QN-001C: the RF path used the generic max-over-mJ helper")
    monkeypatch.setattr(rabi_frequency_module,
                        "calculate_dipole_matrix_element", _boom)
    omega = calculate_rf_rabi_frequency(
        atom, _ryd1(1.5), _ryd2(1.5), _rf(polarization=0, e_field=0.5))
    assert omega > 0.0


def test_rf_rabi_scales_linearly_with_the_field(atom):
    """Omega is strictly linear in E_RF: the field enters ONLY through
    Omega = |d| E / hbar (no intensity convention, no hidden saturation)."""
    e1, e2 = 0.25, 2.5
    omega1 = calculate_rf_rabi_frequency(
        atom, _ryd1(-1.5), _ryd2(-1.5), _rf(0, e1))
    omega2 = calculate_rf_rabi_frequency(
        atom, _ryd1(-1.5), _ryd2(-1.5), _rf(0, e2))
    assert omega2 / omega1 == pytest.approx(e2 / e1, rel=1e-12)


def test_config_level_rf_wrapper_runs_the_same_declared_channel(atom):
    """laser_rf_config.calculate_rf_rabi_frequency (production entry point)
    with the LO dictionary drives exactly the declared mJ channel (the LO
    polarisation q = 0 requires mj1 = mj2)."""
    spy = _SpyFsAtom(atom)
    r1_dict = {"n": 26, "l": 2, "j": 2.5, "mj": -1.5, "label": "26D5/2"}
    r2_dict = {"n": 27, "l": 1, "j": 1.5, "mj": -1.5, "label": "27P3/2"}
    omega = calculate_rf_rabi_frequency_config(
        spy, r1_dict, r2_dict, LO_RF_PARAMS)
    assert len(spy.fs_calls) == 1
    args, kwargs = spy.fs_calls[0]
    assert kwargs == {}
    assert args == (26, 2, 2.5, -1.5, 27, 1, 1.5, -1.5,
                    LO_RF_PARAMS["polarization"])
    predicted_ea0 = _sympy_predicted_signed_dme_ea0(
        atom, 2.5, -1.5, 1.5, -1.5, LO_RF_PARAMS["polarization"], 26, 2, 27, 1)
    expected = (abs(predicted_ea0) * EA0_TO_CM
                * LO_RF_PARAMS["electric_field_vm"] / HBAR)
    assert omega == pytest.approx(expected, rel=1e-9)


def test_signal_rf_dict_entry_point_is_channel_explicit(atom):
    """get_signal_rf_rabi_frequency (production SIGNAL entry point) runs
    the SAME declared channel contract (SIGNAL polarisation q = 0)."""
    spy = _SpyFsAtom(atom)
    r1_dict = {"n": 26, "l": 2, "j": 2.5, "mj": -1.5, "label": "26D5/2"}
    r2_dict = {"n": 27, "l": 1, "j": 1.5, "mj": -1.5, "label": "27P3/2"}
    omega = get_signal_rf_rabi_frequency(
        spy, r1_dict, r2_dict, SIGNAL_RF_PARAMS)
    assert len(spy.fs_calls) == 1
    assert spy.fs_calls[0][0][-1] == SIGNAL_RF_PARAMS["polarization"]
    predicted_ea0 = _sympy_predicted_signed_dme_ea0(
        atom, 2.5, -1.5, 1.5, -1.5, SIGNAL_RF_PARAMS["polarization"],
        26, 2, 27, 1)
    expected = (abs(predicted_ea0) * EA0_TO_CM
                * SIGNAL_RF_PARAMS["electric_field_vm"] / HBAR)
    assert omega == pytest.approx(expected, rel=1e-9)
