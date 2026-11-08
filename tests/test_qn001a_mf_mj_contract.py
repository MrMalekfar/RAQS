"""QN-001A — mF/mJ quantum-number contract and the primary probe HFS Rabi path.

Removed defect (QN-001A root cause)
-----------------------------------
``AtomicLevel.__post_init__`` used to manufacture one quantum number from the
other::

    mj := mf   (when mf was given)    else    mj := 0.0

which conflated mJ and mF and injected the INVALID sublevel ``mj = 0`` for every
half-integer ``j`` (the configured 5S1/2 ground level has j = 1/2). ARC's
``getRabiFrequency`` then returned exactly ``0.0`` for the driven probe
transition, and pre-F05 that zero was silently replaced by a dipole-based
substitute.

QN-001A contract (this file pins it)
------------------------------------
* ``mj`` and ``mf`` are independent OPTIONAL quantum numbers — never inferred
  from each other, never defaulted;
* validation only: ``|m| <= j`` and ``j - m`` must be an integer;
* the probe transition 5S1/2 F=3, mF=0 -> 5P3/2 F=4, mF=0 is computed by the
  designated PRIMARY ARC path ``getDipoleMatrixElementHFS`` +
  ``Omega = |d| E_peak / hbar`` with the CONFIGURED polarisation q;
* every failure raises — no generic ``getRabiFrequency`` retry, no other mJ/mF/q
  trial, no cached, hard-coded or zero value (F05 must stay green);
* an HFS -> FS transition without an explicit Rydberg mJ contract raises
  ``PhysicsConfigurationError`` (the EXPECTED next blocker, task section 6).

Requirement map: items A..L of task section 8.
"""
import numpy as np
import pytest
from scipy import constants as sc

from rydberg_aoa.arc import rabi_frequency as rabi_frequency_module
from rydberg_aoa.arc.rabi_frequency import (
    calculate_hfs_rabi_frequency,
    calculate_rabi_frequency,
)
from rydberg_aoa.arc.types import AtomicLevel, LaserParams
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)

HBAR = sc.hbar
EA0_TO_CM = 8.478e-30  # arc/dipole.EA0_TO_CM — project conversion, e·a0 -> C·m

# The configured probe transition of the production model (section 4)
GROUND = AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.0, label="5S1/2 F=3")
EXCITED = AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=0.0, label="5P3/2 F=4")
RYD1 = AtomicLevel(n=26, l=2, j=2.5, label="26D5/2")   # mj must stay None

# Frozen OP-001 probe operating point (config/laser_rf_config.LASER_PARAMS):
# P = 0.10 uW and the 1/e² radius derived from the Simons 94 um FWHM diameter.
PROBE_POWER_MW = 0.0001
PROBE_WAIST_MM = 0.07983624922707379


def probe_laser(polarization=0):
    """LaserParams for the configured probe transition."""
    return LaserParams(wavelength_nm=780.241476, power_mw=PROBE_POWER_MW,
                       beam_waist_mm=PROBE_WAIST_MM, detuning_mhz=0.0,
                       polarization=polarization)


def coupling_laser():
    """LaserParams for the configured coupling laser (section 6 blocker)."""
    return LaserParams(wavelength_nm=483.4, power_mw=30.0,
                       beam_waist_mm=0.12230197665769335,
                       detuning_mhz=0.0, polarization=0)


def e_peak(power_w, waist_m):
    """Independent Gaussian-beam peak field: E = sqrt(2 I0/(eps0 c))."""
    i0 = 2.0 * power_w / (np.pi * waist_m ** 2)
    return np.sqrt(2.0 * i0 / (sc.epsilon_0 * sc.c))


class _SpyHFSAtom:
    """ARC stand-in recording HFS calls and FORBIDDING the generic Rabi call."""

    def __init__(self, d_ea0=-2.259585008991063):
        self.d_ea0 = d_ea0
        self.hfs_calls = []

    def getDipoleMatrixElementHFS(self, n1, l1, j1, f1, mf1,
                                  n2, l2, j2, f2, mf2, q, s=0.5):
        self.hfs_calls.append({
            "n1": n1, "l1": l1, "j1": j1, "f1": f1, "mf1": mf1,
            "n2": n2, "l2": l2, "j2": j2, "f2": f2, "mf2": mf2, "q": q,
        })
        return self.d_ea0

    def getRabiFrequency(self, *args, **kwargs):
        raise AssertionError(
            "QN-001A: the probe HFS path called the generic "
            "getRabiFrequency — the configured mF/mJ contract was bypassed")

    @property
    def getRabiFrequencyHFS(self):
        raise AssertionError(
            "QN-001A: the legacy getRabiFrequencyHFS branch must be gone "
            "(no such method exists on the installed ARC 3.9.0)")


# ---------------------------------------------------------------------------
# A. mF does not imply mJ
# ---------------------------------------------------------------------------
def test_a_hyperfine_level_leaves_mj_unset():
    """F=3, mF=0 with j=1/2 must leave mj=None (QN-001A item A)."""
    assert GROUND.f == 3.0 and GROUND.mf == 0.0 and GROUND.mj is None
    assert EXCITED.f == 4.0 and EXCITED.mf == 0.0 and EXCITED.mj is None
    # no manufactured projection for the fine-structure state either
    assert RYD1.mj is None and RYD1.mf is None and RYD1.f is None


# ---------------------------------------------------------------------------
# B + C. explicit valid mJ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mj", [0.5, -0.5])
def test_b_c_explicit_half_integer_mj_is_valid(mj):
    level = AtomicLevel(n=5, l=0, j=0.5, mj=mj)
    assert level.mj == mj
    assert level.mf is None          # mJ never copied into mF


# ---------------------------------------------------------------------------
# D. mj = 0 is invalid for j = 1/2
# ---------------------------------------------------------------------------
def test_d_mj_zero_is_rejected_for_half_integer_j():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        AtomicLevel(n=5, l=0, j=0.5, mj=0.0)
    assert "mj=0.0" in str(excinfo.value)
    assert "j=0.5" in str(excinfo.value)


# ---------------------------------------------------------------------------
# E + F. neither quantum number is ever copied into the other
# ---------------------------------------------------------------------------
def test_e_mf_is_never_copied_into_mj():
    level = AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=2.0)   # valid mF = +2
    assert level.mf == 2.0
    assert level.mj is None


def test_f_mj_is_never_copied_into_mf():
    level = AtomicLevel(n=5, l=1, j=1.5, mj=1.5)
    assert level.mj == 1.5
    assert level.mf is None


def test_f2_invalid_projections_and_orphan_mf_are_rejected():
    with pytest.raises(PhysicsConfigurationError):      # mF given without F
        AtomicLevel(n=5, l=0, j=0.5, mf=0.0)
    with pytest.raises(PhysicsConfigurationError):      # |mF| > F
        AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=4.0)
    with pytest.raises(PhysicsConfigurationError):      # integer F -> integer mF
        AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.5)
    with pytest.raises(PhysicsConfigurationError):      # |mj| > j
        AtomicLevel(n=5, l=1, j=1.5, mj=2.5)


# ---------------------------------------------------------------------------
# G + H. probe HFS call contract: exact states, configured q, no generic call
# ---------------------------------------------------------------------------
def test_g_probe_calls_arc_hfs_dme_with_exact_states_and_configured_q():
    atom = _SpyHFSAtom()
    omega = calculate_rabi_frequency(atom, EXCITED, probe_laser(polarization=0),
                                     from_level=GROUND, ground_state=GROUND)
    assert len(atom.hfs_calls) == 1          # exactly ONE designated call
    call = atom.hfs_calls[0]
    assert (call["n1"], call["l1"], call["j1"]) == (5, 0, 0.5)
    assert (call["f1"], call["mf1"]) == (3.0, 0.0)      # F=3, mF=0
    assert (call["n2"], call["l2"], call["j2"]) == (5, 1, 1.5)
    assert (call["f2"], call["mf2"]) == (4.0, 0.0)      # F=4, mF=0
    assert call["q"] == 0                   # configured pi transition
    assert omega > 0.0
    # item H: _SpyHFSAtom.getRabiFrequency raises AssertionError if reached


@pytest.mark.parametrize("q_configured", [0, 1, -1])
def test_g2_q_comes_from_the_configured_laser(q_configured):
    """q is taken from laser_params.polarization — never hard-coded (section 5)."""
    atom = _SpyHFSAtom()
    calculate_rabi_frequency(atom, EXCITED,
                             probe_laser(polarization=q_configured),
                             from_level=GROUND, ground_state=GROUND)
    assert [c["q"] for c in atom.hfs_calls] == [q_configured]


def test_h_direct_hfs_helper_never_calls_getRabi_frequency():
    """Item H: reaching the generic ARC call fails the test (spy raises)."""
    omega = calculate_hfs_rabi_frequency(_SpyHFSAtom(), GROUND, EXCITED,
                                         probe_laser(0))
    assert omega > 0.0


# ---------------------------------------------------------------------------
# I. Omega = |d_HFS| * E_peak / hbar   (independent oracle)
# ---------------------------------------------------------------------------
def test_i_omega_matches_independent_mu_E_over_hbar_oracle():
    """Deterministic stub: the full chain is checked to machine precision."""
    atom = _SpyHFSAtom(d_ea0=-2.5)
    omega = calculate_hfs_rabi_frequency(atom, GROUND, EXCITED, probe_laser(0))
    expected = (2.5 * EA0_TO_CM
                * e_peak(PROBE_POWER_MW * 1e-3, PROBE_WAIST_MM * 1e-3) / HBAR)
    assert omega == pytest.approx(expected, rel=1e-15)


def test_i2_real_arc_probe_value_is_the_hfs_element():
    """Numerical validation of the probe with the INSTALLED ARC (section 9)."""
    from arc import Rubidium85
    atom = Rubidium85()
    omega = calculate_rabi_frequency(atom, EXCITED, probe_laser(0),
                                     from_level=GROUND, ground_state=GROUND)
    d_ea0 = atom.getDipoleMatrixElementHFS(5, 0, 0.5, 3.0, 0,
                                           5, 1, 1.5, 4.0, 0, 0)
    expected = (abs(d_ea0) * EA0_TO_CM
                * e_peak(PROBE_POWER_MW * 1e-3, PROBE_WAIST_MM * 1e-3) / HBAR)
    assert omega == pytest.approx(expected, rel=1e-12)
    # the generic fine-structure element is a DIFFERENT transition (~8% apart)
    d_fs = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 5, 1, 1.5, 0.5, 0)
    omega_fs = (abs(d_fs) * EA0_TO_CM
                * e_peak(PROBE_POWER_MW * 1e-3, PROBE_WAIST_MM * 1e-3) / HBAR)
    assert abs(omega - omega_fs) / omega > 0.05


# ---------------------------------------------------------------------------
# J + K. ARC HFS failure / unusable element -> PhysicsCalculationError
# ---------------------------------------------------------------------------
def test_j_arc_hfs_failure_raises_with_the_original_cause():
    class _FailingHFSElementAtom:
        def getDipoleMatrixElementHFS(self, *args, **kwargs):
            raise ValueError("element not in ARC database")

    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_hfs_rabi_frequency(_FailingHFSElementAtom(), GROUND, EXCITED,
                                     probe_laser(0))
    message = str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "hyperfine dipole matrix element" in message   # designated path
    assert "no generic getRabiFrequency" in message       # no retry elsewhere


@pytest.mark.parametrize("bad_element", [0.0, np.nan, np.inf, -np.inf,
                                         None, "n/a"])
def test_k_unusable_hfs_element_raises(bad_element):
    with pytest.raises(PhysicsCalculationError):
        calculate_hfs_rabi_frequency(_SpyHFSAtom(d_ea0=bad_element),
                                     GROUND, EXCITED, probe_laser(0))


# ---------------------------------------------------------------------------
# L. no fallback of any kind is invoked on the primary path
# ---------------------------------------------------------------------------
def test_l_no_dipole_module_fallback_is_invoked(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "QN-001A: the probe path used the removed dipole-based fallback")

    monkeypatch.setattr(rabi_frequency_module,
                        "calculate_dipole_matrix_element", _boom)
    omega = calculate_rabi_frequency(_SpyHFSAtom(), EXCITED, probe_laser(0),
                                     from_level=GROUND, ground_state=GROUND)
    assert omega > 0.0


# ---------------------------------------------------------------------------
# Section 6 — the EXPECTED next blocker: coupling HFS -> FS without an mJ
# ---------------------------------------------------------------------------
def test_coupling_hfs_to_fs_without_explicit_mj_fails_fast():
    """5P3/2 F=4, mF=0 -> 26D5/2 must NOT be given an implicit Rydberg mJ."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_rabi_frequency(_SpyHFSAtom(), RYD1, coupling_laser(),
                                 from_level=EXCITED, ground_state=GROUND)
    message = str(excinfo.value)
    assert "explicit Rydberg mJ" in message
    assert "Zeeman-channel contract" in message
    assert "scan mJ" in message
    assert "F05" in message       # required by the F05 BLOCKED-skip contract


def test_fs_path_refuses_a_contradicting_declared_mj2():
    lower = AtomicLevel(n=5, l=0, j=0.5, mj=0.5)
    upper = AtomicLevel(n=5, l=1, j=1.5, mj=1.5)   # mj1 + q = 0.5 != 1.5
    with pytest.raises(PhysicsConfigurationError):
        calculate_rabi_frequency(_SpyHFSAtom(), upper, probe_laser(0),
                                 from_level=lower, ground_state=lower)


@pytest.mark.parametrize("bad_q", [2, -2, 0.5])
def test_invalid_configured_polarisation_is_rejected(bad_q):
    with pytest.raises(PhysicsConfigurationError):
        calculate_rabi_frequency(_SpyHFSAtom(), EXCITED,
                                 probe_laser(polarization=bad_q),
                                 from_level=GROUND, ground_state=GROUND)
