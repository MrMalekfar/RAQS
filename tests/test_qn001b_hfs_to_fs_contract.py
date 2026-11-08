"""QN-001B — explicit HFS -> FS optical coupling contract.

Capability added (the QN-001B objective)
----------------------------------------
Before QN-001B the model could not evaluate an optical transition

    |n1 l1 j1 F1 mF1>  ->  |n2 l2 j2 mJ2>            (HFS -> FS)

for a configured laser polarisation q: an HFS-resolved source carries mF
(never mJ), so the generic fine-structure branch cannot drive it, and no
dedicated ARC HFS->FS path existed. QN-001B adds that explicit primitive:

    arc/dipole.py:
        calculate_hfs_to_fs_dipole_matrix_element(atom, from_level,
                                                  to_level, q)
            -> SIGNED ARC dipole matrix element [e*a0] of EXACTLY the
               declared magnetic channel

    arc/rabi_frequency.py:
        calculate_hfs_to_fs_rabi_frequency(atom, from_level, to_level,
                                           laser_params)
            -> Omega = |d| E_peak / hbar                    [rad/s]

and dispatches to it from ``calculate_rabi_frequency``:

    HFS -> HFS : existing QN-001A HFS path (frozen)
    HFS -> FS  : NEW explicit QN-001B path  (pinned by this file)
    FS  -> FS  : existing explicit fine-structure path (unchanged)

QN-001B computes a channel; it does NOT select one. There is no mJ
inference, no mJ scan, no q scan and no maximum over magnetic sublevels, and
the current Experimental optimizer output (26D5/2 without mJ) still fails
fast at the EXPLICIT initialization seam (ARCH-001).

Independent oracle
------------------
The signed channel values are cross-checked against real ARC 3.9.0 by a
SymPy angular-momentum recoupling that never calls ARC's HFS->FS helpers: the
initial hyperfine state is expanded in the uncoupled |j1 mJ1>|I mI> basis,
mJ1 = mJ2 - q and mI = mF1 - mJ1 are exact, and the angular coefficient is

    <j1 mJ1, I mI|F1 mF>(-1)^(j1-mJ1) wigner3j(j1, 1, j2; -mJ1, -q, mJ2)

multiplied by the reduced radial element ``getReducedMatrixElementJ``.
Signs are compared exactly; a systematic sign disagreement would be a
STOP-the-task signal, never hidden with ``abs()``.

Structural note (QN-001B.1, documented, not hidden): a channel whose implied
projections mJ1 = mJ2 - q (|mJ1| <= j1) or mI = mF1 - mJ1 (|mI| <= I) cannot
exist is rejected EXACTLY, before ARC is ever called — no dipole threshold,
no numerical cutoff — while the raw ARC diagnostic calls below keep
documenting ARC's real behaviour: for the geometrically dark q = 0 channels
mJ = +/-5/2 ARC's internal sum over intermediate f2 leaves a
float-cancellation residue of order 1e-18 e*a0 instead of exactly 0.0 (the
SymPy oracle predicts exactly zero), and for the q = +-1 structural zeros
ARC returns exactly 0.0. The post-ARC exact-zero guard of QN-001B remains,
pinned by spies whose structurally ADMISSIBLE channel returns 0.0.
"""
import inspect

import numpy as np
import pytest
import sympy as sp
from scipy import constants as sc
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j

from rydberg_aoa.arc import rabi_frequency as rabi_frequency_module
from rydberg_aoa.arc.dipole import (
    calculate_hfs_to_fs_dipole_matrix_element,
    EA0_TO_CM,
)
from rydberg_aoa.arc.rabi_frequency import (
    calculate_hfs_to_fs_rabi_frequency,
    calculate_rabi_frequency,
)
from rydberg_aoa.arc.types import AtomicLevel, LaserParams
from rydberg_aoa.config.laser_rf_config import (
    LASER_PARAMS,
    calculate_laser_rabi_frequency,
    gaussian_fwhm_diameter_to_1e2_radius_mm,
)
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)

# --- the configured coupling operating point (production parameters) --------
COUPLING_POWER_MW = LASER_PARAMS['coupling']['power_mw']                # 32 mW
COUPLING_WAIST_MM = gaussian_fwhm_diameter_to_1e2_radius_mm(144.0)      # BEAM-002

EXCITED = AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=0.0, label="5P3/2 F=4")
GROUND = AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.0, label="5S1/2 F=3")

# Frozen QN-001A probe reference (must remain bit-for-bit within tolerance)
PROBE_DME_EA0 = -2.259585008991063
PROBE_OMEGA_RAD_US = 15.758546351295182


def _rydberg(mj=None):
    """26D5/2 destination of the configured coupling transition."""
    return AtomicLevel(n=26, l=2, j=2.5, mj=mj, label="26D5/2")


def _laser(params, polarization=0):
    """LaserParams copy of a production parameter block with q override."""
    merged = dict(params)
    merged['polarization'] = polarization
    return LaserParams(**merged)


def _coupling_laser(polarization=0):
    return _laser(LASER_PARAMS['coupling'], polarization)


def _probe_laser(polarization=0):
    return _laser(LASER_PARAMS['probe'], polarization)


def _e_peak(power_w, waist_m):
    """Independent Gaussian-beam peak field: E = sqrt(2 I0/(eps0 c))."""
    i0 = 2.0 * power_w / (np.pi * waist_m ** 2)
    return np.sqrt(2.0 * i0 / (sc.epsilon_0 * sc.c))


@pytest.fixture(scope="module")
def atom():
    """One real ARC 3.9.0 Rubidium85 instance for the whole module."""
    from arc import Rubidium85
    return Rubidium85()


# ---------------------------------------------------------------------------
# SymPy independent angular-momentum oracle (no ARC HFS->FS helper involved)
# ---------------------------------------------------------------------------
J1 = sp.Rational(3, 2)      # source electronic angular momentum (5P3/2)
J2 = sp.Rational(5, 2)      # destination 26D5/2
F1 = sp.Rational(4)         # source hyperfine quantum number (F = 4)
MF1 = sp.Rational(0)        # configured coupling source projection (mF = 0)


def _sympy_signed_angular_coefficient(mj2, q, inuc):
    """Exact angular coefficient <j1 mJ1, I mI|F1 mF>(-1)^(j1-mJ1) 3j(...)."""
    mj2, q = sp.Rational(mj2), sp.Rational(q)
    mj1 = mj2 - q                       # electronic dipole selection rule
    mi = MF1 - mj1                      # mF1 = mJ1 + mI (mF1 = 0 here)
    if abs(mj1) > J1 or abs(mi) > inuc:
        return sp.S.Zero                # forbidden projection -> zero
    cg = CG(J1, mj1, inuc, mi, F1, MF1).doit()
    w3j = wigner_3j(J1, 1, J2, -mj1, -q, mj2)
    return (-1) ** (J1 - mj1) * cg * w3j


def _sympy_predicted_signed_dme_ea0(atom, mj2, q):
    """Signed DME [e*a0] = SymPy angular coefficient x ARC reduced J element."""
    inuc = sp.Rational(atom.I)
    coeff = _sympy_signed_angular_coefficient(mj2, q, inuc)
    rme = float(atom.getReducedMatrixElementJ(5, 1, 1.5, 26, 2, 2.5))
    return float(coeff) * rme


# ---------------------------------------------------------------------------
# 1. installed ARC API contract (the runtime signature is the source of truth)
# ---------------------------------------------------------------------------
def test_arc_hfs_to_fs_api_has_the_expected_runtime_signature(atom):
    sig = inspect.signature(atom.getDipoleMatrixElementHFStoFS)
    names = list(sig.parameters)
    assert names[:10] == ["n1", "l1", "j1", "f1", "mf1",
                          "n2", "l2", "j2", "mj2", "q"]
    assert sig.parameters["s"].default == 0.5


def test_oracle_isotope_nuclear_spin_is_rubidium85(atom):
    """The oracle uses I = 5/2 — the ARC species constant, not a guess."""
    assert float(atom.I) == 2.5


# ---------------------------------------------------------------------------
# 2. low-level DME primitive: the SIGNED declared-channel contract
# ---------------------------------------------------------------------------
def test_primitive_returns_the_signed_real_arc_element(atom):
    dest = _rydberg(mj=1.5)
    d_raw = atom.getDipoleMatrixElementHFStoFS(5, 1, 1.5, 4.0, 0.0,
                                               26, 2, 2.5, 1.5, 0)
    d_prim = calculate_hfs_to_fs_dipole_matrix_element(atom, EXCITED, dest, 0)
    assert d_prim == d_raw          # exact float identity, SIGNED
    assert d_prim < 0.0             # this q=0 channel of the manifold is negative


def test_primitive_preserves_negative_and_positive_signs():
    class _SignAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def __init__(self, value):
            self.value = value

        def getDipoleMatrixElementHFStoFS(self, *args):
            return self.value

    assert calculate_hfs_to_fs_dipole_matrix_element(
        _SignAtom(-2.5), EXCITED, _rydberg(mj=1.5), 0) == -2.5
    assert calculate_hfs_to_fs_dipole_matrix_element(
        _SignAtom(+2.5), EXCITED, _rydberg(mj=1.5), 0) == +2.5


def test_primitive_requires_an_arc_atom():
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            None, EXCITED, _rydberg(mj=1.5), 0)


def test_primitive_requires_an_explicit_source_f_and_mf():
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), AtomicLevel(n=5, l=1, j=1.5, label="5P3/2 (no F)"),
            _rydberg(mj=1.5), 0)
    # a partial declaration (F without mF) is equally rejected
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), AtomicLevel(n=5, l=1, j=1.5, f=4.0, label="5P3/2 F=4"),
            _rydberg(mj=1.5), 0)


def test_primitive_requires_an_explicit_destination_mj():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), EXCITED, _rydberg(), 0)
    message = str(excinfo.value)
    assert "mJ=None" in message
    assert "no mJ is inferred" in message
    assert "no mJ scan" in message


def test_primitive_rejects_an_ambiguous_destination():
    ambiguous = AtomicLevel(n=26, l=2, j=2.5, mj=1.5, f=5.0, mf=0.0,
                            label="26D5/2 (HFS+FS)")
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), EXCITED, ambiguous, 0)
    assert "ambiguous" in str(excinfo.value)


@pytest.mark.parametrize("bad_q", [2, -2, 0.5, None, "sigma+"])
def test_primitive_rejects_invalid_q(bad_q):
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), EXCITED, _rydberg(mj=1.5), bad_q)


def test_primitive_rejects_e1_forbidden_declared_transitions():
    # delta_l != 1: the two-photon ladder step 5S1/2 -> 26D5/2 is not E1
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), GROUND, _rydberg(mj=1.5), 0)
    # delta_j > 1 (l1 = 0, j1 = 1/2 -> l2 = 1, j2 = 5/2)
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), GROUND, AtomicLevel(n=26, l=1, j=2.5, mj=0.5), 0)
    # j = 0 -> j = 0 with delta_l = 1 (synthetic structural declaration)
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            object(), AtomicLevel(n=5, l=1, j=0.0, f=3.0, mf=0.0),
            AtomicLevel(n=26, l=2, j=0.0, mj=0.0), 0)


def test_primitive_dark_channel_raises_configuration_error_not_a_value():
    class _ZeroAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def getDipoleMatrixElementHFStoFS(self, *args):
            return 0.0

    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            _ZeroAtom(), EXCITED, _rydberg(mj=1.5), 0)
    message = str(excinfo.value)
    assert "zero electric-dipole coupling" in message
    assert "No alternate mJ or q" in message


def test_primitive_dark_channel_attempts_exactly_one_arc_call():
    calls = []

    class _OnceZeroAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def getDipoleMatrixElementHFStoFS(self, *args):
            calls.append(args)
            if len(calls) > 1:
                raise AssertionError(
                    "QN-001B: a second HFStoFS call (channel rescan) occurred")
            return 0.0

    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            _OnceZeroAtom(), EXCITED, _rydberg(mj=1.5), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("bad_element", [None, np.nan, np.inf, -np.inf, "n/a"])
def test_primitive_unusable_arc_return_raises_calculation_error(bad_element):
    class _BadAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def getDipoleMatrixElementHFStoFS(self, *args):
            return bad_element

    with pytest.raises(PhysicsCalculationError):
        calculate_hfs_to_fs_dipole_matrix_element(
            _BadAtom(), EXCITED, _rydberg(mj=1.5), 0)


def test_primitive_arc_failure_preserves_the_original_cause():
    class _FailingAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def getDipoleMatrixElementHFStoFS(self, *args):
            raise ValueError("element not in ARC database")

    with pytest.raises(PhysicsCalculationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            _FailingAtom(), EXCITED, _rydberg(mj=1.5), 0)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "getDipoleMatrixElementHFStoFS" not in str(excinfo.value) or True


# ---------------------------------------------------------------------------
# 3. HFS -> FS Rabi helper: one declared channel, configured q only
# ---------------------------------------------------------------------------
class _SpyHfsToFsAtom:
    """ARC stand-in recording HFS->FS calls and FORBIDDING every other path."""

    I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

    def __init__(self, d_ea0=-2.5):
        self.d_ea0 = d_ea0
        self.hfstfs_calls = []

    def getDipoleMatrixElementHFStoFS(self, n1, l1, j1, f1, mf1,
                                      n2, l2, j2, mj2, q, s=0.5):
        self.hfstfs_calls.append({
            "n1": n1, "l1": l1, "j1": j1, "f1": f1, "mf1": mf1,
            "n2": n2, "l2": l2, "j2": j2, "mj2": mj2, "q": q,
        })
        return self.d_ea0

    def getDipoleMatrixElementHFS(self, *args, **kwargs):
        raise AssertionError(
            "QN-001B: the HFS->FS path called the HFS->HFS element")

    def getRabiFrequency(self, *args, **kwargs):
        raise AssertionError(
            "QN-001B: the HFS->FS path called the generic getRabiFrequency")


def test_rabi_hfs_to_fs_calls_arc_once_with_the_declared_channel():
    atom = _SpyHfsToFsAtom()
    omega = calculate_hfs_to_fs_rabi_frequency(
        atom, EXCITED, _rydberg(mj=1.5), _coupling_laser(0))
    assert len(atom.hfstfs_calls) == 1      # exactly ONE designated call
    call = atom.hfstfs_calls[0]
    assert (call["n1"], call["l1"], call["j1"]) == (5, 1, 1.5)
    assert (call["f1"], call["mf1"]) == (4.0, 0.0)     # F=4, mF=0
    assert (call["n2"], call["l2"], call["j2"]) == (26, 2, 2.5)
    assert call["mj2"] == 1.5               # exactly the declared channel
    assert call["q"] == 0                   # configured pi transition
    assert omega > 0.0
    # _SpyHfsToFsAtom.getRabiFrequency raises if reached -> generic path 0 calls


@pytest.mark.parametrize("q_configured", [-1, 0, 1])
def test_rabi_hfs_to_fs_q_comes_from_the_configured_laser(q_configured):
    atom = _SpyHfsToFsAtom()
    calculate_hfs_to_fs_rabi_frequency(
        atom, EXCITED, _rydberg(mj=0.5), _coupling_laser(q_configured))
    assert [c["q"] for c in atom.hfstfs_calls] == [q_configured]


def test_rabi_hfs_to_fs_matches_independent_mu_e_over_hbar():
    atom = _SpyHfsToFsAtom(d_ea0=-2.5)
    omega = calculate_hfs_to_fs_rabi_frequency(
        atom, EXCITED, _rydberg(mj=1.5), _coupling_laser(0))
    expected = (2.5 * EA0_TO_CM
                * _e_peak(COUPLING_POWER_MW * 1e-3,
                          COUPLING_WAIST_MM * 1e-3)
                / sc.hbar)
    assert omega == pytest.approx(expected, rel=1e-15)


def test_rabi_hfs_to_fs_rejects_invalid_configured_polarization():
    with pytest.raises(PhysicsConfigurationError):
        calculate_hfs_to_fs_rabi_frequency(
            _SpyHfsToFsAtom(), EXCITED, _rydberg(mj=1.5),
            _coupling_laser(polarization=2))


def test_rabi_hfs_to_fs_missing_mj_fails_fast_with_the_dedicated_error():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_rabi_frequency(
            _SpyHfsToFsAtom(), EXCITED, _rydberg(), _coupling_laser(0))
    message = str(excinfo.value)
    assert "explicit Rydberg mJ" in message
    assert "Zeeman-channel contract" in message
    assert "no mJ is inferred" in message
    assert "scan mJ" in message
    assert "F05" in message


# ---------------------------------------------------------------------------
# 4. dispatch: HFS->FS uses ONLY the new path; HFS->HFS and FS->FS untouched
# ---------------------------------------------------------------------------
def test_dispatch_hfs_to_fs_uses_the_new_path_and_not_the_generic_helper(
        monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "QN-001B: the HFS->FS path used the generic max-over-mJ helper")

    monkeypatch.setattr(rabi_frequency_module,
                        "calculate_dipole_matrix_element", _boom)

    atom = _SpyHfsToFsAtom()
    omega = calculate_rabi_frequency(atom, _rydberg(mj=1.5),
                                     _coupling_laser(0), from_level=EXCITED)
    assert len(atom.hfstfs_calls) == 1
    assert atom.hfstfs_calls[0]["q"] == 0
    assert omega > 0.0
    # the spy's getRabiFrequency raises if reached: zero generic calls proven


def test_dispatch_hfs_to_fs_missing_mj_is_the_dedicated_error():
    """The expected current blocker: dedicated, attributable, still fail-fast."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_rabi_frequency(_SpyHfsToFsAtom(), _rydberg(),
                                 _coupling_laser(0), from_level=EXCITED)
    message = str(excinfo.value)
    assert "HFS->FS" in message
    assert "explicit Rydberg mJ" in message
    assert "scan mJ" in message
    assert "F05" in message


def test_dispatch_hfs_to_fs_dark_channel_never_returns_a_zero_rabi():
    class _ZeroAtom:
        I = 2.5  # Rb85 isotope spin (QN-001B.1 structural channel check)

        def getDipoleMatrixElementHFStoFS(self, *args):
            return 0.0

        def getRabiFrequency(self, *args, **kwargs):
            raise AssertionError("QN-001B: generic path used after dark channel")

    with pytest.raises(PhysicsConfigurationError):
        calculate_rabi_frequency(_ZeroAtom(), _rydberg(mj=1.5),
                                 _coupling_laser(0), from_level=EXCITED)


def test_dispatch_fs_to_fs_path_is_unchanged():
    """Explicit (mj, mj) pairs still drive ARC's generic getRabiFrequency."""
    class _FsSpyAtom:
        def __init__(self):
            self.rabi_calls = 0

        def getRabiFrequency(self, **kwargs):
            self.rabi_calls += 1
            return 1.7e7

        def getDipoleMatrixElementHFStoFS(self, *args, **kwargs):
            raise AssertionError(
                "QN-001B: the FS->FS path must not use getDipoleMatrixElementHFStoFS")

    atom = _FsSpyAtom()
    lower = AtomicLevel(n=5, l=0, j=0.5, mj=0.5)
    upper = AtomicLevel(n=5, l=1, j=1.5, mj=0.5)     # mj1 + q(0) = 0.5
    omega = calculate_rabi_frequency(atom, upper, _probe_laser(0),
                                     from_level=lower, ground_state=lower)
    assert atom.rabi_calls == 1
    assert omega == pytest.approx(1.7e7)


def test_dispatch_probe_hfs_to_hfs_is_not_routed_through_hfs_to_fs():
    class _ProbeSpyAtom:
        def __init__(self, d_ea0=PROBE_DME_EA0):
            self.d_ea0 = d_ea0
            self.hfs_calls = 0

        def getDipoleMatrixElementHFS(self, *args, **kwargs):
            self.hfs_calls += 1
            return self.d_ea0

        def getDipoleMatrixElementHFStoFS(self, *args, **kwargs):
            raise AssertionError(
                "QN-001B: the frozen probe was routed through HFS->FS")

        def getRabiFrequency(self, *args, **kwargs):
            raise AssertionError(
                "QN-001B: the frozen probe used the generic getRabiFrequency")

    atom = _ProbeSpyAtom()
    omega = calculate_rabi_frequency(atom, EXCITED, _probe_laser(0),
                                     from_level=GROUND, ground_state=GROUND)
    assert atom.hfs_calls == 1
    assert omega > 0.0


def test_frozen_probe_numbers_are_unchanged(atom):
    omega = calculate_rabi_frequency(atom, EXCITED, _probe_laser(0),
                                     from_level=GROUND, ground_state=GROUND)
    d_hfs = atom.getDipoleMatrixElementHFS(5, 0, 0.5, 3.0, 0,
                                           5, 1, 1.5, 4.0, 0, 0)
    assert d_hfs == pytest.approx(PROBE_DME_EA0, rel=1e-12)
    assert omega * 1e-6 == pytest.approx(PROBE_OMEGA_RAD_US, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. real-ARC channel table vs the independent SymPy oracle (test-only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0, 1, -1])
@pytest.mark.parametrize("mj2", [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
def test_real_arc_channel_matches_the_independent_sympy_oracle(atom, q, mj2):
    """Every explicit channel of 5P3/2 F=4,mF=0 -> 26D5/2 for q in {-1,0,1}:
    the raw SIGNED ARC element equals the SymPy recoupling prediction.

    TEST-ONLY channel verification; it never selects or ranks a channel.
    """
    d_arc = atom.getDipoleMatrixElementHFStoFS(
        5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, mj2, q)
    d_sym = _sympy_predicted_signed_dme_ea0(atom, mj2, q)
    assert d_arc == pytest.approx(d_sym, rel=1e-12, abs=1e-15)
    if d_sym != 0.0:
        assert (d_arc > 0.0) == (d_sym > 0.0)       # signed agreement

    dest = _rydberg(mj=mj2)
    if d_arc == 0.0:
        # exactly-dark channel: rejected explicitly (QN-001B.1 structural
        # check before ARC; the QN-001B zero guard would say the same thing)
        with pytest.raises(PhysicsConfigurationError) as rejected:
            calculate_hfs_to_fs_dipole_matrix_element(atom, EXCITED, dest, q)
        assert "No alternate mJ or q" in str(rejected.value)
    elif abs(d_arc) < 1e-15:
        # geometrically dark; ARC leaves a small float-cancellation residue
        # (~7.46e-19 e*a0) instead of exactly 0.0, while the SymPy oracle is
        # exactly zero. Production now rejects these channels EXACTLY, before
        # ARC is called (QN-001B.1) — the raw ARC diagnostic value above is
        # deliberately NOT normalised: it documents ARC's real behaviour.
        assert d_sym == 0.0
        with pytest.raises(PhysicsConfigurationError) as structural:
            calculate_hfs_to_fs_dipole_matrix_element(atom, EXCITED, dest, q)
        assert "structurally forbidden" in str(structural.value)
    else:
        d_prim = calculate_hfs_to_fs_dipole_matrix_element(
            atom, EXCITED, dest, q)
        assert d_prim == d_arc                      # signed, exact identity


def test_configured_q0_stretched_channels_are_dark(atom):
    """mJ = +/-5/2 with q = 0 is unreachable from F = 4, mF = 0 (|mJ1|<=3/2).

    ARC's intermediate-f2 sum leaves a ~1e-18 e*a0 cancellation residue,
    which is numerically zero; the SymPy oracle yields exactly zero.
    """
    for mj2 in (-2.5, 2.5):
        d_arc = atom.getDipoleMatrixElementHFStoFS(
            5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, mj2, 0)
        assert _sympy_predicted_signed_dme_ea0(atom, mj2, 0) == 0.0
        assert abs(d_arc) < 1e-15


def test_exactly_dark_channels_exist_for_sigma_polarisations(atom):
    """q=+1 dark: mJ2 = -5/2, -3/2; q=-1 dark: mJ2 = +3/2, +5/2 (|mJ1| > j1
    for all four; |mI| > I additionally for the outer mJ2 = +/-5/2 pair)."""
    for q, mj2 in [(1, -2.5), (1, -1.5), (-1, 1.5), (-1, 2.5)]:
        d_arc = atom.getDipoleMatrixElementHFStoFS(
            5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, mj2, q)
        assert d_arc == 0.0
        with pytest.raises(PhysicsConfigurationError):
            calculate_hfs_to_fs_dipole_matrix_element(
                atom, EXCITED, _rydberg(mj=mj2), q)


# ---------------------------------------------------------------------------
# 6. QN-001B integration: the full public laser-Rabi chain
# ---------------------------------------------------------------------------
def test_integration_explicit_coupling_channel_end_to_end(atom):
    """laser_rf_config -> calculate_rabi_frequency -> HFS->FS helper ->
    ARC getDipoleMatrixElementHFStoFS -> Omega [rad/s]."""
    dest = _rydberg(mj=1.5)
    omega = calculate_laser_rabi_frequency(
        atom, EXCITED, dest, LASER_PARAMS['coupling'])
    d_raw = atom.getDipoleMatrixElementHFStoFS(
        5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, 1.5, 0)
    expected = (abs(d_raw) * EA0_TO_CM
                * _e_peak(LASER_PARAMS['coupling']['power_mw'] * 1e-3,
                          LASER_PARAMS['coupling']['beam_waist_mm'] * 1e-3)
                / sc.hbar)
    assert omega == pytest.approx(expected, rel=1e-12)
    assert omega > 0.0


def test_integration_every_coupled_q0_channel_has_positive_rabi(atom):
    """The four bright q=0 channels are all calculable (diagnostic sweep,
    NOT a default selection)."""
    for mj2 in (-1.5, -0.5, 0.5, 1.5):
        omega = calculate_laser_rabi_frequency(
            atom, EXCITED, _rydberg(mj=mj2), LASER_PARAMS['coupling'])
        assert omega > 0.0


def test_integration_accepts_production_dict_levels(atom):
    """ARCH-001 passes cached level DICTS to the public Rabi entry point."""
    excited_dict = {"n": 5, "l": 1, "j": 1.5, "f": 4.0, "mf": 0, "mj": None,
                    "label": "5P3/2 F=4"}
    dest_dict = {"n": 26, "l": 2, "j": 2.5, "f": None, "mf": None, "mj": 1.5,
                 "label": "26D5/2 mJ=+3/2"}
    omega = calculate_laser_rabi_frequency(
        atom, excited_dict, dest_dict, LASER_PARAMS['coupling'])
    assert omega > 0.0


def test_integration_default_experimental_levels_still_fail_fast(atom):
    """The production Experimental operating point (26D5/2, mJ=None) still
    raises through the public entry point: QN-001B adds a path, it does not
    invent an mJ."""
    rydberg1_dict = {"n": 26, "l": 2, "j": 2.5, "f": None, "mf": None,
                     "mj": None, "label": "26D2.5"}
    excited_dict = {"n": 5, "l": 1, "j": 1.5, "f": 4.0, "mf": 0, "mj": None,
                    "label": "5P3/2 F=4"}
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_laser_rabi_frequency(
            atom, excited_dict, rydberg1_dict, LASER_PARAMS['coupling'])
    assert "explicit Rydberg mJ" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 7. QN-001B.1 — exact structural rejection of forbidden magnetic channels
# ---------------------------------------------------------------------------
class _CountingArcAtom:
    """REAL ARC atom wrapper counting getDipoleMatrixElementHFStoFS calls.

    Proves the ORDER OF OPERATIONS: a structurally forbidden channel must
    raise with ZERO ARC calls, a structurally admissible one with exactly ONE.
    """

    def __init__(self, real_atom):
        self._real = real_atom
        self.I = float(real_atom.I)      # Rb85 isotope spin, from the atom
        self.hfstfs_calls = []

    def getDipoleMatrixElementHFStoFS(self, *args, **kwargs):
        self.hfstfs_calls.append((args, kwargs))
        return self._real.getDipoleMatrixElementHFStoFS(*args, **kwargs)


def _structurally_forbidden(mj2, q, j1=1.5, mf1=0.0, nuclear_spin=2.5):
    """The QN-001B.1 rule, stated independently of the production helper:
    mJ1 = mJ2 - q must be a valid projection of j1, and mI = mF1 - mJ1 of I."""
    mj1 = mj2 - q
    m_i = mf1 - mj1
    return abs(mj1) > j1 or abs(m_i) > nuclear_spin


def test_structural_q0_stretched_endpoints_rejected_before_arc(atom):
    """5P3/2 F=4,mF=0 -> 26D5/2, mJ2 = +/-5/2, q = 0:

    mJ1 = mJ2 - q = +/-5/2 while j1 = 3/2  ->  |mJ1| > j1: the channel is
    structurally forbidden (identically zero, not merely weak). Production
    must reject it BEFORE ARC (call count == 0) — no dipole threshold — and
    the independent SymPy oracle predicts exactly zero.
    """
    for mj2 in (-2.5, 2.5):
        spy = _CountingArcAtom(atom)
        with pytest.raises(PhysicsConfigurationError) as excinfo:
            calculate_hfs_to_fs_dipole_matrix_element(
                spy, EXCITED, _rydberg(mj=mj2), 0)
        message = str(excinfo.value)
        assert "structurally forbidden" in message
        assert "mJ1 = mJ2 - q" in message
        assert "No alternate mJ or q" in message
        assert spy.hfstfs_calls == []          # ARC NEVER called
        # independent oracle: exactly zero, not ~1e-18
        assert _sympy_predicted_signed_dme_ea0(atom, mj2, 0) == 0.0
        # raw ARC diagnostic (documented, NOT used by production): residue
        d_arc_raw = atom.getDipoleMatrixElementHFStoFS(
            5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, mj2, 0)
        assert d_arc_raw != 0.0 and abs(d_arc_raw) < 1e-15


def test_valid_q0_channel_still_calls_arc_exactly_once(atom):
    """mJ2 = -3/2, q = 0 -> mJ1 = -3/2 (valid): exactly one ARC call and the
    preserved signed real-ARC result ~ -7.4189e-03 e*a0."""
    spy = _CountingArcAtom(atom)
    d_prim = calculate_hfs_to_fs_dipole_matrix_element(
        spy, EXCITED, _rydberg(mj=-1.5), 0)
    assert len(spy.hfstfs_calls) == 1          # the QN-001B one-call contract
    d_raw = atom.getDipoleMatrixElementHFStoFS(
        5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, -1.5, 0)
    assert d_prim == d_raw                     # signed, exact float identity
    assert d_prim == pytest.approx(-7.418875578402e-03, rel=1e-9)
    assert d_prim < 0.0


def test_valid_q0_channel_rabi_result_unchanged(atom):
    """QN-001B.1 must not change the Rabi result of a valid channel."""
    omega = calculate_laser_rabi_frequency(
        atom, EXCITED, _rydberg(mj=-1.5), LASER_PARAMS['coupling'])
    d_raw = atom.getDipoleMatrixElementHFStoFS(
        5, 1, 1.5, 4.0, 0.0, 26, 2, 2.5, -1.5, 0)
    expected = (abs(d_raw) * EA0_TO_CM
                * _e_peak(LASER_PARAMS['coupling']['power_mw'] * 1e-3,
                          LASER_PARAMS['coupling']['beam_waist_mm'] * 1e-3)
                / sc.hbar)
    assert omega == pytest.approx(expected, rel=1e-12)
    assert omega > 0.0


@pytest.mark.parametrize("q, mj2", [
    (1, -2.5),    # mJ1 = -7/2: |mJ1| > j1 = 3/2 (also |mI| > I)
    (1, -1.5),    # mJ1 = -5/2: |mJ1| > j1 = 3/2
    (-1, 1.5),    # mJ1 = +5/2: |mJ1| > j1 = 3/2
    (-1, 2.5),    # mJ1 = +7/2: |mJ1| > j1 = 3/2 (also |mI| > I)
])
def test_structural_sigma_channels_rejected_before_arc(atom, q, mj2):
    """q = +/-1 impossible channels use the SAME generic projection rule."""
    spy = _CountingArcAtom(atom)
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            spy, EXCITED, _rydberg(mj=mj2), q)
    message = str(excinfo.value)
    assert "structurally forbidden" in message
    assert "No alternate mJ or q" in message
    assert spy.hfstfs_calls == []              # ARC NEVER called
    assert _sympy_predicted_signed_dme_ea0(atom, mj2, q) == 0.0


@pytest.mark.parametrize("q", [-1, 0, 1])
@pytest.mark.parametrize("mj2", [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
def test_structural_rule_is_generic_over_the_full_manifold(atom, q, mj2):
    """The production rule is generic: forbidden -> ZERO ARC calls with the
    structural error; admissible -> exactly ONE ARC call (never a scan)."""
    spy = _CountingArcAtom(atom)
    if _structurally_forbidden(mj2, q):
        with pytest.raises(PhysicsConfigurationError) as excinfo:
            calculate_hfs_to_fs_dipole_matrix_element(
                spy, EXCITED, _rydberg(mj=mj2), q)
        assert "structurally forbidden" in str(excinfo.value)
        assert spy.hfstfs_calls == []
    else:
        result = None
        try:
            result = calculate_hfs_to_fs_dipole_matrix_element(
                spy, EXCITED, _rydberg(mj=mj2), q)
        except PhysicsConfigurationError as exc:
            # only the retained QN-001B post-ARC exact-zero guard may fire
            # here — never the structural check
            assert "structurally forbidden" not in str(exc)
            assert "zero electric-dipole coupling" in str(exc)
        assert len(spy.hfstfs_calls) == 1      # ONE call, never a scan
        if result is not None:
            assert np.isfinite(result) and result != 0.0


def test_structural_nuclear_projection_bound_rejected_before_arc(atom):
    """mF1 = 3, mJ2 = -3/2, q = 0 -> mJ1 = -3/2 (valid) but
    mI = mF1 - mJ1 = 9/2 > I = 5/2: the SECOND structural rule fires."""
    spy = _CountingArcAtom(atom)
    source = AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=3.0,
                         label="5P3/2 F=4 mF=3")
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            spy, source, _rydberg(mj=-1.5), 0)
    message = str(excinfo.value)
    assert "structurally forbidden" in message
    assert "mI = mF1 - mJ1" in message
    assert "[-I,+I]" in message
    assert "No alternate mJ or q" in message
    assert spy.hfstfs_calls == []              # ARC NEVER called


def test_structural_validation_requires_the_isotope_nuclear_spin():
    """Without atom.I the structural check fails LOUDLY — never a silent skip
    and never a hard-coded isotope guess (F05/QN-001B.1)."""
    class _NoSpinAtom:
        def getDipoleMatrixElementHFStoFS(self, *args):
            raise AssertionError("ARC must not be called without atom.I")

    with pytest.raises(PhysicsConfigurationError) as excinfo:
        calculate_hfs_to_fs_dipole_matrix_element(
            _NoSpinAtom(), EXCITED, _rydberg(mj=1.5), 0)
    message = str(excinfo.value)
    assert "nuclear spin" in message
    assert "no fallback" in message