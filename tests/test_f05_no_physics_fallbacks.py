"""F05 — no silent physics fallbacks: fail-fast contract regression tests.

These tests encode the F05 contract:

    For every PHYSICS-DERIVED quantity, if the designated calculation path
    fails, returns invalid data, or returns a physically unusable value, the
    production code MUST raise a :class:`rydberg_aoa.exceptions.PhysicsError`
    subclass instead of substituting 0 / None / a cached value / an
    approximate formula / a hard-coded constant / a config default.

Two classes of tests are used deliberately:

* **behavioural** tests with fake atom objects - they prove what the
  production code does in each failure mode without depending on the ARC
  database, and they assert that the REMOVED substitute (the dipole-based
  laser-Rabi formula, the placeholder dipoles, the 3.58e-30 probe fallback)
  is never invoked;
* **source** tests - they prove that the removed fabricated constants are gone
  from the production sources, so a future edit cannot silently reintroduce
  one of them.

QN-001A status (the mF/mJ defect itself is now FIXED): ``rydberg_aoa.arc.types.AtomicLevel``
(and its ``level_finder`` mirror) no longer manufacture ``mj`` from ``mf`` nor
default it to ``0``; ``mj``/``mf`` are independent optional quantum numbers that
are validated when present (``tests/test_qn001a_mf_mj_contract.py`` pins that).
The probe transition is now computed by the designated PRIMARY ARC HFS path
(``getDipoleMatrixElementHFS`` + ``Omega = |d| E_peak / hbar``).

Remaining OPEN blocker (documented, NOT hidden): the coupling transition
``5P3/2 F=4, mF=0 -> 26D5/2`` is HFS -> fine-structure and the production model
still declares no Rydberg mJ / Zeeman channel. ``calculate_rabi_frequency``
therefore fails fast with ``PhysicsConfigurationError`` inside the explicit
ARCH-001 seam ``rydberg_aoa.config.atomic_config.initialize_atomic_physics()``
(importing the module publishes no physics at all).
``test_atomic_config_import_is_never_silently_zero`` pins the requirement that
this failure stays explicit either way.
"""
import importlib
import pathlib

import numpy as np
import pytest
from scipy import constants as sc

import rydberg_aoa
import rydberg_aoa.arc.dipole as dipole_module
import rydberg_aoa.arc.dipole_matrix as dipole_matrix_module
import rydberg_aoa.arc.rabi_frequency as rabi_frequency
from rydberg_aoa.arc.types import AtomicLevel, LaserParams, RFParams
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)

HBAR = sc.hbar

GROUND = AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.0, label="5S1/2 F=3")
EXCITED = AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=0.0, label="5P3/2 F=4")
RYD1 = AtomicLevel(n=26, l=2, j=2.5, label="26D2.5")
RYD2 = AtomicLevel(n=27, l=1, j=1.5, label="27P1.5")

PROBE_LASER = LaserParams(wavelength_nm=780.241476, power_mw=0.0001,
                          beam_waist_mm=0.07983624922707379,
                          detuning_mhz=0.0, polarization=0)
RF_FIELD = RFParams(frequency_hz=132.649e9, electric_field_vm=60.0,
                    polarization=0)

# QN-001A: the generic fine-structure path (getRabiFrequency) is now reached
# only by transitions WITHOUT hyperfine resolution, and it requires an
# EXPLICIT mJ on the driving level. The fine-structure fail-fast contract
# tests below therefore use FS levels carrying a valid mj — using the
# HFS-resolved GROUND/EXCITED instead would (wrongly) enter the new primary
# HFS path and test a different branch.
FS_GROUND = AtomicLevel(n=5, l=0, j=0.5, mj=0.5, label="5S1/2 (FS)")
FS_EXCITED = AtomicLevel(n=5, l=1, j=1.5, mj=0.5, label="5P3/2 (FS)")


class _ZeroRabiAtom:
    """ARC stand-in whose fine-structure Rabi element vanishes."""

    def getRabiFrequency(self, **kwargs):
        return 0.0


class _ValidRabiAtom:
    """ARC stand-in returning a physically usable angular Rabi frequency."""

    OMEGA = 1.703e7  # rad/s, measured for mj = +-0.5 of the probe transition

    def getRabiFrequency(self, **kwargs):
        return self.OMEGA


class _FailingRabiAtom:
    def getRabiFrequency(self, **kwargs):
        raise ValueError("ARC element unavailable")


class _UnavailableElementAtom:
    """ARC stand-in whose fine-structure dipole elements all fail."""

    def getDipoleMatrixElement(self, *args, **kwargs):
        raise KeyError("element not in ARC database")


class _ValidElementAtom:
    """ARC stand-in returning a finite fine-structure dipole element [ea0]."""

    def __init__(self, element=2.4406307548801123):
        self.element = element

    def getDipoleMatrixElement(self, *args, **kwargs):
        return self.element


def _forbid_dipole_substitute(monkeypatch):
    """Make any use of the removed dipole-based laser-Rabi path an error."""
    def _boom(*args, **kwargs):
        raise AssertionError(
            "F05 regression: the laser-Rabi path called the removed "
            "dipole-based substitute")

    monkeypatch.setattr(rabi_frequency, "calculate_dipole_matrix_element", _boom)


def _import_or_skip(module_name):
    """Import a production module or report the test as BLOCKED (never PASS).

    While the open coupling mJ / Zeeman-channel contract (see the module
    docstring) keeps the ``rydberg_aoa.physics`` package un-importable, the
    dependent assertions cannot run; they must be reported as blocked rather
    than silently green.
    """
    try:
        return importlib.import_module(module_name)
    except PhysicsError as exc:
        pytest.skip("BLOCKED by the open F05 fail-fast state: %s" % exc)


# ---------------------------------------------------------------------------
# 1. exception hierarchy (backward compatibility with RuntimeError contracts)
# ---------------------------------------------------------------------------
def test_physics_error_derives_from_runtime_error():
    """RF-003 / decay-rate contracts use ``except RuntimeError``."""
    assert issubclass(PhysicsError, RuntimeError)


def test_physics_error_subclasses_are_distinct():
    assert issubclass(PhysicsConfigurationError, PhysicsError)
    assert issubclass(PhysicsCalculationError, PhysicsError)
    assert not issubclass(PhysicsConfigurationError, PhysicsCalculationError)


# ---------------------------------------------------------------------------
# 2. the fabricated substitutes are gone from the production sources
# ---------------------------------------------------------------------------
PACKAGE = pathlib.Path(rydberg_aoa.__file__).parent

REMOVED_TOKENS = [
    ("arc/dipole.py", "1e-35"),
    ("arc/dipole.py", "1e-32"),
    ("arc/dipole.py", "'dipole_si': dipole_si_min"),
    ("arc/dipole.py", "placeholder_dipole_si"),
    ("arc/dipole_matrix.py", "3.58e-30"),
    ("arc/transitions.py", "1e-28"),
    ("arc/interface.py", "dipole_moment=1e-30"),
    ("arc/interface.py", "Placeholder value"),
    ("arc/interface.py", "manual n+1"),
    ("arc/rabi_frequency.py", "Falling back to dipole-based calculation"),
    ("config/atomic_config.py", "1.0e15"),
    ("config/atomic_config.py", "'delta_p_rad_us': 0.0"),
    ("level_finder/arc_level_finder.py", "using default 480.0 nm"),
    ("level_finder/arc_level_finder.py", "dipole_matrix = None"),
    # F05-16 / F05-17 (completion pass): the density fallback lives in
    # config/parameters.py — the atomic_config guard above never covered it —
    # and config/parameter_manager.py carried placeholder physics constants.
    ("config/parameters.py", "1.0e15"),
    ("config/parameter_manager.py", "Placeholder - would"),
    ("config/parameter_manager.py", "Placeholder - 0 for resonance"),
    ("config/parameter_manager.py", "return 0.0, 0.0, 0.0"),
    # the three exact historic placeholder Rabi constants must never return
    ("config/parameter_manager.py", "1.36e7"),
    ("config/parameter_manager.py", "8.304e7"),
    ("config/parameter_manager.py", "1.0643e8"),
]

FAIL_FAST_MODULES = [
    "arc/rabi_frequency.py",
    "arc/dipole.py",
    "arc/dipole_matrix.py",
    "arc/interface.py",
    "arc/transitions.py",
    "config/atomic_config.py",
    "config/laser_rf_config.py",
    "config/parameters.py",
    "config/parameter_manager.py",
    "level_finder/arc_level_finder.py",
    "physics/detuning/calculate_all_detunings.py",
]


@pytest.mark.parametrize("relative_path,token", REMOVED_TOKENS)
def test_fabricated_substitute_removed_from_source(relative_path, token):
    source = (PACKAGE / relative_path).read_text(encoding="utf-8")
    assert token not in source, (
        "F05 regression: the fabricated substitute %r is back in %s"
        % (token, relative_path))


@pytest.mark.parametrize("relative_path", FAIL_FAST_MODULES)
def test_fail_fast_module_uses_the_physics_exception_hierarchy(relative_path):
    source = (PACKAGE / relative_path).read_text(encoding="utf-8")
    assert "F05" in source, "%s carries no F05 annotation" % relative_path
    assert ("from ..exceptions import" in source
            or "from ...exceptions import" in source), (
        "%s does not import the physics exception hierarchy" % relative_path)


# ---------------------------------------------------------------------------
# 3. laser Rabi frequency: fail fast, never substitute
# ---------------------------------------------------------------------------
def test_laser_rabi_requires_arc_atom():
    with pytest.raises(PhysicsConfigurationError):
        rabi_frequency.calculate_rabi_frequency(
            None, EXCITED, PROBE_LASER, from_level=GROUND, ground_state=GROUND)


def test_laser_rabi_requires_from_level():
    with pytest.raises(PhysicsConfigurationError):
        rabi_frequency.calculate_rabi_frequency(
            _ValidRabiAtom(), EXCITED, PROBE_LASER, from_level=None,
            ground_state=None)


def test_laser_rabi_rejects_non_positive_laser_parameters():
    dead_laser = LaserParams(wavelength_nm=780.241476, power_mw=0.0,
                             beam_waist_mm=0.07983624922707379,
                             detuning_mhz=0.0, polarization=0)
    with pytest.raises(PhysicsConfigurationError):
        rabi_frequency.calculate_rabi_frequency(
            _ValidRabiAtom(), EXCITED, dead_laser, from_level=GROUND,
            ground_state=GROUND)


def test_laser_rabi_returns_arc_value_and_never_calls_the_dipole_path(monkeypatch):
    """Positive control: a usable ARC result is returned unchanged [rad/s]."""
    _forbid_dipole_substitute(monkeypatch)
    omega = rabi_frequency.calculate_rabi_frequency(
        _ValidRabiAtom(), FS_EXCITED, PROBE_LASER, from_level=FS_GROUND,
        ground_state=FS_GROUND)
    assert omega == pytest.approx(_ValidRabiAtom.OMEGA)


def test_laser_rabi_raises_when_arc_returns_zero(monkeypatch):
    """The historical dipole-based substitution must be gone: a zero ARC
    element is a calculation failure for a DRIVEN transition."""
    _forbid_dipole_substitute(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        rabi_frequency.calculate_rabi_frequency(
            _ZeroRabiAtom(), FS_EXCITED, PROBE_LASER, from_level=FS_GROUND,
            ground_state=FS_GROUND)
    message = str(excinfo.value)
    assert "non-positive" in message
    assert "mj1" in message and "0" in message


def test_laser_rabi_rejects_nan_and_none_results(monkeypatch):
    _forbid_dipole_substitute(monkeypatch)

    class _NanAtom:
        def getRabiFrequency(self, **kwargs):
            return np.nan

    with pytest.raises(PhysicsCalculationError):
        rabi_frequency.calculate_rabi_frequency(
            _NanAtom(), FS_EXCITED, PROBE_LASER, from_level=FS_GROUND,
            ground_state=FS_GROUND)


def test_laser_rabi_wraps_arc_failure_and_preserves_the_cause(monkeypatch):
    _forbid_dipole_substitute(monkeypatch)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        rabi_frequency.calculate_rabi_frequency(
            _FailingRabiAtom(), FS_EXCITED, PROBE_LASER, from_level=FS_GROUND,
            ground_state=FS_GROUND)
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# 4. RF Rabi frequency: fail fast, never substitute
# ---------------------------------------------------------------------------
def test_rf_rabi_requires_arc_atom():
    with pytest.raises(PhysicsConfigurationError):
        rabi_frequency.calculate_rf_rabi_frequency(None, RYD1, RYD2, RF_FIELD)


def test_rf_rabi_uses_the_physical_dipole_and_field(monkeypatch):
    """Positive control against an independent oracle: Omega = |d| E / hbar.

    QN-001C: the RF path takes the SIGNED FS->FS element [e*a0] of exactly
    the ONE declared channel from the dedicated primitive, applies the
    single e*a0 -> C*m conversion, and then the C-001 formula. A negative
    element proves the sign is preserved by the primitive and magnitude-
    taken exactly once here.
    """
    dipole_ea0 = -522.6488589818568            # SIGNED channel element [e*a0]
    monkeypatch.setattr(rabi_frequency,
                        "calculate_fs_to_fs_dipole_matrix_element",
                        lambda *args, **kwargs: dipole_ea0)
    omega = rabi_frequency.calculate_rf_rabi_frequency(
        object(), RYD1, RYD2, RF_FIELD)
    expected = (abs(dipole_ea0) * dipole_module.EA0_TO_CM
                * RF_FIELD.electric_field_vm / HBAR)
    assert omega == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("bad_dipole", [
    0.0, -0.0, None, np.nan, np.inf, {}, {"dipole_si": 4.4310170264481825e-27},
    {"element": 1.0}, "not-a-number",
])
def test_rf_rabi_raises_on_unusable_dipole(monkeypatch, bad_dipole):
    """A contract-violating primitive return (old dict contract, zero, None,
    non-numeric, non-finite) must raise, never propagate into Omega."""
    monkeypatch.setattr(rabi_frequency,
                        "calculate_fs_to_fs_dipole_matrix_element",
                        lambda *args, **kwargs: bad_dipole)
    with pytest.raises(PhysicsCalculationError):
        rabi_frequency.calculate_rf_rabi_frequency(
            object(), RYD1, RYD2, RF_FIELD)


# ---------------------------------------------------------------------------
# 5. dipole matrix elements: fail fast, never fabricate
# ---------------------------------------------------------------------------
def test_dipole_element_requires_arc_atom():
    with pytest.raises(PhysicsConfigurationError):
        dipole_module.calculate_dipole_matrix_element(None, GROUND, 5, 1, 1.5)


def test_dipole_element_keeps_selection_rule_zero():
    """A parity-forbidden element is a legitimate physics zero, not a failure."""
    value = dipole_module.calculate_dipole_matrix_element(
        _ValidElementAtom(), GROUND, 5, 0, 0.5)   # delta_l = 0 -> forbidden
    assert value == 0.0


def test_dipole_element_returns_the_physical_matrix_element():
    result = dipole_module.calculate_dipole_matrix_element(
        _ValidElementAtom(), RYD1, 27, 1, 1.5)
    assert isinstance(result, dict)
    assert result["dipole_si"] > 0.0


def test_dipole_element_never_fabricates_a_placeholder():
    """Every fine-structure element failing must raise - not return 1e-32."""
    with pytest.raises(PhysicsCalculationError):
        dipole_module.calculate_dipole_matrix_element(
            _UnavailableElementAtom(), RYD1, 27, 1, 1.5)


def test_calculate_dipole_matrix_raises_on_unusable_element(monkeypatch):
    monkeypatch.setattr(dipole_module, "calculate_dipole_matrix_element",
                        lambda *args, **kwargs: 0.0)
    levels = {"ground": GROUND, "excited": EXCITED, "rydberg1": RYD1,
              "rydberg2": RYD2}
    with pytest.raises(PhysicsCalculationError):
        dipole_matrix_module.calculate_dipole_matrix(levels, _ValidElementAtom())


# ---------------------------------------------------------------------------
# 6. detuning inputs are validated (no default/zero substitution)
# ---------------------------------------------------------------------------
def test_detunings_require_every_input_and_reject_invalid_values():
    module = _import_or_skip(
        "rydberg_aoa.physics.detuning.calculate_all_detunings")
    levels = {"ground": GROUND, "excited": EXCITED, "rydberg1": RYD1,
              "rydberg2": RYD2}

    with pytest.raises(PhysicsConfigurationError):
        module.calculate_all_detunings(
            _ValidElementAtom(), levels,
            {"wavelength_nm": 780.241476}, {"wavelength_nm": 483.4}, {})

    with pytest.raises(PhysicsConfigurationError):
        module.calculate_all_detunings(
            _ValidElementAtom(), levels,
            {"wavelength_nm": None}, {"wavelength_nm": 483.4},
            {"frequency_hz": 132.649e9})


# ---------------------------------------------------------------------------
# 7. the exposed configuration defect must stay EXPLICIT, never a silent 0.0
# ---------------------------------------------------------------------------
def test_atomic_config_import_is_never_silently_zero():
    """ARCH-001: importing ``atomic_config`` publishes NOTHING; the F05
    fail-fast moved to the explicit ``initialize_atomic_physics()`` seam.

    Forbidden outcome (the one F05 removed): a SUCCESSFUL initialization whose
    Rabi frequencies are silently zero.  A blocked pipeline must fail loudly
    with a PhysicsError naming F05, or stay unpublished - never a silent 0.0.
    """
    try:
        acfg = importlib.import_module("rydberg_aoa.config.atomic_config")
    except PhysicsError as exc:
        message = str(exc)
        assert "F05" in message, (
            "the fail-fast error must name the F05 contract: %s" % message)
        pytest.skip("BLOCKED: atomic_config fails fast at import -> %s" % message)

    # ARCH-001: a plain import runs no physics and publishes no values, so no
    # silently-zero Rabi frequency can be observed before the seam ran.
    assert acfg.is_atomic_physics_initialized() is False, (
        "importing atomic_config initialized atomic physics (ARCH-001)")
    for name in ("OMEGA_P", "OMEGA_C", "OMEGA_RF_BASE", "OMEGA_SIGNAL_RF_BASE"):
        assert getattr(acfg, name) is None, (
            "F05: %s was published by a plain import (must stay unavailable "
            "until the explicit seam runs)" % name)

    # The fail-fast now lives at the explicit seam: it may either raise an F05
    # PhysicsError (current open QN-001A coupling contract) or complete, in
    # which case every Rabi frequency must be strictly positive.
    try:
        acfg.initialize_atomic_physics()
    except PhysicsError as exc:
        message = str(exc)
        assert "F05" in message, (
            "the fail-fast error must name the F05 contract: %s" % message)
        pytest.skip("BLOCKED: atomic_config fails fast on the open coupling "
                    "mJ / Zeeman-channel contract -> %s" % message)

    for name in ("OMEGA_P", "OMEGA_C", "OMEGA_RF_BASE", "OMEGA_SIGNAL_RF_BASE"):
        value = float(getattr(acfg, name))
        assert np.isfinite(value) and value > 0.0, (
            "F05: %s was silently resolved to %r" % (name, value))


# ---------------------------------------------------------------------------
# 8. F05 completion pass — the two category-A leftovers (O-2 + manager)
# ---------------------------------------------------------------------------
def test_atomic_density_fails_fast_instead_of_default_constant(configured_atom, monkeypatch):
    """F05-16 (report item O-2, previously BLOCKED): ARC density failure
    must raise a PhysicsError naming F05, never return the 1.0e15 constant."""
    parameters = importlib.import_module("rydberg_aoa.config.parameters")

    # ATOM-001 uses the already-selected ARC instance, not a second Rb85
    # constructor. Inject failure at its actual density boundary.
    acfg = importlib.import_module('rydberg_aoa.config.atomic_config')
    error = RuntimeError('ARC vapor-pressure model unavailable')
    def fail_density(temperature):
        raise error
    monkeypatch.setattr(acfg.ATOM, 'getNumberDensity', fail_density)
    with pytest.raises(PhysicsCalculationError) as excinfo:
        parameters.calculate_atomic_density_with_arc()
    assert "F05" in str(excinfo.value)
    assert excinfo.value.__cause__ is error


def test_parameter_manager_placeholder_calculators_raise():
    """F05-17: the (dead but importable) manager must raise, never emit the
    historical placeholder Rabi constants or resonance-zero detunings."""
    from rydberg_aoa.config.parameter_manager import ParameterManager

    manager = ParameterManager()
    for name in ("_calculate_probe_rabi", "_calculate_coupling_rabi",
                 "_calculate_rf_rabi", "_calculate_coupling_detuning",
                 "_calculate_rf_detuning"):
        with pytest.raises(PhysicsError):
            getattr(manager, name)()


def test_parameter_manager_missing_inputs_raise_instead_of_zero_detunings():
    """F05-17: missing inputs -> PhysicsError, never the (0.0, 0.0, 0.0)
    resonance substitution for ``all_detunings``."""
    from rydberg_aoa.config.parameter_manager import ParameterManager

    manager = ParameterManager()
    manager.load_defaults()          # 'levels' stays None until optimization
    with pytest.raises(PhysicsError):
        manager.get("all_detunings")





