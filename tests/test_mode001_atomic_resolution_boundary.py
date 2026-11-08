"""MODE-001 — atomic-resolution mode boundary: PRESCRIBED vs EXPERIMENTAL.

What this file pins
-------------------
1. Backward compatibility — the configured default is EXPERIMENTAL and
   ``initialize_atomic_physics()`` with no arguments keeps resolving exactly
   as before (the experimental cache key stays bit-identical).
2. Ambiguous configuration fails loudly (F05) — EXPERIMENTAL may not carry a
   ladder, PRESCRIBED may not be missing one, an unknown mode is never guessed.
3. Declaration contract — a PRESCRIBED ladder must carry an EXPLICIT Rydberg
   ``mJ``; ``mF``/``mJ`` are never inferred from each other; the structural E1
   rules are enforced before ARC is called.
4. Zero search — while resolving a prescribed ladder the level optimizer, the
   level finder, the legacy ``arc/dipole_matrix`` max-scan and the generic
   ``calculate_dipole_matrix_element`` q/mJ sweep are NEVER called; the three
   QN-001A.1/B/C declared-channel primitives are.
5. Wavelength immutability (section 13) — the applied probe/coupling
   wavelengths and the RF frequency/field are inputs and survive a prescribed
   resolution untouched.
6. Detuning convention (section 14) — ``Delta = omega_applied - omega_atomic``
   in ``rad/us``, checked against an ARC-energy oracle that does not use the
   project's own detuning helpers.
7. LO/signal RF polarisation consistency (sections 19/20) fails loudly when
   the two fields address different magnetic channels.
8. Cache identity — prescribed and experimental keys can never collide and two
   declared ladders differing only in a Rydberg ``mJ`` get different keys.

Scope note: example ``mJ`` values appear ONLY in this file (tests/docs), never
in production configuration — that is what
``test_production_has_no_default_prescribed_ladder`` asserts.
"""

import copy

import numpy as np
import pytest

from rydberg_aoa.arc.types import AtomicLevel
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import atomic_resolution as ar
from rydberg_aoa.config.laser_rf_config import (
    LASER_PARAMS,
    RF_PARAMS,
    SIGNAL_RF_PARAMS,
)
from rydberg_aoa.exceptions import PhysicsConfigurationError
from tests.atomic_init_support import (
    restore_published_atomic_state,
    snapshot_published_atomic_state,
)

# Internal seam exercised directly (private by convention, public by contract).
_require_lo_signal_rf_polarisation_consistency = (
    acfg._require_lo_signal_rf_polarisation_consistency)

# ---------------------------------------------------------------------------
# The declared ladder used by the runtime tests.
#
# These mF/mJ values are a TEST FIXTURE, not production configuration: the
# project deliberately ships no default Rydberg mJ (MODE-001 / QN-001C), so an
# example declaration lives here and nowhere else.  Every field is pi (q = 0)
# as configured in laser_rf_config.
# ---------------------------------------------------------------------------
def prescribed_spec(**overrides):
    """A structurally valid four-manifold declaration (q = 0 everywhere)."""
    levels = {
        "ground": AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.0,
                              label="5S1/2 F=3"),
        "excited": AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=0.0,
                               label="5P3/2 F=4"),
        # j = 5/2 admits only HALF-INTEGER mJ; q = 0 => mJ2 = mJ1.
        "rydberg1": AtomicLevel(n=26, l=2, j=2.5, mj=-0.5, label="26D5/2"),
        "rydberg2": AtomicLevel(n=27, l=1, j=1.5, mj=-0.5, label="27P3/2"),
    }
    levels.update(overrides)
    return ar.PrescribedLadderSpec(**levels)


def prescribed_request(**overrides):
    """A validated PRESCRIBED request wrapping :func:`prescribed_spec`."""
    return ar.AtomicResolutionRequest(
        mode=ar.AtomicResolutionMode.PRESCRIBED,
        prescribed=prescribed_spec(**overrides),
    )


# Frozen QN-001A reference (tests/test_qn001b_hfs_to_fs_contract.py): an
# INDEPENDENT oracle for the probe Rabi frequency, not produced by the code
# under test here.
FROZEN_PROBE_OMEGA_RAD_US = 15.758546351295182


# ---------------------------------------------------------------------------
# Isolation: the resolution switch and every cache the prescribed branch
# writes must be restored so this module cannot contaminate other tests.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_resolution_state():
    """Snapshot/restore the mode switch, the four caches and the identity.

    MODE-001: the initialized REQUEST is restored as well — a prescribed run
    inside one test must never steer another test's no-argument readers.
    """
    mode = ar.ATOMIC_RESOLUTION_MODE
    ladder = ar.PRESCRIBED_LADDER
    caches = {
        name: dict(getattr(acfg, name))
        for name in ("_ATOMIC_LEVELS_CACHE", "_DETUNING_CACHE",
                     "_DIPOLE_MATRIX_CACHE", "_OPTIMIZATION_RESULTS_CACHE")
    }
    identity = acfg._INITIALIZED_RESOLUTION_IDENTITY
    request = acfg._INITIALIZED_REQUEST
    yield
    ar.ATOMIC_RESOLUTION_MODE = mode
    ar.PRESCRIBED_LADDER = ladder
    for name, snapshot in caches.items():
        getattr(acfg, name).clear()
        getattr(acfg, name).update(snapshot)
    acfg._INITIALIZED_RESOLUTION_IDENTITY = identity
    acfg._INITIALIZED_REQUEST = request


@pytest.fixture
def published_state():
    """ARCH-001 published surface, restored after each runtime test."""
    state = snapshot_published_atomic_state()
    yield state
    restore_published_atomic_state(state)


# ===========================================================================
# 1. The single canonical switch and its backward-compatible default
# ===========================================================================
def test_default_mode_is_experimental():
    """MODE-001 backward compatibility: nothing opts into PRESCRIBED."""
    assert ar.DEFAULT_ATOMIC_RESOLUTION_MODE is ar.AtomicResolutionMode.EXPERIMENTAL
    assert ar.ATOMIC_RESOLUTION_MODE is ar.AtomicResolutionMode.EXPERIMENTAL


def test_production_has_no_default_prescribed_ladder():
    """Example mJ values belong to tests/docs only (never production config)."""
    assert ar.PRESCRIBED_LADDER is None, (
        "production must not ship a default declared ladder with example mJ"
    )


def test_resolve_without_argument_consults_the_switch():
    """initialize_atomic_physics() with no args keeps today's behaviour."""
    request = ar.resolve_resolution_request(None)
    assert request.mode is ar.AtomicResolutionMode.EXPERIMENTAL
    assert request.prescribed is None
    assert ar.resolution_request_identity(request) == ("experimental",)


def test_switch_round_trip_preserves_the_default():
    """set_* must restore exactly — no residue from a temporary switch."""
    ar.set_prescribed_ladder(prescribed_spec())
    assert ar.get_configured_resolution_request().is_prescribed
    ar.use_experimental_mode()
    assert ar.ATOMIC_RESOLUTION_MODE is ar.AtomicResolutionMode.EXPERIMENTAL
    assert ar.PRESCRIBED_LADDER is None


def test_set_resolution_mode_validates_before_mutating():
    """A rejected switch must NOT leave the process half-switched."""
    ar.set_resolution_mode(ar.AtomicResolutionMode.EXPERIMENTAL, None)
    with pytest.raises(PhysicsConfigurationError):
        ar.set_resolution_mode(ar.AtomicResolutionMode.PRESCRIBED, None)
    # unchanged: still EXPERIMENTAL with no ladder
    assert ar.ATOMIC_RESOLUTION_MODE is ar.AtomicResolutionMode.EXPERIMENTAL
    assert ar.PRESCRIBED_LADDER is None


def test_describe_resolution_request_names_both_branches():
    """Reporting must say WHICH branch resolved the ladder."""
    assert "EXPERIMENTAL" in ar.describe_resolution_request(
        ar.resolve_resolution_request(None))
    described = ar.describe_resolution_request(prescribed_request())
    assert "PRESCRIBED" in described and "26D5/2" in described


# ===========================================================================
# 2. Ambiguous requests fail loudly (F05)
# ===========================================================================
def test_experimental_request_must_not_carry_a_ladder():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        ar.AtomicResolutionRequest(
            mode=ar.AtomicResolutionMode.EXPERIMENTAL,
            prescribed=prescribed_spec())
    assert "ambiguous" in str(excinfo.value)


def test_prescribed_request_requires_a_ladder():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        ar.AtomicResolutionRequest(mode=ar.AtomicResolutionMode.PRESCRIBED)
    assert "PrescribedLadderSpec" in str(excinfo.value)


def test_bare_prescribed_mode_without_a_ladder_is_refused():
    """A bare PRESCRIBED string must never silently fall back to search."""
    with pytest.raises(PhysicsConfigurationError):
        ar.resolve_resolution_request("prescribed")


def test_unknown_mode_is_never_guessed():
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        ar.resolve_resolution_request("auto")
    assert "never guessed" in str(excinfo.value)


def test_unsupported_request_type_is_refused():
    with pytest.raises(PhysicsConfigurationError):
        ar.resolve_resolution_request(object())


# ===========================================================================
# 3. Declaration contract of a PRESCRIBED ladder
# ===========================================================================
def test_missing_rydberg_mj_fails_loudly():
    """Production PRESCRIBED without an explicit Rydberg mJ must FAIL.

    The mJ is a required user input: it is never inferred from mF, never
    defaulted to 0 and never scanned for a maximum (QN-001A / QN-001C).
    """
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(
            rydberg1=AtomicLevel(n=26, l=2, j=2.5, label="26D5/2 (no mJ)"))
    message = str(excinfo.value)
    assert "mJ" in message and "never inferred" in message


def test_both_rydberg_states_need_an_explicit_mj():
    """The second Rydberg state is checked as strictly as the first."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(
            rydberg2=AtomicLevel(n=27, l=1, j=1.5, label="27P3/2 (no mJ)"))
    assert "rydberg2" in str(excinfo.value)


def test_hyperfine_and_mj_may_not_be_declared_together():
    """Ambiguity is refused instead of being interpreted (QN-001C)."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(
            rydberg1=AtomicLevel(n=26, l=2, j=2.5, mj=-0.5, f=3.0, mf=0.0,
                                 label="ambiguous"))
    assert "ambiguous" in str(excinfo.value)


@pytest.mark.parametrize("role", ["ground", "excited"])
def test_probe_pair_must_be_hfs_resolved(role):
    """ground/excited need BOTH F and mF (mF is never taken from mJ)."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(**{
            role: AtomicLevel(n=5, l=0 if role == "ground" else 1,
                              j=0.5 if role == "ground" else 1.5,
                              label=f"{role} without HFS")})
    message = str(excinfo.value)
    assert "F" in message and "mF" in message
    assert "never inferred" in message


def test_j_zero_to_j_zero_is_forbidden():
    """j = 0 -> j = 0 has an identically vanishing E1 matrix element.

    The excited state is also declared synthetically (j = 1) so that the
    probe and coupling legs still satisfy E1 and the ``j = 0 -> j = 0`` rule
    is the one that actually fires on the RF leg.
    """
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(
            excited=AtomicLevel(n=5, l=1, j=1.0, f=2.0, mf=0.0,
                                label="5P1 (synthetic)"),
            rydberg1=AtomicLevel(n=26, l=2, j=0.0, mj=0.0,
                                 label="26D0 (synthetic)"),
            rydberg2=AtomicLevel(n=27, l=1, j=0.0, mj=0.0,
                                 label="27P0 (synthetic)"))
    assert "j = 0" in str(excinfo.value)


def test_quantum_numbers_are_never_repaired():
    """n and l must be physical integers with l <= n - 1."""
    with pytest.raises(PhysicsConfigurationError):
        prescribed_spec(rydberg1=AtomicLevel(
            n=26.5, l=2, j=2.5, mj=-0.5, label="non-integer n"))
    with pytest.raises(PhysicsConfigurationError):
        prescribed_spec(rydberg1=AtomicLevel(
            n=3, l=5, j=2.5, mj=-0.5, label="l exceeds n"))


def test_invalid_magnetic_projection_is_rejected_by_atomic_level():
    """|mJ| <= j and j - mJ must be an integer (QN-001A), so a half-integer
    j cannot carry an integer mJ."""
    with pytest.raises(PhysicsConfigurationError):
        AtomicLevel(n=26, l=2, j=2.5, mj=0.0,
                    label="integer mJ on half-integer j")


def test_declared_levels_are_not_mutated_by_declaration():
    """PRESCRIBED must not modify the caller's declaration (no label rewrite)."""
    bare = AtomicLevel(n=26, l=2, j=2.5, mj=-0.5, label="")
    spec = prescribed_spec(rydberg1=bare)
    assert bare.label == ""            # caller's object untouched
    assert spec.rydberg1.label != ""   # a canonical label exists for reporting
    assert spec.rydberg1 is not bare


@pytest.mark.parametrize(
    "rydberg1,rydberg2,why",
    [
        # delta_l = 0 (D -> D is not an E1 transition)
        (AtomicLevel(n=26, l=2, j=2.5, mj=-0.5, label="26D5/2"),
         AtomicLevel(n=27, l=2, j=2.5, mj=-0.5, label="27D5/2"), "delta_l"),
        # delta_j = 2 (j = 1/2 -> j = 5/2)
        (AtomicLevel(n=6, l=0, j=0.5, mj=-0.5, label="6S1/2"),
         AtomicLevel(n=6, l=1, j=2.5, mj=-0.5, label="6P5/2"), "delta_j"),
    ],
)
def test_e1_selection_rules_are_enforced_before_arc(rydberg1, rydberg2, why):
    """A structurally forbidden ladder can never be driven, so it is rejected
    before any ARC evaluation — and never repaired by a search."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        prescribed_spec(rydberg1=rydberg1, rydberg2=rydberg2)
    assert why in str(excinfo.value)


# ===========================================================================
# 4. Cache identity: no collision between modes or between ladders
# ===========================================================================
def test_experimental_cache_key_stays_bit_identical():
    """The default key must not move: existing entries stay valid."""
    key = acfg._resolution_cache_key()
    raw = acfg._generate_cache_key(LASER_PARAMS["probe"],
                                   LASER_PARAMS["coupling"], RF_PARAMS)
    assert key == raw and len(key) == 5


def test_prescribed_key_has_a_different_arity():
    """Different tuple arity => a prescribed entry can never collide."""
    prescribed_key = acfg._resolution_cache_key(prescribed_request())
    experimental_key = acfg._resolution_cache_key()
    assert len(prescribed_key) == len(experimental_key) + 15
    assert prescribed_key != experimental_key


def test_ladders_differing_only_in_rydberg_mj_get_different_keys():
    """Switching the declared Zeeman channel must not return stale data."""
    other = prescribed_request(
        rydberg1=AtomicLevel(n=26, l=2, j=2.5, mj=0.5, label="26D5/2"),
        rydberg2=AtomicLevel(n=27, l=1, j=1.5, mj=0.5, label="27P3/2"))
    assert (acfg._resolution_cache_key(prescribed_request())
            != acfg._resolution_cache_key(other))


def test_level_identity_is_never_none():
    """Regression: a misplaced ``return`` used to make level identities None
    so distinct ladders hashed identically."""
    for role, level in prescribed_spec().levels().items():
        identity = ar._level_identity(level)
        assert identity is not None, f"_level_identity({role}) returned None"
        assert len(identity) == 6


def test_request_identity_carries_all_four_states():
    identity = ar.resolution_request_identity(prescribed_request())
    assert identity[0] == "prescribed" and len(identity) == 5
    assert all(part is not None for part in identity[1:])


# ===========================================================================
# 5. Zero search: PRESCRIBED never touches a level/mJ/q search
# ===========================================================================
def test_prescribed_resolution_never_runs_a_level_search(monkeypatch):
    """The EXPERIMENTAL optimizer path must be unreachable in PRESCRIBED.

    Belt and braces: the branch entry point itself is replaced, plus the two
    legacy dipole scanners that would silently re-select a magnetic channel.
    """
    import rydberg_aoa.arc.dipole as dipole_mod
    import rydberg_aoa.arc.dipole_matrix as dipole_matrix_mod

    hits = {"optimizer": 0, "legacy_matrix": 0, "generic_dme": 0}

    def _banned(name):
        def _raise(*args, **kwargs):
            hits[name] += 1
            raise AssertionError(
                f"PRESCRIBED called {name}: a declared ladder must never be "
                "re-resolved by a search (MODE-001 section 9-11)")
        return _raise

    monkeypatch.setattr(acfg, "_initialize_atomic_levels",
                        _banned("optimizer"))
    monkeypatch.setattr(dipole_matrix_mod, "calculate_dipole_matrix",
                        _banned("legacy_matrix"))
    monkeypatch.setattr(dipole_mod, "calculate_dipole_matrix_element",
                        _banned("generic_dme"))

    levels = acfg._initialize_prescribed_levels(prescribed_request())

    assert hits == {"optimizer": 0, "legacy_matrix": 0, "generic_dme": 0}, (
        f"forbidden search invoked: {hits}")
    assert sorted(levels) == ["excited", "ground", "rydberg1", "rydberg2"]
    assert levels["rydberg1"]["mj"] == -0.5   # declared value, verbatim


def test_prescribed_dipoles_come_from_the_declared_channel_primitives(
        monkeypatch):
    """Positive control: the three QN primitives ARE the dipole source."""
    import rydberg_aoa.arc.dipole as dipole_mod

    calls = {name: 0 for name in (
        "calculate_hfs_to_hfs_dipole_matrix_element",
        "calculate_hfs_to_fs_dipole_matrix_element",
        "calculate_fs_to_fs_dipole_matrix_element",
    )}
    for name in calls:
        real = getattr(dipole_mod, name)

        def _wrap(real=real, name=name):
            def _spy(*args, **kwargs):
                calls[name] += 1
                return real(*args, **kwargs)
            return _spy
        monkeypatch.setattr(dipole_mod, name, _wrap())

    acfg._initialize_prescribed_levels(prescribed_request())
    assert all(count >= 1 for count in calls.values()), calls


def test_prescribed_resolution_never_rewrites_the_applied_fields():
    """Section 13: wavelengths/frequencies are INPUTS, never repaired."""
    before = copy.deepcopy(
        (LASER_PARAMS, RF_PARAMS, SIGNAL_RF_PARAMS))
    probe_before = LASER_PARAMS["probe"]["wavelength_nm"]
    coupling_before = LASER_PARAMS["coupling"]["wavelength_nm"]
    rf_before = (RF_PARAMS["frequency_hz"], RF_PARAMS["electric_field_vm"])

    acfg._initialize_prescribed_levels(prescribed_request())

    assert copy.deepcopy((LASER_PARAMS, RF_PARAMS, SIGNAL_RF_PARAMS)) == before
    assert LASER_PARAMS["probe"]["wavelength_nm"] == probe_before
    assert LASER_PARAMS["coupling"]["wavelength_nm"] == coupling_before
    assert (RF_PARAMS["frequency_hz"],
            RF_PARAMS["electric_field_vm"]) == rf_before


def test_prescribed_detuning_is_not_forced_to_resonance():
    """Section 13 corollary: the configured coupling wavelength (483.4 nm) is
    NOT the 5P -> 26D transition wavelength, so a prescribed Delta_c must stay
    large.  A near-zero value would prove a force-resonance mutation."""
    acfg._initialize_prescribed_levels(prescribed_request())
    key = acfg._resolution_cache_key(prescribed_request())
    delta_c = acfg._DETUNING_CACHE[key]["delta_c_rad_us"]
    assert np.isfinite(delta_c)
    assert abs(delta_c) > 1.0, (
        "Delta_c collapsed toward resonance: the applied coupling wavelength "
        "appears to have been rewritten (forbidden by MODE-001 section 13)")


def test_prescribed_detuning_cache_is_complete():
    acfg._initialize_prescribed_levels(prescribed_request())
    key = acfg._resolution_cache_key(prescribed_request())
    entry = acfg._DETUNING_CACHE[key]
    assert set(entry) == {"delta_p_rad_us", "delta_c_rad_us",
                          "delta_rf_rad_us"}
    assert all(np.isfinite(entry[name]) for name in entry)


# ===========================================================================
# 6. Detuning convention (section 14): Delta = omega_applied - omega_atomic
# ===========================================================================
def _arc_probe_detuning_rad_us():
    """INDEPENDENT oracle: ARC level energies -> Delta_p.

    Deliberately avoids the project's own ``physics.detuning`` helpers so the
    sign convention is checked against first principles rather than against
    itself.

        f_atomic  = (E_5P3/2 - E_5S1/2) / h
        f_applied = c / lambda_applied
        Delta_p   = 2*pi*(f_applied - f_atomic) * 1e-6      [rad/us]
    """
    h = 6.62607015e-34          # J s   (exact, SI 2019)
    e = 1.602176634e-19         # C     (exact, SI 2019)
    c = 299792458.0             # m/s   (exact, SI)

    energy_lower_ev = acfg.ATOM.getEnergy(5, 0, 0.5)   # 5S1/2
    energy_upper_ev = acfg.ATOM.getEnergy(5, 1, 1.5)   # 5P3/2
    f_atomic_hz = (energy_upper_ev - energy_lower_ev) * e / h

    wavelength_m = LASER_PARAMS["probe"]["wavelength_nm"] * 1e-9
    f_applied_hz = c / wavelength_m
    return 2.0 * np.pi * (f_applied_hz - f_atomic_hz) * 1e-6


def test_probe_detuning_sign_and_magnitude_match_an_arc_oracle():
    """Delta_p must be omega_applied - omega_atomic, in rad/us."""
    acfg._initialize_prescribed_levels(prescribed_request())
    key = acfg._resolution_cache_key(prescribed_request())
    delta_p = acfg._DETUNING_CACHE[key]["delta_p_rad_us"]

    expected = _arc_probe_detuning_rad_us()
    assert np.isfinite(expected)
    # The configured 780.241476 nm sits just off the D2 line, so the detuning
    # is small but NONZERO — a sign error would flip it and a missing 2*pi
    # would scale it by ~6.3.
    assert delta_p == pytest.approx(expected, rel=1e-6), (
        f"detuning convention mismatch: cached {delta_p!r} rad/us vs ARC "
        f"oracle {expected!r} rad/us (= omega_applied - omega_atomic)")


def test_detuning_units_are_rad_per_microsecond():
    """A Hz/rad-s confusion would show up as a 1e6 or 2*pi scale error."""
    acfg._initialize_prescribed_levels(prescribed_request())
    key = acfg._resolution_cache_key(prescribed_request())
    delta_p = acfg._DETUNING_CACHE[key]["delta_p_rad_us"]
    # frequency offset of the probe is of order kHz -> rad/us of order 1e-2..1e1
    assert 1e-4 < abs(delta_p) < 1e4


# ===========================================================================
# 7. LO / signal RF polarisation consistency (sections 19/20)
# ===========================================================================
def test_consistent_rf_polarisations_are_accepted():
    _require_lo_signal_rf_polarisation_consistency()   # both q = 0 by default


def test_inconsistent_rf_polarisations_fail_loudly(monkeypatch):
    """Both fields drive ONE R1 -> R2 coherence, so disagreeing q means they
    would beat on different magnetic sublevels.  Neither value is reconciled."""
    monkeypatch.setitem(SIGNAL_RF_PARAMS, "polarization", 1)
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        _require_lo_signal_rf_polarisation_consistency()
    message = str(excinfo.value)
    assert "DIFFERENT magnetic" in message
    # the declared values are reported, not rewritten
    assert SIGNAL_RF_PARAMS["polarization"] == 1
    assert RF_PARAMS["polarization"] == 0


def test_inconsistent_rf_polarisations_block_the_prescribed_resolution(
        monkeypatch):
    """The check runs before anything is computed or cached."""
    monkeypatch.setitem(SIGNAL_RF_PARAMS, "polarization", -1)
    with pytest.raises(PhysicsConfigurationError):
        acfg._initialize_prescribed_levels(prescribed_request())
    key = acfg._resolution_cache_key(prescribed_request())
    assert key not in acfg._ATOMIC_LEVELS_CACHE


def test_missing_rf_polarisation_is_not_defaulted():
    """q is a required field: no hard-coded 0 fallback (QN-001C / F05)."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        acfg._configured_q({"frequency_hz": 1.0}, "test RF")
    assert "polarisation" in str(excinfo.value)


# ===========================================================================
# 8. Full pipeline wiring (initialize_atomic_physics)
# ===========================================================================
def test_initialize_signature_stays_backward_compatible():
    """All arguments stay keyword-only with the historical defaults, so every
    existing ``initialize_atomic_physics()`` call keeps its meaning."""
    import inspect

    signature = inspect.signature(acfg.initialize_atomic_physics)
    assert set(signature.parameters) == {"force", "resolution_request"}
    assert signature.parameters["force"].default is False
    assert signature.parameters["resolution_request"].default is None
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in signature.parameters.values())


def test_full_prescribed_pipeline_runs_end_to_end(published_state):
    """PRESCRIBED resolves the ladder the EXPERIMENTAL branch cannot: it
    declares the Rydberg mJ, so the QN-001B/QN-001C boundaries are satisfied
    and the whole pipeline completes instead of failing fast."""
    request = prescribed_request()
    probe_before = LASER_PARAMS["probe"]["wavelength_nm"]
    coupling_before = LASER_PARAMS["coupling"]["wavelength_nm"]

    assert acfg.initialize_atomic_physics(
        force=True, resolution_request=request) is True

    # published runtime
    assert acfg.is_atomic_physics_initialized() is True
    assert acfg._INITIALIZED_RESOLUTION_IDENTITY == \
        ar.resolution_request_identity(request)
    assert acfg.ATOMIC_LEVELS["rydberg1"]["mj"] == -0.5
    assert acfg.ATOMIC_LEVELS["rydberg2"]["mj"] == -0.5

    # the applied fields survived (section 13)
    assert LASER_PARAMS["probe"]["wavelength_nm"] == probe_before
    assert LASER_PARAMS["coupling"]["wavelength_nm"] == coupling_before

    # the probe Rabi frequency matches the FROZEN QN-001A reference — an
    # oracle produced independently of this module's code path
    assert acfg.OMEGA_P == pytest.approx(FROZEN_PROBE_OMEGA_RAD_US,
                                         rel=1e-12)
    assert acfg.OMEGA_C > 0.0
    assert acfg.OMEGA_RF_BASE > 0.0
    assert acfg.OMEGA_SIGNAL_RF_BASE > 0.0
    assert all(gamma > 0.0
               for gamma in (acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43))

    # guarded readers accept the published state
    acfg.require_atomic_physics_initialized()
    matrix = acfg.get_cached_dipole_matrix()
    assert set(matrix) == {"probe", "coupling", "rf"}
    assert all(matrix[name]["dipole_si"] > 0.0 for name in matrix)
    # the matrix names the DECLARED transitions, verbatim
    assert matrix["rf"]["transition"] == "26D5/2 -> 27P3/2"

    # The published RF dipole must be bit-identical to the RAW ARC element of
    # the declared channel (mJ1 = -1/2 -> mJ2 = -1/2 with the configured q = 0)
    # — no scan, no max, no unit round trip.  ARC's own call is the oracle.
    raw_ea0 = acfg.ATOM.getDipoleMatrixElement(
        26, 2, 2.5, -0.5, 27, 1, 1.5, -0.5, 0)
    # ARC_RF_DIPOLE_EA0 is a magnitude (as the Experimental branch publishes);
    # the SIGNED element is kept verbatim in the matrix entry.
    assert acfg.ARC_RF_DIPOLE_EA0 == abs(raw_ea0)
    assert matrix["rf"]["dipole_ea0"] == raw_ea0

    # MODE-001: the stored initialized request is the request that was run,
    # and it survives a full cache wipe — every no-argument reader must still
    # find the PRESCRIBED entries (and the legacy search path must never be
    # reached on self-heal).
    assert acfg._INITIALIZED_REQUEST is not None
    stored_request, from_switch = acfg._initialized_resolution_request()
    assert from_switch is False
    assert stored_request.is_prescribed
    assert (ar.resolution_request_identity(stored_request)
            == ar.resolution_request_identity(request))
    detunings = acfg.get_cached_detunings()
    assert detunings == (acfg.DELTA_P, acfg.DELTA_C, acfg.DELTA_RF)
    assert acfg.get_arc_calculated_rf_dipole() == acfg.ARC_RF_DIPOLE_EA0
    acfg._ATOMIC_LEVELS_CACHE.clear()
    acfg._DETUNING_CACHE.clear()
    acfg._DIPOLE_MATRIX_CACHE.clear()
    acfg._OPTIMIZATION_RESULTS_CACHE.clear()
    healed = acfg.get_cached_dipole_matrix()
    assert healed is not None
    assert healed["rf"]["transition"] == "26D5/2 -> 27P3/2"
    assert healed["rf"]["dipole_ea0"] == raw_ea0
    assert healed["rf"]["dipole_si"] == matrix["rf"]["dipole_si"]


def test_prescribed_initialization_is_idempotent_for_the_same_request(
        published_state):
    request = prescribed_request()
    assert acfg.initialize_atomic_physics(
        force=True, resolution_request=request) is True
    assert acfg.initialize_atomic_physics(
        resolution_request=request) is True   # no recompute, no error


def test_changing_resolution_after_initialization_fails_loudly(
        published_state):
    """Handing an EXPERIMENTAL result to a PRESCRIBED caller (or another
    declared ladder's numbers to this one) would be silent wrong physics."""
    assert acfg.initialize_atomic_physics(
        force=True, resolution_request=prescribed_request()) is True

    with pytest.raises(PhysicsConfigurationError) as excinfo:
        acfg.initialize_atomic_physics()   # default = EXPERIMENTAL
    assert "force=True" in str(excinfo.value)

    other = prescribed_request(
        rydberg1=AtomicLevel(n=26, l=2, j=2.5, mj=0.5, label="26D5/2"),
        rydberg2=AtomicLevel(n=27, l=1, j=1.5, mj=0.5, label="27P3/2"))
    with pytest.raises(PhysicsConfigurationError):
        acfg.initialize_atomic_physics(resolution_request=other)


def test_experiment_default_is_still_what_a_bare_call_asks_for():
    """A bare call must ask for EXPERIMENTAL — it never opts into PRESCRIBED."""
    request = ar.resolve_resolution_request(None)
    assert request.mode is ar.AtomicResolutionMode.EXPERIMENTAL
    assert request.prescribed is None
    assert ar.resolution_request_identity(request) == ("experimental",)
