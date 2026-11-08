"""ARCH-001TX — a FAILED ``initialize_atomic_physics()`` must leave NO state.

ARCH-001 made the seam transactional for the PUBLISHED surface (compatibility
globals + initialized flag are written only on success).  This module pins the
ARCH-001TX extension: the CONFIGURATION state that pipeline steps rewrite
*before* any later failure can occur is rolled back as well:

* ``LASER_PARAMS`` — the Experimental branch rewrites
  ``LASER_PARAMS['coupling']['wavelength_nm']`` at level step 1
  (historically leaking ``483.4 -> 483.238...`` nm when the open QN-001A
  coupling boundary then aborted the run at step 4);
* ``params.SI_Constants.LAMBDA_P`` — rewritten from ``c / f_21``;
* ``params.LEVEL_SELECTION_TOLERANCES`` — preserved on failure; LAB-001
  removed the historical 10x widening and automatic retry;
* ``params.OMEGA_S_BASE`` / ``params.OMEGA_LO_BASE`` — rewritten near the
  end of step 4 while step 5 can still fail;
* the four pipeline caches (level / detuning / dipole / optimization).

Contracts preserved (never weakened by this rollback):

* the QN-001A missing-mJ boundary still raises ``PhysicsConfigurationError``
  naming F05, with type, message and exception chain untouched (F05);
* a failed ``force=True`` attempt still leaves the runtime UNPUBLISHED —
  ``_clear_published_atomic_state()`` ran on purpose and is not undone;
* the PRESCRIBED success path is unchanged (publish-on-success, section 13
  wavelength immutability);
* nothing here mocks away the real boundary: the ``real seam`` test runs
  the production Experimental pipeline until the documented QN-001A
  fail-fast and verifies the rollback on that real path.
* a reference held BEFORE the run (``laser_ref`` / ``probe_ref`` /
  ``coupling_ref``) stays valid afterwards: same objects, restored values
  (ARCH-001TX-R1) — asserted on the real seam and on injected
  in-place/replacement/removal mutations through the seam.

Isolation: every test snapshots the full configuration surface (including
``_INITIALIZED_RESOLUTION_IDENTITY``, which the shared published-state helper
does not capture) and restores it afterwards, so collection order cannot
matter in either direction.  The teardown reinstalls the ORIGINAL nested
member dictionaries (``LASER_PARAMS['probe' | 'coupling']``) and restores
their contents in place — a teardown using the pre-R1 clear()/update() with
deep-copied replacements would orphan pre-run references and could conceal
the defect this module pins or contaminate later tests.
"""

import copy

import numpy as np
import pytest

from rydberg_aoa.arc.types import AtomicLevel
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import atomic_resolution as ar
from rydberg_aoa.config import laser_rf_config as lrf
from rydberg_aoa.config import parameters as params
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)
from tests.atomic_init_support import (
    PUBLISHED_ATOMIC_GLOBALS,
    is_documented_qn001a_coupling_block,
    restore_published_atomic_state,
    snapshot_published_atomic_state,
)

_CACHE_NAMES = (
    "_ATOMIC_LEVELS_CACHE",
    "_DETUNING_CACHE",
    "_DIPOLE_MATRIX_CACHE",
    "_OPTIMIZATION_RESULTS_CACHE",
)


# ---------------------------------------------------------------------------
# The declared ladder used by the prescribed steps (test fixture, never
# production configuration — see tests/test_mode001_atomic_resolution_boundary.py)
# ---------------------------------------------------------------------------
def _prescribed_request():
    """A validated PRESCRIBED request (q = 0, explicit Rydberg mJ)."""
    levels = {
        "ground": AtomicLevel(n=5, l=0, j=0.5, f=3.0, mf=0.0,
                              label="5S1/2 F=3"),
        "excited": AtomicLevel(n=5, l=1, j=1.5, f=4.0, mf=0.0,
                               label="5P3/2 F=4"),
        "rydberg1": AtomicLevel(n=26, l=2, j=2.5, mj=-0.5, label="26D5/2"),
        "rydberg2": AtomicLevel(n=27, l=1, j=1.5, mj=-0.5, label="27P3/2"),
    }
    return ar.AtomicResolutionRequest(
        mode=ar.AtomicResolutionMode.PRESCRIBED,
        prescribed=ar.PrescribedLadderSpec(**levels),
    )


def _snapshot_configuration():
    """Everything ARCH-001TX rolls back, plus the full published surface.

    ``laser_members`` keeps the ORIGINAL probe/coupling member objects so the
    teardown restore reinstalls them instead of deep-copied replacements
    (ARCH-001TX-R1: test cleanup must preserve nested references too).
    """
    return {
        "laser_params": copy.deepcopy(lrf.LASER_PARAMS),
        "laser_members": dict(lrf.LASER_PARAMS),
        "rf_params": copy.deepcopy(lrf.RF_PARAMS),
        "lo_rf_params": copy.deepcopy(lrf.LO_RF_PARAMS),
        "signal_rf_params": copy.deepcopy(lrf.SIGNAL_RF_PARAMS),
        "lambda_p": params.SI_Constants.LAMBDA_P,
        "tolerances": dict(params.LEVEL_SELECTION_TOLERANCES),
        "omega_s_base": params.OMEGA_S_BASE,
        "omega_lo_base": params.OMEGA_LO_BASE,
        "caches": {name: dict(getattr(acfg, name)) for name in _CACHE_NAMES},
        "initialized_identity": acfg._INITIALIZED_RESOLUTION_IDENTITY,
        "published": snapshot_published_atomic_state(),
    }


def _restore_configuration(snap):
    """In-place restore (same discipline as the production rollback).

    ARCH-001TX-R1: the nested member dictionaries are reinstalled as their
    ORIGINAL objects and their contents are restored in place.  A
    ``clear()``/``update()`` with deep-copied replacements here (the pre-R1
    pattern) would orphan any reference retained before the run and could
    conceal the very defect this module pins — or contaminate later tests
    with objects that no earlier reference observes.
    """
    # remove keys a test may have added beyond the entry state
    for key in [key for key in lrf.LASER_PARAMS
                if key not in snap["laser_params"]]:
        del lrf.LASER_PARAMS[key]
    # reinstall the ORIGINAL probe/coupling objects (undoing any
    # replacement), then restore their contents IN PLACE (undoing any
    # in-place rewrite or nested-key removal)
    for key, saved in snap["laser_params"].items():
        original = snap["laser_members"][key]
        if isinstance(original, dict):
            lrf.LASER_PARAMS[key] = original
            for member in [member for member in original
                           if member not in saved]:
                del original[member]
            original.update(saved)
        else:
            lrf.LASER_PARAMS[key] = saved
    lrf.RF_PARAMS.clear()
    lrf.RF_PARAMS.update(snap["rf_params"])
    lrf.LO_RF_PARAMS.clear()
    lrf.LO_RF_PARAMS.update(snap["lo_rf_params"])
    lrf.SIGNAL_RF_PARAMS.clear()
    lrf.SIGNAL_RF_PARAMS.update(snap["signal_rf_params"])
    params.SI_Constants.LAMBDA_P = snap["lambda_p"]
    params.LEVEL_SELECTION_TOLERANCES.clear()
    params.LEVEL_SELECTION_TOLERANCES.update(snap["tolerances"])
    params.OMEGA_S_BASE = snap["omega_s_base"]
    params.OMEGA_LO_BASE = snap["omega_lo_base"]
    for name in _CACHE_NAMES:
        getattr(acfg, name).clear()
        getattr(acfg, name).update(snap["caches"][name])
    acfg._INITIALIZED_RESOLUTION_IDENTITY = snap["initialized_identity"]
    restore_published_atomic_state(snap["published"])


@pytest.fixture
def configuration_state():
    """Snapshot the whole ARCH-001TX surface; restore it after the test."""
    snap = _snapshot_configuration()
    yield snap
    _restore_configuration(snap)


def _force_uninitialized(monkeypatch):
    """Force the pre-init runtime (flag False, nothing published)."""
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", False)
    for name in PUBLISHED_ATOMIC_GLOBALS:
        monkeypatch.setattr(acfg, name, None)
    monkeypatch.setattr(acfg, "_INITIALIZED_REQUEST", None)
    monkeypatch.setattr(acfg, "_INITIALIZED_RESOLUTION_IDENTITY", None)
    monkeypatch.setattr(acfg, "_RF_ORIENTATION", None)


def _assert_nothing_published():
    assert acfg.is_atomic_physics_initialized() is False, (
        "a failed initialization flipped the initialized flag")
    for name in PUBLISHED_ATOMIC_GLOBALS:
        assert getattr(acfg, name) is None, (
            f"{name} was published despite a failed initialization")
    assert acfg._INITIALIZED_REQUEST is None, (
        "a resolution request survived a failed initialization")
    assert acfg._INITIALIZED_RESOLUTION_IDENTITY is None, (
        "a resolution identity survived a failed initialization")
    assert acfg._RF_ORIENTATION is None, (
        "an RF orientation survived a failed initialization")


# ===========================================================================
# 1. REAL seam: the documented QN-001A boundary, full rollback, reproducer
# ===========================================================================
def test_failed_experimental_seam_rolls_back_and_prescribed_detuning_survives(
        configuration_state, monkeypatch):
    """The original defect (ARCH-001TX reproducer), on the REAL pipeline.

    The production Experimental pipeline runs until the documented QN-001A
    coupling fail-fast — but only AFTER level step 1 has rewritten the
    coupling wavelength, ``SI_Constants.LAMBDA_P`` and the caches.  Before
    ARCH-001TX those writes leaked, so a later PRESCRIBED resolution
    computed ``Delta_c = 0.0`` from the rewritten (resonant) wavelength.
    Now:

    * the boundary still raises the exact ``PhysicsConfigurationError``
      naming F05 (no error translation);
    * every configuration mutation is rolled back;
    * the subsequent PRESCRIBED detuning stays large.
    """
    snap = configuration_state
    _force_uninitialized(monkeypatch)
    # ARCH-001TX-R1: retain the references BEFORE the run; every one of them
    # must observe the rollback afterwards.
    laser_ref = lrf.LASER_PARAMS
    probe_ref = lrf.LASER_PARAMS["probe"]
    coupling_ref = lrf.LASER_PARAMS["coupling"]
    probe_entry = dict(probe_ref)
    coupling_before = lrf.LASER_PARAMS["coupling"]["wavelength_nm"]
    lambda_p_before = params.SI_Constants.LAMBDA_P
    tolerances_before = dict(params.LEVEL_SELECTION_TOLERANCES)

    # --- the real seam, real ARC optimizer, real QN-001A fail-fast ---------
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        acfg.initialize_atomic_physics(
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    assert is_documented_qn001a_coupling_block(excinfo.value), (
        "the seam did not stop at the documented QN-001A HFS->FS boundary: "
        f"{excinfo.value}")
    assert "F05" in str(excinfo.value)

    # --- nothing was published (ARCH-001 contract, unchanged) --------------
    _assert_nothing_published()

    # --- every configuration mutation was rolled back (ARCH-001TX) ---------
    assert lrf.LASER_PARAMS == snap["laser_params"], (
        "LASER_PARAMS leaked the pre-boundary coupling-wavelength rewrite")
    assert lrf.LASER_PARAMS["coupling"]["wavelength_nm"] == coupling_before
    # ARCH-001TX-R1: the same rollback, seen through the RETAINED refs ------
    assert lrf.LASER_PARAMS is laser_ref, (
        "the rollback orphaned the pre-run LASER_PARAMS reference")
    assert acfg.LASER_PARAMS is laser_ref, (
        "atomic_config.LASER_PARAMS diverged from laser_rf_config.LASER_PARAMS")
    assert lrf.LASER_PARAMS["probe"] is probe_ref, (
        "the rollback replaced LASER_PARAMS['probe'] with a deep copy")
    assert lrf.LASER_PARAMS["coupling"] is coupling_ref, (
        "the rollback replaced LASER_PARAMS['coupling'] with a deep copy")
    assert dict(probe_ref) == probe_entry, (
        "the retained probe reference does not show the restored values")
    assert coupling_ref["wavelength_nm"] == coupling_before, (
        "the retained coupling reference still shows the rewritten "
        f"wavelength ({coupling_ref['wavelength_nm']!r})")
    assert params.SI_Constants.LAMBDA_P == lambda_p_before, (
        "SI_Constants.LAMBDA_P leaked the dynamic probe-wavelength rewrite")
    assert dict(params.LEVEL_SELECTION_TOLERANCES) == tolerances_before, (
        "LEVEL_SELECTION_TOLERANCES leaked a fallback widening")
    assert params.OMEGA_S_BASE == snap["omega_s_base"]
    assert params.OMEGA_LO_BASE == snap["omega_lo_base"]
    for name in _CACHE_NAMES:
        assert getattr(acfg, name) == snap["caches"][name], (
            f"{name} leaked entries from the failed attempt")

    # --- the reproducer: PRESCRIBED Delta_c cannot collapse to 0.0 ---------
    request = _prescribed_request()
    acfg._initialize_prescribed_levels(request)
    key = acfg._resolution_cache_key(request)
    delta_c = acfg._DETUNING_CACHE[key]["delta_c_rad_us"]
    assert np.isfinite(delta_c)
    assert abs(delta_c) > 1.0, (
        "Delta_c collapsed toward resonance after a failed EXPERIMENTAL "
        "attempt — the applied coupling wavelength leaked "
        "(MODE-001 section 13 / ARCH-001TX)")


# ===========================================================================
# 1b. Injected nested mutations (in-place, replacement, removal) roll back
#     through every retained reference (ARCH-001TX-R1)
# ===========================================================================
def test_injected_nested_mutation_rolls_back_through_retained_references(
        configuration_state, monkeypatch):
    """A step-1 failure after every KIND of nested misuse — in-place rewrite,
    nested-key removal, whole-member replacement and a new root key — must
    restore the ORIGINAL probe/coupling objects with their ORIGINAL contents,
    observable through references retained before the run.
    """
    _force_uninitialized(monkeypatch)

    laser_ref = lrf.LASER_PARAMS
    probe_ref = lrf.LASER_PARAMS["probe"]
    coupling_ref = lrf.LASER_PARAMS["coupling"]
    entry_laser = copy.deepcopy(lrf.LASER_PARAMS)
    entry_probe = dict(probe_ref)
    entry_coupling = dict(coupling_ref)

    def _mutating_step_one():
        # every mutation shape the real pipeline (or a future refactor) may
        # apply to a configuration mapping:
        lrf.LASER_PARAMS["coupling"]["wavelength_nm"] = 999.9  # in place
        del lrf.LASER_PARAMS["probe"]["power_mw"]              # nested removal
        lrf.LASER_PARAMS["coupling"] = {"wavelength_nm": 111.0}  # replacement
        lrf.LASER_PARAMS["arch001tx_extra"] = {"x": 1}         # new root key
        raise RuntimeError("ARCH001TX-R1 sentinel step-1 failure")

    monkeypatch.setattr(acfg, "_initialize_atomic_levels", _mutating_step_one)

    with pytest.raises(RuntimeError, match="ARCH001TX-R1 sentinel step-1"):
        acfg.initialize_atomic_physics(
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    # identity: the ORIGINAL member objects are back under their keys -------
    assert lrf.LASER_PARAMS is laser_ref
    assert lrf.LASER_PARAMS["probe"] is probe_ref, (
        "a probe member mutation was not restored in the original object")
    assert lrf.LASER_PARAMS["coupling"] is coupling_ref, (
        "a replaced coupling member was not reinstalled as the original object")
    # values: restored INSIDE the retained objects ----------------------------
    assert dict(probe_ref) == entry_probe, (
        "the retained probe reference does not show the restored contents")
    assert dict(coupling_ref) == entry_coupling, (
        "the retained coupling reference does not show the restored contents")
    assert "power_mw" in probe_ref, (
        "the removed nested key was not restored in the retained probe object")
    assert "arch001tx_extra" not in lrf.LASER_PARAMS, (
        "a root key added by the failed run survived the rollback")
    assert lrf.LASER_PARAMS == entry_laser
    _assert_nothing_published()


# ===========================================================================
# 2. Fake seam: a failure AFTER step 4 rewrote the heterodyne bases
# ===========================================================================
def test_failure_after_step4_rolls_back_heterodyne_bases(
        configuration_state, monkeypatch):
    """Step 4 writes ``params.OMEGA_S_BASE``/``OMEGA_LO_BASE``; step 5 can
    still fail afterwards.  Those writes must roll back too."""
    import rydberg_aoa.arc.decay_rates as decay_rates_module
    from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation

    _force_uninitialized(monkeypatch)
    # Known sentinels: if rollback never runs, the step-4 values (8.0/6.0)
    # survive and this test fails instead of passing vacuously.
    monkeypatch.setattr(params, "OMEGA_S_BASE", -1.0)
    monkeypatch.setattr(params, "OMEGA_LO_BASE", -2.0)

    fake_levels = {
        "ground": {"n": 5, "l": 0, "j": 0.5},
        "excited": {"n": 5, "l": 1, "j": 1.5},
        "rydberg1": {"n": 26, "l": 2, "j": 2.5},
        "rydberg2": {"n": 27, "l": 1, "j": 1.5},
    }
    monkeypatch.setattr(acfg, "_initialize_atomic_levels",
                        lambda: fake_levels)
    monkeypatch.setattr(acfg, "_resolve_arc_rf_dipole",
                        lambda *args, **kwargs: 1.0e-27)
    monkeypatch.setattr(acfg, "ATOM", object())  # non-None: ARC branch taken
    monkeypatch.setattr(decay_rates_module, "calculate_decay_rates",
                        lambda *args, **kwargs: (2.0, 3.0, 4.0))
    monkeypatch.setattr(
        decay_rates_module, "resolve_rydberg_orientation",
        lambda *args, **kwargs: resolve_rf_orientation(-1.0))
    # Rabi helpers return rad/s; the seam divides by US_TO_S_FACTOR.
    monkeypatch.setattr(acfg, "calculate_laser_rabi_frequency",
                        lambda *args, **kwargs: 4.0 * acfg.US_TO_S_FACTOR)
    monkeypatch.setattr(acfg, "calculate_rf_rabi_frequency",
                        lambda *args, **kwargs: 6.0 * acfg.US_TO_S_FACTOR)
    monkeypatch.setattr(acfg, "get_signal_rf_rabi_frequency",
                        lambda *args, **kwargs: 8.0 * acfg.US_TO_S_FACTOR)

    def _detunings_boom(*args, **kwargs):
        raise PhysicsCalculationError(
            "ARCH001TX sentinel step-5 detuning failure")

    monkeypatch.setattr(acfg, "_resolve_cached_detunings", _detunings_boom)

    with pytest.raises(PhysicsCalculationError,
                       match="ARCH001TX sentinel step-5"):
        acfg.initialize_atomic_physics(
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    assert params.OMEGA_S_BASE == -1.0, (
        "step-4 OMEGA_S_BASE write survived the failed transaction "
        f"(got {params.OMEGA_S_BASE!r})")
    assert params.OMEGA_LO_BASE == -2.0, (
        "step-4 OMEGA_LO_BASE write survived the failed transaction "
        f"(got {params.OMEGA_LO_BASE!r})")
    _assert_nothing_published()


# ===========================================================================
# 3. LAB-001: an empty result must not retry or widen tolerances
# ===========================================================================
def test_empty_selection_fails_without_retry_or_tolerance_widening(
        configuration_state, monkeypatch):
    """A failed selection leaves the configured domain and runtime intact."""
    import rydberg_aoa.level_finder as level_finder

    _force_uninitialized(monkeypatch)
    monkeypatch.setitem(params.LEVEL_SELECTION_TOLERANCES,
                        "frequency_tolerance_hz", 12345.0)

    attempts = []
    class _FlakyOptimizer:
        """Empty primary result; any second call is a contract violation."""
        def __init__(self, arc_atom=None):
            self.calls = 0

        def optimize(self, *args, **kwargs):
            self.calls += 1
            attempts.append(params.LEVEL_SELECTION_TOLERANCES['frequency_tolerance_hz'])
            if self.calls == 1:
                return None
            raise RuntimeError("ARCH001TX sentinel fallback failure")

    monkeypatch.setattr(level_finder, "SingleStageLevelOptimizer",
                        _FlakyOptimizer)

    with pytest.raises(RuntimeError,
                       match="Cannot determine atomic levels"):
        acfg.initialize_atomic_physics(
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    assert attempts == [12345.0], "selection attempted an automatic retry"
    assert params.LEVEL_SELECTION_TOLERANCES["frequency_tolerance_hz"] == \
        12345.0, (
            "configured frequency tolerance changed after failed selection")
    assert lrf.LASER_PARAMS == configuration_state["laser_params"], (
        "LASER_PARAMS changed during the failed fallback attempt")
    _assert_nothing_published()


# ===========================================================================
# 4. Exception identity and chaining survive the rollback (F05)
# ===========================================================================
def test_exception_identity_and_cause_survive_rollback(
        configuration_state, monkeypatch):
    """Rollback must re-raise the ORIGINAL object — no wrapping, no
    translation, ``__cause__`` intact."""
    _force_uninitialized(monkeypatch)
    monkeypatch.setattr(acfg, "_initialize_atomic_levels", lambda: {
        "ground": {"n": 5, "l": 0, "j": 0.5},
        "excited": {"n": 5, "l": 1, "j": 1.5},
        "rydberg1": {"n": 26, "l": 2, "j": 2.5},
        "rydberg2": {"n": 27, "l": 1, "j": 1.5},
    })

    sentinel_cause = ValueError("ARCH001TX sentinel root cause")
    sentinel = PhysicsCalculationError(
        "ARCH001TX sentinel chained failure")

    def _boom(*args, **kwargs):
        raise sentinel from sentinel_cause

    monkeypatch.setattr(acfg, "_resolve_arc_rf_dipole", _boom)

    with pytest.raises(PhysicsCalculationError) as excinfo:
        acfg.initialize_atomic_physics(
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    assert excinfo.value is sentinel, (
        "the rollback re-raised a different exception object")
    assert excinfo.value.__cause__ is sentinel_cause, (
        "the exception chain was broken by the rollback")
    assert "ARCH001TX sentinel chained failure" in str(excinfo.value)
    _assert_nothing_published()


# ===========================================================================
# 5. force=True semantics: caches restored, previous runtime NOT re-advertised
# ===========================================================================
def test_failed_forced_rerun_restores_caches_and_stays_unpublished(
        configuration_state, monkeypatch):
    """Establish a real PRESCRIBED success, then fail a forced EXPERIMENTAL
    re-run at step 1.

    * force contract (ARCH-001/MODE-001, unchanged): the previous runtime
      was cleared on purpose and must NOT be re-advertised after the failure;
    * ARCH-001TX: the caches/configuration are restored to the PRE-ATTEMPT
      (post-success) state instead of keeping whatever the failed run left.
    """
    request = _prescribed_request()
    assert acfg.initialize_atomic_physics(
        force=True, resolution_request=request) is True
    assert acfg.is_atomic_physics_initialized() is True

    caches_after_success = {
        name: dict(getattr(acfg, name)) for name in _CACHE_NAMES}
    laser_after_success = copy.deepcopy(lrf.LASER_PARAMS)
    lambda_p_after_success = params.SI_Constants.LAMBDA_P
    omega_s_after_success = params.OMEGA_S_BASE
    omega_lo_after_success = params.OMEGA_LO_BASE

    def _boom():
        raise RuntimeError("ARCH001TX sentinel forced failure")

    monkeypatch.setattr(acfg, "_initialize_atomic_levels", _boom)
    with pytest.raises(RuntimeError, match="ARCH001TX sentinel forced"):
        acfg.initialize_atomic_physics(
            force=True,
            resolution_request=ar.AtomicResolutionMode.EXPERIMENTAL)

    # force contract: the previous runtime is NOT re-advertised
    _assert_nothing_published()

    # ARCH-001TX: state restored to the pre-attempt (post-success) snapshot
    for name in _CACHE_NAMES:
        assert getattr(acfg, name) == caches_after_success[name], (
            f"{name} does not match the pre-attempt state after a failed "
            "forced re-run")
    assert lrf.LASER_PARAMS == laser_after_success
    assert params.SI_Constants.LAMBDA_P == lambda_p_after_success
    assert params.OMEGA_S_BASE == omega_s_after_success
    assert params.OMEGA_LO_BASE == omega_lo_after_success
