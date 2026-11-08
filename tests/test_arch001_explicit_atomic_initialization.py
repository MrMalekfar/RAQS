"""ARCH-001 — explicit atomic-physics initialization seam.

Historical defect: level optimization, ARC dipoles, decay rates, Rabi
frequencies and detunings were computed at *import time* of
``rydberg_aoa.config.atomic_config``.  Merely importing a production module
(e.g. ``rydberg_aoa.simulation.runner``) therefore started the whole ARC
ladder pipeline, and with the QN-001A coupling contract open the import
itself failed with ``PhysicsConfigurationError`` — surfacing a physics
configuration problem as an import side effect.

ARCH-001 contract verified here (production code in
``rydberg_aoa/config/atomic_config.py``):

A. Importing ``atomic_config`` (or ``main``) performs NO level optimization
   and publishes NO resolved physics (``ATOMIC_LEVELS`` stays ``None``).
B. Importing the runner / physics / visualization modules does not flip the
   initialization flag.
C. ``is_atomic_physics_initialized()`` / ``require_atomic_physics_initialized()``
   are honest state readers; the guard never auto-initializes.
D. Every guarded reader fails loudly (``PhysicsConfigurationError`` naming
   ``initialize_atomic_physics``) while uninitialized — no defaults, no
   implicit init.
E. ``initialize_atomic_physics()`` is TRANSACTIONAL: a failure at any step
   leaves the runtime exactly as uninitialized as it started (nothing
   partially published).
F. ``initialize_atomic_physics()`` is IDEMPOTENT; a second call does not
   re-run the ARC pipeline unless ``force=True``.
G. ``plot_energy_levels`` / ``calculate_transmission_from_rho21`` read the
   atomic state through the module object (no stale ``from ... import
   ATOMIC_LEVELS`` snapshot that could silently go out of date).
H. Public main delegates to the explicit profile CLI; the legacy
   programmatic CLI initializes before constructing its runner.

State hygiene: every test in this module snapshots the published atomic
state (flag + compatibility globals + ``params.OMEGA_*_BASE``) and restores
it afterwards, so no test can leak mocked physics into the rest of the
session regardless of collection order.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import parameters as params
from rydberg_aoa.exceptions import PhysicsConfigurationError

REPO_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Published compatibility state that initialize_atomic_physics() may write.
_PUBLISHED_GLOBALS = (
    "ATOMIC_LEVELS",
    "ARC_RF_DIPOLE_EA0",
    "GAMMA_21",
    "GAMMA_32",
    "GAMMA_43",
    "OMEGA_P",
    "OMEGA_C",
    "OMEGA_RF_BASE",
    "OMEGA_SIGNAL_RF_BASE",
    "DELTA_P",
    "DELTA_C",
    "DELTA_RF",
)


def _snapshot():
    """Capture every piece of published atomic state (plus heterodyne bases)."""
    return {
        "flag": acfg._ATOMIC_PHYSICS_INITIALIZED,
        "globals": {name: getattr(acfg, name) for name in _PUBLISHED_GLOBALS},
        # MODE-001: the stored initialized request is published state too.
        "initialized_request": acfg._INITIALIZED_REQUEST,
        # RF-DIR-001B: the resolved RF-edge orientation is published state too.
        "rf_orientation": acfg._RF_ORIENTATION,
        "omega_s_base": params.OMEGA_S_BASE,
        "omega_lo_base": params.OMEGA_LO_BASE,
    }


def _restore(state):
    """Restore a snapshot taken by :func:`_snapshot` (exact, field by field)."""
    for name, value in state["globals"].items():
        setattr(acfg, name, value)
    acfg._ATOMIC_PHYSICS_INITIALIZED = state["flag"]
    acfg._INITIALIZED_REQUEST = state["initialized_request"]
    acfg._RF_ORIENTATION = state["rf_orientation"]
    params.OMEGA_S_BASE = state["omega_s_base"]
    params.OMEGA_LO_BASE = state["omega_lo_base"]


@pytest.fixture(autouse=True)
def _atomic_state_isolation():
    """Snapshot published atomic state before and restore it after each test."""
    state = _snapshot()
    yield
    _restore(state)


def _force_uninitialized(monkeypatch):
    """Force the pre-init runtime (flag False, no published physics)."""
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", False)
    for name in _PUBLISHED_GLOBALS:
        monkeypatch.setattr(acfg, name, None)
    # MODE-001: no stored request may survive into the forced pre-init state.
    monkeypatch.setattr(acfg, "_INITIALIZED_REQUEST", None)
    # RF-DIR-001B: nor may a published RF-edge orientation.
    monkeypatch.setattr(acfg, "_RF_ORIENTATION", None)


# ---------------------------------------------------------------------------
# A. Import is side-effect free (subprocess: fresh interpreter, real code)
# ---------------------------------------------------------------------------
def test_import_atomic_config_and_main_perform_no_physics(tmp_path):
    """A fresh ``import atomic_config`` / ``import main`` runs no ARC
    optimization and publishes nothing — the ARCH-001 headline invariant.

    Runs in a subprocess so it observes the *real* import path exactly as a
    production process would, unaffected by this session's module state.
    """
    marker = tmp_path / "marker.txt"
    script = textwrap.dedent(
        """
        import os

        import rydberg_aoa.config.atomic_config as acfg

        first = (acfg.is_atomic_physics_initialized(), acfg.ATOMIC_LEVELS)

        import main  # production entry point; must not run the pipeline either

        second = (acfg.is_atomic_physics_initialized(), acfg.ATOMIC_LEVELS)

        with open(os.environ["ARCH001_MARKER"], "w") as fh:
            fh.write(repr((first, second)))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["ARCH001_MARKER"] = str(marker)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # A pure import must not create a log or initialize physics.
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "importing atomic_config / main must succeed without running physics\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # No level-optimization / Rabi output may be produced by a plain import.
    combined = result.stdout + result.stderr
    assert "Optimizing atomic levels" not in combined, (
        "import triggered the ARC level optimization (import-time physics)"
    )
    assert "ARC Rabi Frequency Calculations" not in combined, (
        "import triggered the ARC Rabi pipeline (import-time physics)"
    )
    assert marker.exists(), "subprocess did not reach the marker write"
    first, second = eval(marker.read_text())  # noqa: S307 - self-generated repr
    assert first == (False, None), (
        f"atomic_config import published physics: flag/levels = {first!r}"
    )
    assert second == (False, None), (
        f"importing main published physics: flag/levels = {second!r}"
    )
    assert not (tmp_path / 'terminal_log.txt').exists()


def test_importing_runner_and_physics_modules_does_not_initialize():
    """B: importing the heavy production modules leaves the seam untouched.

    In-process on purpose: those modules are usually already imported by the
    rest of the suite, and re-importing must not flip the flag either way.
    """
    before = acfg.is_atomic_physics_initialized()
    import rydberg_aoa.simulation.runner  # noqa: F401
    import rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21  # noqa: F401
    import rydberg_aoa.visualization.plot_energy_levels  # noqa: F401

    after = acfg.is_atomic_physics_initialized()
    assert before is after, (
        "importing production modules changed the initialization flag "
        f"({before} -> {after}); imports must stay side-effect free"
    )


# ---------------------------------------------------------------------------
# C + D. Guards and guarded readers fail loudly, never implicitly initialize
# ---------------------------------------------------------------------------
def test_is_atomic_physics_initialized_reflects_flag(monkeypatch):
    """C: the predicate is an honest reader of the seam flag."""
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", False)
    assert acfg.is_atomic_physics_initialized() is False
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", True)
    assert acfg.is_atomic_physics_initialized() is True


def test_require_guard_raises_only_when_uninitialized(monkeypatch):
    """C: the guard raises a self-explanatory error iff uninitialized."""
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", False)
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        acfg.require_atomic_physics_initialized()
    message = str(excinfo.value)
    assert "initialize_atomic_physics" in message, (
        f"guard error must tell the caller how to initialize: {message}"
    )
    assert "ARCH-001" in message, f"guard error must name ARCH-001: {message}"

    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", True)
    acfg.require_atomic_physics_initialized()  # no raise when initialized


def test_every_guarded_reader_fails_loudly_without_implicit_init(monkeypatch):
    """D: no reader may initialize implicitly or substitute a default."""
    _force_uninitialized(monkeypatch)

    readers = (
        acfg.get_cached_detunings,
        acfg.get_cached_dipole_matrix,
        acfg.get_arc_calculated_rf_dipole,
        acfg.get_rf_orientation,
        acfg.print_arc_summary,
    )
    for reader in readers:
        with pytest.raises(PhysicsConfigurationError) as excinfo:
            reader()
        assert "initialize_atomic_physics" in str(excinfo.value), (
            f"{reader.__name__}: error must point at the explicit seam"
        )
        # The read attempt itself must not have initialized anything.
        assert acfg.is_atomic_physics_initialized() is False, (
            f"{reader.__name__} auto-initialized on read (forbidden by ARCH-001)"
        )
        assert acfg.ATOMIC_LEVELS is None, (
            f"{reader.__name__} published levels on a failed read"
        )


def test_runner_entry_points_require_initialization(monkeypatch):
    """B/D: the simulation runner refuses to run without resolved physics."""
    from rydberg_aoa.simulation.runner import SimulationRunner

    _force_uninitialized(monkeypatch)
    runner = SimulationRunner()
    with pytest.raises(PhysicsConfigurationError):
        runner.run_steady_state_spectrum()
    with pytest.raises(PhysicsConfigurationError):
        runner._sensor_probe_detunings((0.0, 0.0, 0.0))
    assert acfg.is_atomic_physics_initialized() is False


# ---------------------------------------------------------------------------
# E. Transactional initialization: failure publishes nothing
# ---------------------------------------------------------------------------
def test_failed_level_optimization_publishes_nothing(monkeypatch):
    """Step-1 failure: the original exception propagates, state unchanged."""
    _force_uninitialized(monkeypatch)

    def _boom():
        raise RuntimeError("ARCH001 sentinel level failure")

    monkeypatch.setattr(acfg, "_initialize_atomic_levels", _boom)

    with pytest.raises(RuntimeError, match="ARCH001 sentinel"):
        acfg.initialize_atomic_physics()

    assert acfg.is_atomic_physics_initialized() is False
    for name in _PUBLISHED_GLOBALS:
        assert getattr(acfg, name) is None, (
            f"{name} was published despite a failed initialization"
        )


def test_mid_pipeline_failure_publishes_nothing(monkeypatch):
    """Failure AFTER levels were computed (step 2) still publishes nothing.

    This is the transactional core of ARCH-001: partial progress inside the
    pipeline must never become observable as a "valid-looking" runtime.
    """
    _force_uninitialized(monkeypatch)

    fake_levels = {
        "ground": {"n": 5, "l": 0, "j": 0.5},
        "excited": {"n": 5, "l": 1, "j": 1.5},
        "rydberg1": {"n": 26, "l": 2, "j": 2.5},
        "rydberg2": {"n": 27, "l": 1, "j": 1.5},
    }
    monkeypatch.setattr(acfg, "_initialize_atomic_levels", lambda: fake_levels)

    # MODE-001: the seam now passes the resolution's cache key to both
    # readers (so an explicit resolution_request can find its own entry).
    # The fakes accept it exactly like their _decay/_laser siblings do.
    def _dipole_boom(*args, **kwargs):
        raise RuntimeError("ARCH001 sentinel dipole failure")

    monkeypatch.setattr(acfg, "_resolve_arc_rf_dipole", _dipole_boom)

    with pytest.raises(RuntimeError, match="ARCH001 sentinel dipole"):
        acfg.initialize_atomic_physics()

    assert acfg.is_atomic_physics_initialized() is False
    for name in _PUBLISHED_GLOBALS:
        assert getattr(acfg, name) is None, (
            f"{name} leaked into the runtime after a mid-pipeline failure"
        )


# ---------------------------------------------------------------------------
# F. Idempotency / force (ARC steps mocked — control flow, not physics)
# ---------------------------------------------------------------------------
def _install_fake_pipeline(monkeypatch, calls):
    """Replace every ARC-backed step of the seam with a deterministic fake.

    The fakes record invocation counts and return fixed values in the units
    the seam expects (Rabi helpers return rad/s; the seam converts to rad/us
    by dividing by US_TO_S_FACTOR).  Nothing here validates physics — only
    the seam's control flow.
    """
    import rydberg_aoa.arc.decay_rates as decay_rates_module

    fake_levels = {
        "ground": {"n": 5, "l": 0, "j": 0.5},
        "excited": {"n": 5, "l": 1, "j": 1.5},
        "rydberg1": {"n": 26, "l": 2, "j": 2.5},
        "rydberg2": {"n": 27, "l": 1, "j": 1.5},
    }

    def _levels():
        calls["levels"] += 1
        return fake_levels

    # MODE-001: the seam passes the resolution's cache key to these two
    # readers; the fakes ignore it (they return fixed values) exactly as the
    # _decay/_laser/_rf/_signal fakes below already do.
    def _dipole(*args, **kwargs):
        calls["dipole"] += 1
        return 1.0e-27

    def _decay(*args, **kwargs):
        calls["decay"] += 1
        return (2.0, 3.0, 4.0)

    def _orientation(*args, **kwargs):
        # RF-DIR-001B: synthetic DOWNWARD orientation for the seam's publish
        # step (carrier -1 Hz). Control flow only — no physics validated here.
        from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
        return resolve_rf_orientation(-1.0)

    def _laser(*args, **kwargs):
        calls["laser"] += 1
        return 4.0 * acfg.US_TO_S_FACTOR  # rad/s -> 4.0 rad/us after /1e6

    def _rf(*args, **kwargs):
        calls["rf"] += 1
        return 6.0 * acfg.US_TO_S_FACTOR

    def _signal(*args, **kwargs):
        calls["signal"] += 1
        return 8.0 * acfg.US_TO_S_FACTOR

    def _detunings(*args, **kwargs):
        calls["detunings"] += 1
        return (0.1, 0.2, 0.3)

    monkeypatch.setattr(acfg, "_initialize_atomic_levels", _levels)
    monkeypatch.setattr(acfg, "_resolve_arc_rf_dipole", _dipole)
    monkeypatch.setattr(acfg, "ATOM", object())  # non-None: ARC branch taken
    monkeypatch.setattr(decay_rates_module, "calculate_decay_rates", _decay)
    monkeypatch.setattr(
        decay_rates_module, "resolve_rydberg_orientation", _orientation
    )
    monkeypatch.setattr(acfg, "calculate_laser_rabi_frequency", _laser)
    monkeypatch.setattr(acfg, "calculate_rf_rabi_frequency", _rf)
    monkeypatch.setattr(acfg, "get_signal_rf_rabi_frequency", _signal)
    monkeypatch.setattr(acfg, "_resolve_cached_detunings", _detunings)
    return fake_levels


def test_initialize_is_idempotent_and_force_reruns(monkeypatch):
    """A successful init publishes once; re-calls are no-ops unless forced.

    Every ARC-backed step is replaced by a deterministic fake, so this test
    pins the *control flow* of the seam (publish-once, force-to-rerun), never
    any physical value.
    """
    _force_uninitialized(monkeypatch)
    calls = {"levels": 0, "dipole": 0, "decay": 0, "laser": 0, "rf": 0,
             "signal": 0, "detunings": 0}
    fake_levels = _install_fake_pipeline(monkeypatch, calls)

    # --- first call: full pipeline runs and publishes exactly once ---
    assert acfg.initialize_atomic_physics() is True
    assert acfg.is_atomic_physics_initialized() is True
    assert acfg.ATOMIC_LEVELS is fake_levels
    assert acfg.OMEGA_P == pytest.approx(4.0)
    assert acfg.OMEGA_C == pytest.approx(4.0)
    assert acfg.OMEGA_RF_BASE == pytest.approx(6.0)
    assert acfg.OMEGA_SIGNAL_RF_BASE == pytest.approx(8.0)
    assert acfg.GAMMA_21 == pytest.approx(2.0)
    assert acfg.GAMMA_32 == pytest.approx(3.0)
    assert acfg.GAMMA_43 == pytest.approx(4.0)
    # RF-DIR-001B: the synthetic orientation is published with the runtime
    # (guarded reader; None would mean the publish step was skipped).
    assert acfg.get_rf_orientation() is not None
    assert acfg.get_rf_orientation().is_downward
    assert acfg.DELTA_P == pytest.approx(0.1)
    assert acfg.DELTA_C == pytest.approx(0.2)
    assert acfg.DELTA_RF == pytest.approx(0.3)
    # heterodyne bases follow the ARC values (step-4 side effect, unchanged)
    assert params.OMEGA_LO_BASE == pytest.approx(6.0)
    assert params.OMEGA_S_BASE == pytest.approx(8.0)
    first_call = dict(calls)

    # --- second call: idempotent, no step re-executes ---
    assert acfg.initialize_atomic_physics() is True
    assert calls == first_call, (
        f"second call re-ran the pipeline: {first_call} -> {calls}"
    )

    # --- force=True: the pipeline runs again ---
    assert acfg.initialize_atomic_physics(force=True) is True
    assert calls["levels"] == 2, (
        "force=True must re-run the level optimization "
        f"(got {calls['levels']} runs)"
    )
    assert calls["detunings"] == 2
    assert acfg.is_atomic_physics_initialized() is True


def test_forced_failure_clears_previous_success(monkeypatch):
    """A failed ``force=True`` re-run must not keep advertising stale values.

    The forced path un-publishes first (``_clear_published_atomic_state``),
    so a failing re-run leaves the runtime uninitialized rather than showing
    values that no longer match the rewritten caches.
    """
    _force_uninitialized(monkeypatch)
    calls = {"levels": 0, "dipole": 0, "decay": 0, "laser": 0, "rf": 0,
             "signal": 0, "detunings": 0}
    _install_fake_pipeline(monkeypatch, calls)
    assert acfg.initialize_atomic_physics() is True
    assert acfg.is_atomic_physics_initialized() is True

    # 2) forced re-run fails at step 1
    def _boom():
        raise RuntimeError("ARCH001 sentinel forced failure")

    monkeypatch.setattr(acfg, "_initialize_atomic_levels", _boom)
    with pytest.raises(RuntimeError, match="ARCH001 sentinel forced"):
        acfg.initialize_atomic_physics(force=True)

    assert acfg.is_atomic_physics_initialized() is False, (
        "a failed forced re-run must clear the initialized flag"
    )
    for name in _PUBLISHED_GLOBALS:
        assert getattr(acfg, name) is None, (
            f"{name} still advertised after a failed forced re-run"
        )


# ---------------------------------------------------------------------------
# G. Dynamic module access (no stale from-import snapshots)
# ---------------------------------------------------------------------------
def test_plot_energy_levels_reads_state_through_module_object():
    """G: plot_energy_levels must resolve ATOMIC_LEVELS dynamically."""
    from rydberg_aoa.visualization import plot_energy_levels as ple_module

    assert not hasattr(ple_module, "ATOMIC_LEVELS"), (
        "plot_energy_levels holds a stale from-imported ATOMIC_LEVELS snapshot"
    )
    assert getattr(ple_module, "acfg", None) is acfg, (
        "plot_energy_levels must read atomic state via the acfg module object"
    )


def test_plot_energy_levels_refuses_to_draw_uninitialized(monkeypatch):
    """G: the level diagram never draws from stale/absent levels."""
    from rydberg_aoa.visualization.plot_energy_levels import plot_energy_levels

    _force_uninitialized(monkeypatch)
    with pytest.raises(PhysicsConfigurationError):
        plot_energy_levels()
    assert acfg.is_atomic_physics_initialized() is False


def test_transmission_module_has_no_stale_atomic_imports():
    """G: susceptibility reads atomic physics via ``acfg``, not a snapshot."""
    from rydberg_aoa.physics.susceptibility import (
        calculate_transmission_from_rho21 as transmission_module,
    )

    for stale in ("ATOMIC_LEVELS", "ATOM", "get_cached_dipole_matrix"):
        assert not hasattr(transmission_module, stale), (
            f"calculate_transmission_from_rho21 kept a from-imported {stale}"
        )
    assert getattr(transmission_module, "acfg", None) is acfg


# ---------------------------------------------------------------------------
# H. Production entry points call the seam explicitly
# ---------------------------------------------------------------------------
def test_main_delegates_to_explicit_cli_without_implicit_initialization(monkeypatch):
    """H: public main uses the profile CLI, without a second initialization."""
    import main
    import cli
    sentinel = object()
    calls = []
    def explicit_cli():
        calls.append('cli')
        return sentinel
    monkeypatch.setattr(cli, 'main', explicit_cli)
    monkeypatch.setattr(acfg, 'initialize_atomic_physics',
                        lambda **kwargs: pytest.fail('main initialized implicitly'))
    assert main.main() is sentinel
    assert calls == ['cli']


def test_legacy_cli_calls_seam_before_running_simulation():
    """H: the legacy programmatic CLI initializes before building its runner."""
    cli_path = os.path.join(REPO_ROOT, "cli.py")
    with open(cli_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    init_pos = source.find("acfg.initialize_atomic_physics()")
    runner_pos = source.find("SimulationRunner()")
    assert init_pos != -1, "cli.py no longer calls initialize_atomic_physics()"
    assert runner_pos != -1, "cli.py no longer constructs SimulationRunner()"
    assert init_pos < runner_pos, (
        "cli.py constructs SimulationRunner before initialize_atomic_physics()"
    )

