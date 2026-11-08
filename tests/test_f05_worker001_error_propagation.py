"""F05-WORKER-001 — fail-loud worker errors and safe runner publication.

A failed heterodyne element must ABORT the run visibly.  It must never be
turned into a ``None``-filled six-element tuple, and stale/partial results
must never reach phase extraction or AoA estimation.

Covers the worker exception contract (``run_single_heterodyne_worker``), the
observable check before ``np.imag``, the runner batch-validation/publication
contract (``_validate_worker_records`` + ``run_time_domain_simulation`` +
``process_simulation_results``), and exception transport through a real
one-child multiprocessing.Pool.

Physics here is SYNTHETIC (small four-level model); nothing claims laboratory
or EIT validation.
"""

import multiprocessing

import numpy as np
import pytest
import qutip

from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import parameters as params
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)
from rydberg_aoa.simulation import runner as runner_mod
from rydberg_aoa.simulation import time_domain as td
from rydberg_aoa.simulation.runner import (
    SimulationRunner,
    _validate_worker_records,
)
from rydberg_aoa.simulation.time_domain import run_single_heterodyne_worker


# ---------------------------------------------------------------------------
# Synthetic four-level model (no ARC, no atomic initialization)
# ---------------------------------------------------------------------------
def _synthetic_model():
    H = qutip.Qobj(np.diag([0.0, -0.3, -0.5, -0.4]).astype(complex))
    for a, b, omega in ((0, 1, 5.0), (1, 2, 3.0), (2, 3, 2.0)):
        link = np.zeros((4, 4), complex)
        link[a, b] = link[b, a] = omega / 2.0
        H += qutip.Qobj(link)
    c_ops = []
    for (a, b), rate in zip(((0, 1), (1, 2), (2, 3)), (1.0, 0.1, 0.01)):
        jump = np.zeros((4, 4), complex)
        jump[a, b] = 1.0
        c_ops.append(qutip.Qobj(np.sqrt(rate) * jump))
    obs = qutip.Qobj(np.array([[0, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, 0]], complex))
    return H, c_ops, obs


def _nonstationary():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    m = a @ a.conj().T
    return m / np.trace(m)


def _worker_args(i=0):
    return (i, 1.0 + 0.0j, [0.0, 1.0, 2.0], 1.0,
            5.0, 3.0, 2.0,
            0.3, -0.2, 0.1,
            [1.0, 1.0, 1.0], [0.2, 0.2, 0.2])


def _worker_kwargs(**overrides):
    kwargs = dict(noise_model=None, transit_rate_per_us=0.0, rf_orientation=None)
    kwargs.update(overrides)
    return kwargs


def _valid_state(_t):
    """A valid 4x4 density matrix (pure |0><0|) for callback-driving mocks."""
    return qutip.ket2dm(qutip.basis(4, 0))


def _invalid_state(_t):
    """A trace-one Hermitian 4x4 matrix with a negative eigenvalue (-0.2)."""
    return qutip.Qobj(np.diag([1.2, -0.2, 0.0, 0.0]).astype(complex))


def _mock_mesolve_driving_callback(state_factory):
    """Mock mesolve that invokes the worker's e_ops callback for each output
    time with states from ``state_factory(t)`` (NUM-TD-001 integration).

    It drives the REAL certification+qt.expect callback the worker installs,
    so it never duplicates certification logic; ``state_factory`` alone decides
    whether the sampled state is valid.
    """
    def _mock(H_t, rho0, tlist, c_ops, e_ops, options=None, **kwargs):
        op = e_ops[0] if isinstance(e_ops, (list, tuple)) else e_ops
        values = [op(float(t), state_factory(t)) for t in tlist]
        return type("R", (), {"expect": [np.array(values)]})()

    return _mock


def _mock_mesolve_callback_then_return(bad_expect, state_factory=_valid_state):
    """Mock mesolve that first drives the worker's e_ops callback for the
    FULL output grid (so the NUM-TD-001 coverage gate passes), then hands
    back a deliberately malformed ``result.expect`` payload.

    This isolates the F05-WORKER-001 observable guards: certification has
    genuinely run, yet the solver's expectation array itself is broken —
    defense-in-depth behind the coverage check.
    """
    def _mock(H_t, rho0, tlist, c_ops, e_ops, options=None, **kwargs):
        op = e_ops[0] if isinstance(e_ops, (list, tuple)) else e_ops
        for t in tlist:
            op(float(t), state_factory(t))
        return type("R", (), {"expect": [bad_expect]})()

    return _mock


def _rec(i=0, amp=0.5 + 0.25j, xf=None, yf=None, trace=None, tlen=3):
    if xf is None:
        xf = np.array([0.0, 1.0])
    if yf is None:
        yf = np.array([1.0, 2.0])
    if trace is None:
        trace = np.zeros(tlen)
    return (amp, xf, yf, i, trace)


# ---------------------------------------------------------------------------
# 1. Worker exception contract
# ---------------------------------------------------------------------------
def test_worker_propagates_physics_configuration_error_unchanged(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(
        td, "create_sensor_model",
        lambda *a, **k: (_ for _ in ()).throw(
            PhysicsConfigurationError("declared-config failure")))
    with pytest.raises(PhysicsConfigurationError,
                       match="declared-config failure"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


def test_worker_steady_state_failure_propagates_unchanged(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(td, "steadystate",
                        lambda *a, **k: qutip.Qobj(_nonstationary()))
    with pytest.raises(PhysicsCalculationError,
                       match="run_single_heterodyne_worker"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


def test_worker_mesolve_failure_names_stage_and_chains_cause(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(
        td, "mesolve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(PhysicsCalculationError) as excinfo:
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())
    message = str(excinfo.value)
    assert "element 0" in message
    assert "mesolve" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "boom" in str(excinfo.value.__cause__)


def test_worker_if_extraction_failure_names_stage(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    monkeypatch.setattr(
        td, "extract_if_component",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("if boom")))
    with pytest.raises(PhysicsCalculationError, match="IF extraction"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


# ---------------------------------------------------------------------------
# 2. Observable validation before np.imag
# ---------------------------------------------------------------------------
def test_worker_missing_observable_fails(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    # Certification runs for the full grid, yet the expectation payload is
    # missing entirely (NUM-TD-001 makes this the only way to reach the
    # guard: a fabricated expect that never certified fails the coverage gate).
    monkeypatch.setattr(td, "mesolve",
                        _mock_mesolve_callback_then_return([]))
    with pytest.raises(PhysicsCalculationError, match="observable"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


def test_worker_nonfinite_observable_fails_before_imag(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    bad = np.array([0.0, np.nan, 1.0])          # non-finite REAL component
    monkeypatch.setattr(td, "mesolve",
                        _mock_mesolve_callback_then_return(bad))
    with pytest.raises(PhysicsCalculationError, match="non-finite"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


def test_worker_wrong_length_observable_fails(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    bad = np.zeros(5)                            # dispatched grid length is 3
    monkeypatch.setattr(td, "mesolve",
                        _mock_mesolve_callback_then_return(bad))
    with pytest.raises(PhysicsCalculationError, match="time grid"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())


# ---------------------------------------------------------------------------
# 3. Real (bounded) worker success + orientation preservation
# ---------------------------------------------------------------------------
def test_real_worker_success_returns_five_tuple():
    """Bounded real numerical worker success (synthetic physics)."""
    tlist = np.linspace(0.0, 60.0, 600)
    args = (0, 1.0 + 0.5j, tlist, 1.0,
            5.0, 3.0, 2.0,
            0.3, -0.2, 0.1,
            [1.0, 1.0, 1.0], [0.2, 0.2, 0.2])
    result = run_single_heterodyne_worker(*args, noise_model=None,
                                          transit_rate_per_us=0.0,
                                          rf_orientation=None)
    assert isinstance(result, tuple) and len(result) == 5
    amp, xf, yf, idx, trace = result
    assert idx == 0
    assert np.isfinite(amp)
    assert np.all(np.isfinite(trace))
    assert trace.shape[0] == len(tlist)


def test_orientation_none_and_upward_are_bit_identical():
    """RF-DIR-001D: legacy (None) and upward (+1) keep the same coefficient."""
    from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
    up = resolve_rf_orientation(1.0e11)          # positive -> upward edge
    tlist = np.linspace(0.0, 60.0, 600)
    common = (0, 1.0 + 0.5j, tlist, 1.0,
              5.0, 3.0, 2.0, 0.3, -0.2, 0.1,
              [1.0, 1.0, 1.0], [0.2, 0.2, 0.2])
    legacy = run_single_heterodyne_worker(*common, noise_model=None,
                                          transit_rate_per_us=0.0,
                                          rf_orientation=None)
    upward = run_single_heterodyne_worker(*common, noise_model=None,
                                          transit_rate_per_us=0.0,
                                          rf_orientation=up)
    assert legacy[0] == upward[0]


# ---------------------------------------------------------------------------
# 4. Runner batch validation (_validate_worker_records)
# ---------------------------------------------------------------------------
def test_valid_batch_passes_and_is_returned():
    recs = [_rec(0), _rec(1), _rec(2)]
    out = _validate_worker_records(recs, tlist_len=3, expected_count=3,
                                   require_contiguous_indices=True)
    assert out == recs


def test_zero_amplitude_is_valid():
    recs = [_rec(0, amp=0.0 + 0.0j)]
    out = _validate_worker_records(recs, tlist_len=3, expected_count=1,
                                   require_contiguous_indices=True)
    assert out[0][0] == 0.0


def test_none_batch_rejected():
    with pytest.raises(PhysicsCalculationError, match="None"):
        _validate_worker_records(None)


def test_empty_batch_rejected():
    with pytest.raises(PhysicsCalculationError, match="empty"):
        _validate_worker_records([])


def test_wrong_count_rejected():
    with pytest.raises(PhysicsCalculationError, match="exactly 2"):
        _validate_worker_records([_rec(0)], expected_count=2)


def test_wrong_tuple_length_rejected():
    recs = [(0.5 + 0.25j, np.array([0.0]), np.array([1.0]), 0, np.zeros(3),
             "err")]
    with pytest.raises(PhysicsCalculationError, match="5-element"):
        _validate_worker_records(recs, tlist_len=3)


def test_none_field_rejected():
    recs = [(None, np.array([0.0]), np.array([1.0]), 0, np.zeros(3))]
    with pytest.raises(PhysicsCalculationError, match="None field"):
        _validate_worker_records(recs, tlist_len=3)


def test_nonfinite_amplitude_rejected():
    with pytest.raises(PhysicsCalculationError, match="non-finite amplitude"):
        _validate_worker_records([_rec(0, amp=float("nan"))], tlist_len=3)


def test_noninteger_index_rejected():
    rec = _rec(0)
    rec = (rec[0], rec[1], rec[2], 0.5, rec[4])
    with pytest.raises(PhysicsCalculationError, match="non-integer"):
        _validate_worker_records([rec], tlist_len=3)


def test_duplicate_index_rejected():
    with pytest.raises(PhysicsCalculationError, match="duplicate"):
        _validate_worker_records([_rec(0), _rec(0)], expected_count=2,
                                 require_contiguous_indices=True)


def test_noncontiguous_index_rejected():
    with pytest.raises(PhysicsCalculationError, match="do not match"):
        _validate_worker_records([_rec(0), _rec(2)], expected_count=2,
                                 require_contiguous_indices=True)


def test_mismatched_fft_lengths_rejected():
    with pytest.raises(PhysicsCalculationError, match="xf/yf"):
        _validate_worker_records(
            [_rec(0, xf=np.array([0.0, 1.0]), yf=np.array([1.0]))], tlist_len=3)


def test_wrong_trace_length_rejected():
    with pytest.raises(PhysicsCalculationError, match="time grid"):
        _validate_worker_records([_rec(0, trace=np.zeros(5))], tlist_len=3)


def test_nonfinite_trace_rejected():
    with pytest.raises(PhysicsCalculationError, match="signal trace"):
        _validate_worker_records(
            [_rec(0, trace=np.array([0.0, np.inf, 1.0]))], tlist_len=3)


# ---------------------------------------------------------------------------
# 5. Runner publication: no stale/partial results on failure
# ---------------------------------------------------------------------------
def test_runner_failure_leaves_no_stale_results(monkeypatch):
    runner = SimulationRunner()
    runner.results = [_rec(i) for i in range(params.N_ELEMENTS)]
    runner.element_phases_rad = np.zeros(params.N_ELEMENTS)
    monkeypatch.setattr(runner_mod, "calculate_dephasing_rates",
                        lambda: [0.0, 0.0, 0.0])
    monkeypatch.setattr(runner_mod, "resolve_effective_transit_rate",
                        lambda: 0.0)
    monkeypatch.setattr(runner_mod.SimulationRunner, "_sensor_probe_detunings",
                        lambda self, pd: (0.0, 0.0, 0.0))
    for name in ("OMEGA_P", "OMEGA_C", "OMEGA_RF_BASE", "OMEGA_SIGNAL_RF_BASE",
                 "GAMMA_21", "GAMMA_32", "GAMMA_43"):
        monkeypatch.setattr(acfg, name, 1.0)
    monkeypatch.setattr(acfg, "get_cached_detunings", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(acfg, "get_rf_orientation", lambda: None)

    class _BoomPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starmap(self, fn, args):
            raise RuntimeError("pool failed")

    monkeypatch.setattr(multiprocessing, "Pool", _BoomPool)
    with pytest.raises(RuntimeError, match="pool failed"):
        runner.run_time_domain_simulation()
    assert runner.results is None


def test_process_simulation_results_rejects_malformed_records():
    runner = SimulationRunner()
    runner.results = [(None, np.array([0.0]), np.array([1.0]), 0, np.zeros(3))]
    with pytest.raises(PhysicsCalculationError, match="None field"):
        runner.process_simulation_results()


# ---------------------------------------------------------------------------
# 6. Exception transport through a real one-child process
# ---------------------------------------------------------------------------
def test_worker_failure_transports_through_a_real_child_process():
    """A real Pool(1) child failure surfaces in the parent (Windows spawn).

    Uses a deterministic failing input (non-finite detuning) — no monkeypatch,
    which spawn does not inherit.  Asserts parent-side type/context only; it
    does not claim cross-process object identity or identical __cause__.
    """
    tlist = np.linspace(0.0, 1.0, 5)
    args = (0, 1.0 + 0.0j, tlist, 1.0,
            1.0, 1.0, 1.0,
            float("nan"), 0.0, 0.0,          # delta_p = nan -> deterministic
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
            None, 0.0, None)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        with pytest.raises(PhysicsCalculationError) as excinfo:
            pool.starmap(run_single_heterodyne_worker, [args])
    message = str(excinfo.value)
    assert "run_single_heterodyne_worker" in message or "element 0" in message


# ---------------------------------------------------------------------------
# 7. F05-WORKER-001-R1: strict scalar/array/index contracts (public boundary)
# ---------------------------------------------------------------------------
def _process_with(records):
    runner = SimulationRunner()
    runner.results = records
    return runner.process_simulation_results()


@pytest.mark.parametrize("bad_amp", [
    True, np.bool_(True), "1+2j", b"1", np.array([1 + 2j]), np.array(1.0),
    [1 + 2j],
])
def test_public_boundary_rejects_non_scalar_amplitude(bad_amp):
    rec = (bad_amp, np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0, np.zeros(3))
    with pytest.raises(PhysicsCalculationError, match="numeric scalar"):
        _process_with([rec])


def test_public_boundary_wraps_overflow_amplitude_with_cause():
    rec = (10 ** 1000, np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0,
           np.zeros(3))
    with pytest.raises(PhysicsCalculationError) as excinfo:
        _process_with([rec])
    assert "cannot be converted" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OverflowError)


def test_public_boundary_rejects_non_iterable_batch():
    with pytest.raises(PhysicsCalculationError, match="not iterable"):
        _process_with(12345)


def test_public_boundary_rejects_complex_xf():
    rec = (0.5 + 0.25j, np.array([0j, 1j]), np.array([1.0, 2.0]), 0,
           np.zeros(3))
    with pytest.raises(PhysicsCalculationError, match="real array"):
        _process_with([rec])


def test_public_boundary_rejects_complex_trace():
    rec = (0.5 + 0.25j, np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0,
           np.array([0j, 1j, 2j]))
    with pytest.raises(PhysicsCalculationError, match="real array"):
        _process_with([rec])


def test_public_boundary_rejects_empty_trace():
    rec = (0.5 + 0.25j, np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0,
           np.array([]))
    with pytest.raises(PhysicsCalculationError, match="nonempty"):
        _process_with([rec])


def test_public_boundary_rejects_negative_index():
    rec = (0.5 + 0.25j, np.array([0.0, 1.0]), np.array([1.0, 2.0]), -1,
           np.zeros(3))
    with pytest.raises(PhysicsCalculationError, match="negative"):
        _process_with([rec])


def test_public_boundary_preserves_complex_yf_and_zero_amplitude():
    rec = (0.0 + 0.0j, np.array([0.0, 1.0]),
           np.array([1.0 + 2.0j, 3.0]), 0, np.zeros(3))
    complex_responses, xf_0, yf_0, time_traces = _process_with([rec])
    assert complex_responses[0] == 0.0            # zero amplitude preserved
    assert np.iscomplexobj(yf_0)                  # complex yf preserved
    assert yf_0[0] == pytest.approx(1.0 + 2.0j)


def test_worker_rejects_boolean_observable_before_imag(monkeypatch):
    H, c_ops, obs = _synthetic_model()
    monkeypatch.setattr(td, "create_sensor_model",
                        lambda *a, **k: (H, list(c_ops), obs))
    bad = np.array([True, False, True])           # boolean observable
    monkeypatch.setattr(td, "mesolve",
                        _mock_mesolve_callback_then_return(bad))
    with pytest.raises(PhysicsCalculationError, match="non-numeric or boolean"):
        td.run_single_heterodyne_worker(*_worker_args(), **_worker_kwargs())



