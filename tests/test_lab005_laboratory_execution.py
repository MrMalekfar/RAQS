"""Explicit dispatch and independent signed-IF/spatial-phase oracles."""

from dataclasses import replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.constants import C_LIGHT
from rydberg_aoa.config.rf_configuration import configure_rf_fields
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation import runner as module, time_domain as td
from rydberg_aoa.simulation.laboratory_execution import LaboratoryRunConfig, LaboratoryWorkerTask, run_laboratory_worker
from rydberg_aoa.utils.numerical import extract_if_component
from tests.test_lab004_laboratory_runtime import isolation, request, applied_snapshot


def plan(**changes):
    spacing = C_LIGHT / params.RF_SIGNAL_FREQUENCY / 8
    config = LaboratoryRunConfig(tuple(np.linspace(0, 60, 601)),
                                 tuple((i*spacing, 0.) for i in range(3)), -30., 4.,
                                 (.02, .03, .03), .5, sensor_probe_offset_mhz=.125)
    return replace(config, **changes)


def oracle_record(task):
    # Known sinusoidal input; this is seam evidence, not an atomic solver.
    times = np.asarray(task.tlist_us)
    phase = np.angle(task.omega_signal_complex)
    sigma = task.rf_orientation.direction
    trace = np.cos(2*np.pi*task.signed_if_mhz*times + sigma*phase + .35)
    amp, xf, yf = extract_if_component(trace, times, task.signed_if_mhz, task.transient_delay_us)
    return amp, xf, yf, task.element_index, trace


def fake_pool(monkeypatch, captured, worker=oracle_record):
    class Pool:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, target, tasks):
            assert target is run_laboratory_worker
            captured.extend(tasks)
            return [worker(task) for task in reversed(tasks)]
    def context(name):
        assert name == 'spawn'
        return SimpleNamespace(Pool=lambda processes: Pool())
    monkeypatch.setattr(module.multiprocessing, 'get_context', context)


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('angle', [-30., 30.])
def test_explicit_lab_dispatch_and_doa_preserve_carrier_if_orientation(isolation, monkeypatch, sign, angle):
    req = request(isolation)
    if sign < 0:
        configure_rf_fields(lo=req.lo, signal=replace(req.signal, frequency_hz=req.lo.frequency_hz - 1e5))
        req = request(isolation)
    cfg, tasks = plan(target_azimuth_deg=angle), []
    before = applied_snapshot()
    fake_pool(monkeypatch, tasks)
    runner = module.SimulationRunner()
    # A reused runner must recompute the declared geometry, not cached phases.
    runner.array_positions = np.zeros((2, 2))
    runner.element_phases_rad = np.ones(2)
    out = runner.run_laboratory_simulation(req, cfg)
    assert applied_snapshot() == before
    assert out['estimated_aoa'] == angle
    assert out['signal_carrier_hz'] == req.signal.frequency_hz
    assert out['signed_if_mhz'] == sign*.1
    expected_phase = 2*np.pi*req.signal.frequency_hz/C_LIGHT * np.asarray(cfg.positions_m)[:, 0] * np.sin(np.deg2rad(angle))
    assert out['element_phases_rad'] == pytest.approx(expected_phase, abs=1e-15)
    assert [r[3] for r in runner.results] == [0, 1, 2]
    runtime = ac.get_laboratory_runtime()
    for i, task in enumerate(tasks):
        assert task.element_index == i
        assert task.signed_if_mhz == sign*.1
        assert task.omegas_rad_per_us == (ac.OMEGA_P, ac.OMEGA_C, ac.OMEGA_RF_BASE)
        assert task.decay_rates_per_us == runtime.rates
        assert task.dephasing_rates_per_us == cfg.dephasing_rates_per_us
        assert task.transit_rate_per_us == .5
        assert task.transient_delay_us == 4
        assert task.detunings_rad_per_us == (ac.DELTA_P + 2*np.pi*.125, ac.DELTA_C, ac.DELTA_RF)
        assert task.rf_orientation.direction == -1
        assert pickle.loads(pickle.dumps(task)) == task
    snapshots = out['if_snapshots']
    assert snapshots.normalization_polarity == -1
    assert snapshots.diagnostics['start_us'] == 4
    assert snapshots.X[:, 0] / snapshots.X[0, 0] == pytest.approx(np.exp(1j*expected_phase), abs=3e-15)
    responses, *_ = runner.process_simulation_results()
    assert np.array_equal(responses, out['complex_responses'])
    assert responses / responses[0] == pytest.approx(np.exp(1j*expected_phase), abs=3e-15)
    with pytest.raises(PhysicsConfigurationError, match='conflicts'):
        runner.process_simulation_results(rf_orientation=replace(runtime.selection.rf_orientation, direction=1,
                                                                 signed_frequency_hz=abs(runtime.selection.rf_orientation.signed_frequency_hz)))


@pytest.mark.parametrize('changes', [dict(transient_delay_us=True), dict(transit_rate_per_us=-1),
    dict(dephasing_rates_per_us=(.1, .2)), dict(processes=False), dict(target_azimuth_deg=91),
    dict(sensor_probe_offset_mhz=np.inf), dict(tlist_us=tuple(range(9))),
    dict(tlist_us=(0., 1., 3., *range(4, 12))), dict(positions_m=((0., 0.), (0., 0.))),
    dict(positions_m=((True, 0.), (1., 0.))), dict(cycles_per_snapshot=0)])
def test_invalid_explicit_execution_inputs_are_rejected(changes):
    with pytest.raises(PhysicsConfigurationError):
        plan(**changes)


@pytest.mark.parametrize('kind', ['dc', 'nyquist', 'delay', 'record'])
def test_sampling_failure_precedes_atomic_initialization(isolation, monkeypatch, kind):
    req, cfg = request(isolation), plan()
    if kind == 'dc':
        req = replace(req, signal=replace(req.signal, frequency_hz=req.lo.frequency_hz))
    elif kind == 'nyquist':
        cfg = replace(cfg, tlist_us=tuple(np.arange(0, 120, 10.)))
    elif kind == 'delay':
        cfg = replace(cfg, transient_delay_us=100.)
    else:
        cfg = replace(cfg, tlist_us=tuple(np.linspace(0, 5, 51)))
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **kwargs: pytest.fail('invalid sampling reached ARC init'))
    runner = module.SimulationRunner()
    runner.results = [('old',)]
    with pytest.raises(PhysicsConfigurationError):
        runner.run_laboratory_simulation(req, cfg)
    assert runner.results is None


@pytest.mark.parametrize('failure', ['worker', 'zero', 'stale', 'batch'])
def test_failed_run_cannot_publish_partial_or_old_results(isolation, monkeypatch, failure):
    req, cfg, tasks = request(isolation), plan(), []
    cause = PhysicsCalculationError('injected worker failure')
    def worker(task):
        if failure == 'worker':
            raise cause
        record = oracle_record(task)
        if failure == 'zero':
            return (0j, *record[1:])
        if failure == 'batch':
            return (None, *record[1:])
        ac.LASER_PARAMS['probe']['power_mw'] += 1e-6
        return record
    fake_pool(monkeypatch, tasks, worker)
    runner = module.SimulationRunner()
    runner.results = [('old',)]
    with pytest.raises((PhysicsCalculationError, PhysicsConfigurationError)) as error:
        runner.run_laboratory_simulation(req, cfg)
    if failure == 'worker':
        assert error.value is cause
    assert runner.results is None and runner._laboratory_result_runtime is None


def test_processing_old_lab_records_after_reinitialization_is_rejected(isolation, monkeypatch):
    req, cfg = request(isolation), plan()
    fake_pool(monkeypatch, [])
    runner = module.SimulationRunner()
    runner.run_laboratory_simulation(req, cfg)
    ac.initialize_atomic_physics(resolution_request=replace(req, bounds=replace(req.bounds, n_max=28)), force=True)
    with pytest.raises(PhysicsConfigurationError, match='older laboratory'):
        runner.process_simulation_results()


def test_explicit_covariance_estimator_reuses_signed_if_snapshots(isolation, monkeypatch):
    req, cfg = request(isolation), plan(doa_method='esprit', target_azimuth_deg=30.)
    fake_pool(monkeypatch, [])
    out = module.SimulationRunner().run_laboratory_simulation(req, cfg)
    assert out['doa_result'].algorithm == 'esprit'
    assert out['estimated_aoa'] == pytest.approx(30., abs=1e-10)
    assert out['angles'] is out['power'] is None  # No fabricated scan for search-free estimator.
    assert out['if_snapshots'].normalization_polarity == -1


def test_spawn_transport_failure_is_wrapped_with_cause(isolation, monkeypatch):
    cause = OSError('pool creation failed')
    def fail(*args):
        raise cause
    monkeypatch.setattr(module.multiprocessing, 'get_context', fail)
    runner = module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError, match='spawn worker dispatch') as error:
        runner.run_laboratory_simulation(request(isolation), plan())
    assert error.value.__cause__ is cause
    assert runner.results is None


def test_doa_failure_cannot_publish_worker_results(isolation, monkeypatch):
    fake_pool(monkeypatch, [])
    cause = PhysicsCalculationError('DoA failed')
    def fail(*args, **kwargs):
        raise cause
    monkeypatch.setattr(module, 'estimate_doa_from_results', fail)
    runner = module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError) as error:
        runner.run_laboratory_simulation(request(isolation), plan())
    assert error.value is cause and runner.results is None


def test_same_request_with_recalculated_rates_cannot_reuse_old_worker_records(isolation, monkeypatch):
    req, cfg = request(isolation), plan()
    fake_pool(monkeypatch, [])
    runner = module.SimulationRunner()
    runner.run_laboratory_simulation(req, cfg)
    original = ac.get_laboratory_runtime()
    monkeypatch.setattr(isolation, 'getTransitionRate', lambda *args: 100.)
    ac.initialize_atomic_physics(resolution_request=req, force=True)
    current = ac.get_laboratory_runtime()
    assert original.identity == current.identity and original.rates != current.rates
    with pytest.raises(PhysicsConfigurationError, match='older laboratory'):
        runner.process_simulation_results()


def test_dark_drives_cannot_produce_an_invented_angle(isolation, monkeypatch):
    req = request(isolation)
    req = replace(req, signal=replace(req.signal, electric_field_vm=0))
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **kwargs: pytest.fail('zero beat drive reached ARC'))
    with pytest.raises(PhysicsConfigurationError, match='nonzero'):
        module.SimulationRunner().run_laboratory_simulation(req, plan())


@pytest.mark.parametrize('fails', [False, True])
def test_child_adapter_supplies_and_restores_explicit_delay(isolation, monkeypatch, fails):
    req = request(isolation)
    ac.initialize_atomic_physics(resolution_request=req)
    runtime, cfg = ac.get_laboratory_runtime(), plan()
    task = LaboratoryWorkerTask(0, 1j, cfg.tlist_us, req.signed_if_mhz, (1., 2., 3.),
                               runtime.detunings, runtime.rates, cfg.dephasing_rates_per_us,
                               .5, runtime.selection.rf_orientation, 4.)
    previous = params.ATOMIC_TRANSIENT_DELAY_US
    cause = PhysicsCalculationError('child failure')
    def worker(*args, **kwargs):
        assert params.ATOMIC_TRANSIENT_DELAY_US == 4
        assert kwargs['transit_rate_per_us'] == .5
        if fails:
            raise cause
        return 'sentinel'
    monkeypatch.setattr(td, 'run_single_heterodyne_worker', worker)
    if fails:
        with pytest.raises(PhysicsCalculationError) as error:
            run_laboratory_worker(task)
        assert error.value is cause
    else:
        assert run_laboratory_worker(task) == 'sentinel'
    assert params.ATOMIC_TRANSIENT_DELAY_US == previous
