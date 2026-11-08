"""Colored RF: OU node statistics, Hermitian RHS and certified re-solves."""

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.rf_drive_noise import build_rf_realization, RFDriveRealization
from rydberg_aoa.simulation import runner as module, time_domain as td
from rydberg_aoa.utils.numerical import extract_if_component
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool, oracle_record

ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return json.loads((ROOT/'examples/noise005/prescribed_rf.json').read_text())['noise']


def noise(**changes):
    data = declaration()
    data.update(changes)
    return parse_noise_config(data)


@pytest.mark.parametrize('field,bad', [('stationary_std', True), ('stationary_std', -1),
    ('stationary_std', '1'), ('stationary_std', float('nan')), ('correlation_time_us', 0),
    ('units', 'Hz'), ('enabled', 1)])
def test_rf_spec_invalid_even_disabled(field, bad):
    data = declaration()
    data['enabled'] = False
    data['sources']['rf_relative_phase'][field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


@pytest.mark.parametrize('field,bad', [('noise_step_us', 0), ('horizon_us', .35),
    ('spatial_scope', 'independent'), ('preparation', 'steadystate(H(t))'),
    ('interpolation', 'cubic'), ('phase_budget', 'already_in_linewidth')])
def test_unsupported_rf_contract_fails(field, bad):
    data = declaration()
    data['drive'][field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


@pytest.mark.parametrize('field,bad', [('noise_step_us', 1.), ('horizon_us', 60.)])
def test_invalid_execution_precedes_arc(isolation, monkeypatch, field, bad):
    data = declaration()
    data['drive'][field] = bad
    from rydberg_aoa.config import atomic_config as ac
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **k: pytest.fail('invalid RF config reached ARC'))
    with pytest.raises(PhysicsConfigurationError):
        module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=parse_noise_config(data))


def test_nested_readout_requires_one_trial_and_seed_identity():
    data = json.loads((ROOT/'examples/noise005/prescribed_rf_pin.json').read_text())['noise']
    assert parse_noise_config(data).physical_readout_enabled
    for field in ('seed', 'trials'):
        bad = copy.deepcopy(data)
        bad['readout'][field] += 1
        with pytest.raises(PhysicsConfigurationError, match='identical seed and trials'):
            parse_noise_config(bad)


def test_ou_stationary_covariance_and_mean_one_amplitude_independent_oracle():
    data = declaration()
    data['trials'] = 2000
    data['drive'].update(horizon_us=1., noise_step_us=.2)
    cfg = parse_noise_config(data)
    realizations = [build_rf_realization(cfg, trial=i) for i in range(cfg.trials)]
    for name, attribute in [('rf_relative_phase', 'relative_phase_rad'), ('rf_lo_amplitude', 'lo_amplitude')]:
        spec = data['sources'][name]
        sigma = spec['stationary_std']
        values = np.asarray([getattr(r, attribute)[:2] for r in realizations])
        if name == 'rf_lo_amplitude':
            assert abs(values[:, 0].mean()-1) < 6*np.sqrt(np.expm1(sigma**2)/cfg.trials)
            values = np.log(values)+.5*sigma**2
        expected_a = np.exp(-.2/spec['correlation_time_us'])
        assert abs(values[:, 0].mean()) < 6*sigma/np.sqrt(cfg.trials)
        assert abs(values[:, 0].var(ddof=1)/sigma**2-1) < 6*np.sqrt(2/(cfg.trials-1))
        covariance = np.mean(values[:, 0]*values[:, 1])
        assert abs(covariance/sigma**2-expected_a) < 6*np.sqrt((1+expected_a**2)/cfg.trials)


def test_stream_prefix_source_toggle_interpolation_and_pickling():
    cfg = noise()
    realization = build_rf_realization(cfg, trial=0)
    assert pickle.loads(pickle.dumps(realization)) == realization
    assert realization == build_rf_realization(replace(cfg, trials=6), trial=0)
    extended = replace(cfg, drive=replace(cfg.drive, horizon_us=240.))
    extra = build_rf_realization(extended, trial=0)
    assert extra.relative_phase_rad[:len(realization.times_us)] == realization.relative_phase_rad
    assert realization.at(.1) == pytest.approx(tuple((a[0]+a[1])/2 for a in
        (realization.lo_amplitude, realization.signal_amplitude, realization.relative_phase_rad)))
    for source in cfg.sources:
        toggled = replace(cfg, sources=tuple(replace(s, specification=replace(s.specification, enabled=False))
                          if s.name == source.name else s for s in cfg.sources))
        r = build_rf_realization(toggled, trial=0)
        for name, attribute in [('rf_lo_amplitude', 'lo_amplitude'), ('rf_signal_amplitude', 'signal_amplitude'), ('rf_relative_phase', 'relative_phase_rad')]:
            if name != source.name:
                assert getattr(r, attribute) == getattr(realization, attribute)
    with pytest.raises(PhysicsCalculationError, match='no clamp'):
        realization.at(121.)


@pytest.mark.parametrize('bad', [(1., True), (1., np.array(1.)), (1., '1'), (1., complex(1, 0))])
def test_realization_rejects_coercible_nonnumeric_samples(bad):
    with pytest.raises(PhysicsConfigurationError):
        RFDriveRealization((0., 1.), bad, (1., 1.), (0., 0.))


def dynamic_oracle(task):
    record = oracle_record(task)
    if task.rf_drive is None:
        return record
    phase = np.angle(task.omega_signal_complex)
    times = np.asarray(task.tlist_us)
    drive = np.asarray([task.rf_drive.at(t) for t in times])
    trace = drive[:, 1]*np.cos(2*np.pi*task.signed_if_mhz*times + task.rf_orientation.direction*phase + .35 + drive[:, 2])
    trace += .01*(drive[:, 0]-1)
    amp, xf, yf = extract_if_component(trace, times, task.signed_if_mhz, task.transient_delay_us)
    return amp, xf, yf, task.element_index, trace


@pytest.mark.parametrize('method', ['phase_beamforming', 'bartlett', 'music', 'mvdr', 'capon', 'esprit', 'root_music'])
def test_all_methods_recompute_every_trial_with_common_drive(isolation, monkeypatch, method):
    tasks = []
    fake_pool(monkeypatch, tasks, worker=dynamic_oracle)
    req, run, cfg = request(isolation), plan(doa_method=method), noise()
    result = module.SimulationRunner().run_laboratory_simulation(req, run, noise=cfg)
    paths = len(run.positions_m)
    assert len(tasks) == (1+cfg.trials)*paths
    assert result['rf_drive_campaign']['atomic_evolutions_per_path'] == 4
    for i in range(cfg.trials):
        batch = tasks[(i+1)*paths:(i+2)*paths]
        assert all(t.rf_drive is batch[0].rf_drive for t in batch)
        assert batch[0].rf_drive == build_rf_realization(cfg, trial=i)
        trial = result['rf_drive_trial_results'][i]
        for path, trace in enumerate(trial['time_traces']):
            amp, xf, yf = extract_if_component(trace, run.tlist_us, req.signed_if_mhz, run.transient_delay_us)
            assert trial['complex_responses'][path] == np.conj(amp)
            assert np.array_equal(trial['worker_records'][path][2], yf)
    assert not np.array_equal(result['time_traces'], result['nominal_time_traces'])


@pytest.mark.parametrize('kind', ['master', 'sources', 'zero'])
def test_disabled_zero_rf_is_bit_identical_and_rng_free(isolation, monkeypatch, kind):
    fake_pool(monkeypatch, [])
    req, run = request(isolation), plan()
    baseline = module.SimulationRunner().run_laboratory_simulation(req, run)
    data = declaration()
    if kind == 'master':
        data['enabled'] = False
    else:
        for spec in data['sources'].values():
            spec['stationary_std' if kind == 'zero' else 'enabled'] = 0. if kind == 'zero' else False
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('disabled/zero RF RNG'))
    result = module.SimulationRunner().run_laboratory_simulation(req, run, noise=parse_noise_config(data))
    assert np.array_equal(result['time_traces'], baseline['time_traces'])
    assert np.array_equal(result['complex_responses'], baseline['complex_responses'])
    if kind == 'zero':
        assert result['rf_drive_campaign']['atomic_evolutions_per_path'] == 1
        assert all(r is None for r in result['rf_drive_realizations'])


def test_late_atomic_worker_failure_aborts_without_nominal_fallback(isolation, monkeypatch):
    calls = []
    cause = PhysicsCalculationError('injected driven ODE failure')
    def failed(task):
        if task.rf_drive is not None:
            calls.append(task)
            if len(calls) > 3:
                raise cause
        return dynamic_oracle(task)
    fake_pool(monkeypatch, [], worker=failed)
    runner = module.SimulationRunner()
    runner.results = ['stale']
    with pytest.raises(PhysicsCalculationError, match='trial 1 failed') as error:
        runner.run_laboratory_simulation(request(isolation), plan(), noise=noise())
    assert error.value.__cause__ is cause
    assert runner.results is runner._laboratory_result_runtime is None


@pytest.mark.parametrize('readout', ['gaussian', 'pin'])
def test_drive_and_readout_use_same_trial_without_nested_campaign(isolation, monkeypatch, readout):
    data = declaration()
    file = ROOT/('examples/noise003/prescribed_campaign.json' if readout == 'gaussian'
                 else 'examples/noise004/prescribed_pin.json')
    nested = json.loads(file.read_text())['noise']
    nested['trials'] = data['trials']
    data['readout'] = nested
    cfg = parse_noise_config(data)
    tasks = []
    fake_pool(monkeypatch, tasks, worker=dynamic_oracle)
    kwargs = {}
    if readout == 'pin':
        from rydberg_aoa.simulation.detector_readout import DetectorContext
        kwargs['detector_context'] = DetectorContext(1e-7, .1, -.5, 780.241476e-9)
    out = module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=cfg, **kwargs)
    assert len(tasks) == 12  # reference + 3 drive trials, three paths each; no T^2.
    assert len(out['rf_drive_trial_results']) == 3
    for i, trial in enumerate(out['rf_drive_trial_results']):
        assert trial['readout_noise']['trial_index'] == i
        assert np.array_equal(trial['time_traces'], trial['clean_time_traces']+trial['additive_readout_noise'])
    assert out['rf_drive_campaign']['trace_units'] == ('V' if readout == 'pin' else 'Im<sigma21> per real sample')


def synthetic_worker(times, rf_drive, orientation=None, if_sign=1):
    return td.run_single_heterodyne_worker(0, .05*np.exp(.3j), times, if_sign*.1,
        .2, .6, .4, .03, -.02, .01, (.8, .1, .05), (.01, .02, .02),
        transit_rate_per_us=.2, rf_orientation=orientation, rf_drive=rf_drive)


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('if_sign', [1, -1])
def test_real_worker_hermitian_off_grid_and_no_rng_in_rhs(monkeypatch, direction, if_sign):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params, 'ATOMIC_TRANSIENT_DELAY_US', 0.)
    realization = build_rf_realization(noise(), trial=0)
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('RNG inside solver'))
    real = td.mesolve
    def capture(H, *args, **kwargs):
        evo = qt.QobjEvo(H)
        for t in (.1, 3.14159, 8.35):
            a_lo, a_sig, phi = realization.at(t)
            rf = .4*a_lo+.05*np.exp(.3j)*a_sig*np.exp(1j*(direction*2*np.pi*if_sign*.1*t+phi))
            expected = np.array([[0, -.1, 0, 0], [-.1, .03, -.3, 0],
                                 [0, -.3, .01, -.5*np.conj(rf)], [0, 0, -.5*rf, .01+direction*.01]], complex)
            actual = evo(t).full()
            assert np.allclose(actual, expected, rtol=1e-14, atol=1e-15)
            assert np.array_equal(actual, actual.conj().T)
        return real(H, *args, **kwargs)
    monkeypatch.setattr(td, 'mesolve', capture)
    orientation = SimpleNamespace(direction=direction, is_downward=direction<0,
        upper_ladder_index=4 if direction>0 else 3, lower_ladder_index=3 if direction>0 else 4)
    result = synthetic_worker(np.linspace(0, 10, 101), realization, orientation, if_sign)
    assert len(result) == 5 and np.isfinite(result[4]).all()


def test_real_grid_refinement_uses_same_realization_and_zero_identity(monkeypatch):
    from rydberg_aoa.config import parameters as params
    monkeypatch.setattr(params, 'ATOMIC_TRANSIENT_DELAY_US', 0.)
    realization = build_rf_realization(noise(), trial=0)
    coarse = synthetic_worker(np.linspace(0, 20, 201), realization)
    fine = synthetic_worker(np.linspace(0, 20, 401), realization)
    # Same H(t), finer requested output; bound chosen against existing default
    # BDF accuracy, without changing solver tolerances or regenerating noise.
    assert np.allclose(coarse[4], fine[4][::2], rtol=1e-4, atol=1e-6)
    assert np.allclose(coarse[0], fine[0], rtol=1e-3, atol=1e-5)


def test_legacy_hamiltonian_plugin_cannot_obscure_rf_contract():
    realization = build_rf_realization(noise(), trial=0)
    with pytest.raises(PhysicsCalculationError, match='joint model contract'):
        td.run_single_heterodyne_worker(0, .05, np.linspace(0, 10, 101), .1,
            .2, .6, .4, 0, 0, 0, (.8, .1, .05), (.01, .02, .02),
            transit_rate_per_us=.2, rf_drive=realization, noise_model=lambda H, c, **kw: (H, c))


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_real_rf_cli_and_saved_trajectories(tmp_path, mode):
    data = json.loads((ROOT/f'examples/noise005/{mode}_rf.json').read_text())
    data['noise']['trials'] = 1
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    args = [sys.executable, '-B', 'main.py', '--profile', str(path), '--output-dir', str(tmp_path)]
    if mode == 'experimental':
        args.append('--no-plots')
    proc = subprocess.run(args, cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'), capture_output=True,
                          text=True, errors='replace', timeout=240)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    report = json.loads((tmp_path/'run_report.json').read_text(encoding='utf-8'))
    assert report['rf_drive_included'] and report['noise_included']
    assert report['applied_fields'] == data['fields']
    assert not report['spectrum_drive_noise_included']
    with np.load(tmp_path/'simulation_arrays.npz', allow_pickle=False) as a:
        assert not np.array_equal(a['nominal_time_traces'], a['time_traces'])
        expected = build_rf_realization(parse_noise_config(data['noise']), trial=0)
        assert np.array_equal(a['trial_rf_lo_amplitude'][0], expected.lo_amplitude)
        assert np.array_equal(a['drive_noise_times_us'], expected.times_us)
    if mode == 'prescribed':
        assert (tmp_path/'fig_rf_drive_noise.png').is_file()


def test_real_rf_pin_cli_keeps_detector_and_drive_provenance(tmp_path):
    data = json.loads((ROOT/'examples/noise005/prescribed_rf_pin.json').read_text())
    data['noise']['trials'] = data['noise']['readout']['trials'] = 1
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    proc = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(path), '--no-plots',
        '--output-dir', str(tmp_path)], cwd=ROOT, capture_output=True, text=True, errors='replace', timeout=240)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    report = json.loads((tmp_path/'run_report.json').read_text(encoding='utf-8'))
    assert report['rf_drive_included'] and report['readout_noise']['noise_applied']
    assert report['sensing_trace_units'] == 'V'
    assert report['rf_drive_campaign']['atomic_evolutions_per_path'] == 2
    with np.load(tmp_path/'simulation_arrays.npz', allow_pickle=False) as a:
        assert np.array_equal(a['trial_time_traces'][0], a['time_traces'])
        assert np.array_equal(a['time_traces'], a['clean_time_traces']+a['additive_readout_noise'])
        assert a['trial_rf_lo_amplitude'].shape[0] == a['trial_measured_current_a'].shape[0] == 1
