"""Readout trials: independent streams, honest statistics, complete provenance."""

from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.noise_campaign import trace_snr_db, summarize_readout_campaign
from rydberg_aoa.simulation.readout_noise import apply_gaussian_readout
from tests.test_noise002_readout import noise
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool
from rydberg_aoa.simulation import runner as module

ROOT = Path(__file__).resolve().parents[1]


def campaign(n=4, **kwargs):
    return replace(noise(**kwargs), trials=n)


def test_stream_prefix_extension_and_trial_identity():
    clean = np.ones((3, 100))
    a = [apply_gaussian_readout(clean, campaign(3), trial=i)[1] for i in range(3)]
    b = [apply_gaussian_readout(clean, campaign(6), trial=i)[1] for i in range(3)]
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], a[1])
    with pytest.raises(PhysicsConfigurationError):
        apply_gaussian_readout(clean, campaign(3), trial=3)


@pytest.mark.parametrize('method', ['phase_beamforming', 'bartlett', 'music', 'mvdr', 'capon', 'esprit', 'root_music'])
def test_all_methods_all_trials_share_one_clean_evolution(isolation, monkeypatch, method):
    tasks = []
    fake_pool(monkeypatch, tasks)
    req, cfg = request(isolation), plan(doa_method=method)
    runner = module.SimulationRunner()
    out = runner.run_laboratory_simulation(req, cfg, noise=campaign())
    assert len(tasks) == len(cfg.positions_m)  # No repeated physical worker solves.
    summary = out['noise_campaign']
    assert summary['trials_completed'] == summary['trials_requested'] == 4
    errors = []
    for i, trial in enumerate(out['noise_trial_results']):
        assert trial['readout_noise']['trial_index'] == i
        expected = apply_gaussian_readout(trial['clean_time_traces'], campaign(), trial=i)[0]
        assert np.array_equal(trial['time_traces'], expected)
        assert trial['doa_result'].diagnostics['effective_settings'] == out['doa_result'].diagnostics['effective_settings']
        assert trial['if_snapshots'].normalization_polarity == -1
        if method in ('esprit', 'root_music'):
            assert trial['power'] is trial['angles'] is None
        errors.append((trial['estimated_aoa'] - cfg.target_azimuth_deg + 180) % 360 - 180)
    assert summary['rmse_deg'] == pytest.approx(np.sqrt(np.mean(np.square(errors))))
    assert summary['bias_deg'] == pytest.approx(np.mean(errors))
    assert summary['error_sample_std_deg'] == pytest.approx(np.std(errors, ddof=1))
    assert runner.results is out['worker_records']
    assert np.array_equal(out['time_traces'], out['noise_trial_results'][0]['time_traces'])


@pytest.mark.parametrize('config', [campaign(enabled=False), campaign(source=False)])
def test_disabled_campaign_consumes_no_rng_and_no_fake_trials(isolation, monkeypatch, config):
    fake_pool(monkeypatch, [])
    req, cfg = request(isolation), plan()
    clean = module.SimulationRunner().run_laboratory_simulation(req, cfg)
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('disabled RNG'))
    out = module.SimulationRunner().run_laboratory_simulation(req, cfg, noise=config)
    assert 'noise_campaign' not in out
    assert np.array_equal(out['complex_responses'], clean['complex_responses'])


def test_sigma_zero_campaign_retains_all_trials_without_rng(isolation, monkeypatch):
    fake_pool(monkeypatch, [])
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('zero-sigma RNG'))
    out = module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=campaign(sigma=0))
    rows = out['noise_campaign']['trials']
    assert len(rows) == 4
    assert out['noise_campaign']['error_sample_std_deg'] == 0
    for row in rows:
        assert row['trace_snr_db_per_path'] == [None]*3
        assert row['snr_undefined_reason_per_path'] == ['zero_additive_noise_power']*3
    for trial in out['noise_trial_results']:
        assert np.array_equal(trial['time_traces'], trial['clean_time_traces'])


@pytest.mark.parametrize('stage', ['readout', 'doa'])
def test_late_trial_failure_aborts_without_survivor_publication(isolation, monkeypatch, stage):
    fake_pool(monkeypatch, [])
    from rydberg_aoa.simulation import readout_noise
    real = readout_noise.apply_gaussian_readout
    cause = PhysicsCalculationError('injected trial failure')
    def failed(traces, config, *, trial=0):
        if trial == 2:
            raise cause
        return real(traces, config, trial=trial)
    if stage == 'readout':
        monkeypatch.setattr(readout_noise, 'apply_gaussian_readout', failed)
    else:
        real_estimate = module.estimate_doa_from_results
        calls = []
        def failed_estimate(*args, **kwargs):
            calls.append(1)
            if len(calls) == 4:  # clean, trial 0, trial 1, then trial 2.
                raise cause
            return real_estimate(*args, **kwargs)
        monkeypatch.setattr(module, 'estimate_doa_from_results', failed_estimate)
    runner = module.SimulationRunner()
    runner.results = ['stale']
    with pytest.raises(PhysicsCalculationError, match='trial 2 failed') as error:
        runner.run_laboratory_simulation(request(isolation), plan(), noise=campaign())
    assert error.value.__cause__ is cause
    assert runner.results is runner._laboratory_result_runtime is None


def test_if_noise_variance_matches_independent_least_squares_prediction():
    from rydberg_aoa.utils.numerical import extract_if_component
    times = np.linspace(0, 40, 401)
    phase = 2*np.pi*.1*times
    clean = np.cos(phase)[None, :]
    cfg = campaign(300, sigma=.2)
    perturbations = []
    baseline = extract_if_component(clean[0], times, .1)[0]
    for i in range(cfg.trials):
        measured = apply_gaussian_readout(clean, cfg, trial=i)[0]
        perturbations.append(extract_if_component(measured[0], times, .1)[0]-baseline)
    design = np.column_stack([np.ones(len(times)), np.cos(phase), np.sin(phase)])
    covariance = .2**2*np.linalg.inv(design.T@design)
    values = np.array(perturbations)
    for component, index in [(values.real, 1), (values.imag, 2)]:
        expected = covariance[index, index]
        assert abs(component.mean()) < 6*np.sqrt(expected/cfg.trials)
        assert abs(component.var(ddof=1)/expected-1) < 6*np.sqrt(2/(cfg.trials-1))


def test_snr_independent_power_oracle_and_transient_cut():
    # Retained AC = [-1,1]; noise = [.5,.5] -> power ratio 4 = 6.0206 dB.
    values, reason = trace_snr_db([[100, -1, 1]], [[100, .5, .5]], [0, 1, 2], 1)
    assert values == pytest.approx([10*np.log10(4)])
    assert reason == [None]


@pytest.mark.parametrize('scale', [1e-250, 1e250])
def test_snr_scale_invariance_without_overflow_or_epsilon(scale):
    values, _ = trace_snr_db(np.array([[-1, 1]])*scale, np.array([[.5, .5]])*scale, [0, 1], 0)
    assert values == pytest.approx([10*np.log10(4)], abs=1e-10)


@pytest.mark.parametrize('clean,additive,reason', [([[1, 1]], [[1, 1]], 'zero_clean_ac_power'),
    ([[-1, 1]], [[0, 0]], 'zero_additive_noise_power')])
def test_undefined_snr_is_null_with_reason(clean, additive, reason):
    assert trace_snr_db(clean, additive, [0, 1], 0) == ([None], [reason])


@pytest.mark.parametrize('times,delay', [([0, 0], 0), ([0, 1], -1), ([0, 1], 2), ([0, float('nan')], 0)])
def test_invalid_snr_grid_fails(times, delay):
    with pytest.raises(PhysicsCalculationError):
        trace_snr_db([[-1, 1]], [[1, 1]], times, delay)


def test_wrapped_statistics_are_not_naive_angle_mean():
    cfg = SimpleNamespace(target_azimuth_deg=180., transient_delay_us=0.)
    clean = {'estimated_aoa': 180.}
    trials = [dict(noise_config=campaign(2), readout_noise={'trial_index': i}, estimated_aoa=angle,
                   clean_time_traces=np.array([[-1, 1]]), additive_readout_noise=np.array([[1, 1]]),
                   tlist=np.array([0, 1])) for i, angle in enumerate([179., -179.])]
    report = summarize_readout_campaign(trials, clean, cfg)
    assert report['bias_deg'] == 0 and report['rmse_deg'] == 1
    assert report['error_sample_std_deg'] == pytest.approx(np.sqrt(2))
    with pytest.raises(PhysicsCalculationError, match='incomplete'):
        summarize_readout_campaign(trials[:1], clean, cfg)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_actual_campaign_cli_saves_all_trials(tmp_path, mode):
    data = json.loads((ROOT/f'examples/noise003/{mode}_campaign.json').read_text())
    data['noise']['trials'] = 3
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    command = [sys.executable, '-B', 'main.py', '--profile', str(path), '--output-dir', str(tmp_path / 'run')]
    if mode == 'experimental':
        command.append('--no-plots')
    proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'),
                          capture_output=True, text=True, errors='replace', timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path/'run' / 'run_report.json').read_text(encoding='utf-8'))
    assert report['noise_campaign']['trials_completed'] == 3
    assert report['sensing']['estimated_aoa_deg'] == report['noise_campaign']['trials'][0]['estimated_aoa_deg']
    assert report['applied_fields'] == data['fields']
    with np.load(tmp_path/'run' / 'simulation_arrays.npz', allow_pickle=False) as a:
        assert a['trial_time_traces'].shape[0] == 3
        for i in range(3):
            assert np.array_equal(a['trial_time_traces'][i], a['clean_time_traces']+a['trial_additive_readout_noise'][i])
            expected = apply_gaussian_readout(a['clean_time_traces'], parse_noise_config(data['noise']), trial=i)[1]
            assert np.array_equal(a['trial_additive_readout_noise'][i], expected)
        assert np.array_equal(a['time_traces'], a['trial_time_traces'][0])
    if mode == 'prescribed':
        assert (tmp_path/'run' / 'fig_noise_campaign.png').is_file()
