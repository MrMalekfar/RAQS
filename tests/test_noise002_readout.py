"""Effective readout noise; shared measured IF/FFT/DoA, not atomic noise."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.simulation.readout_noise import apply_gaussian_readout
from rydberg_aoa.exceptions import PhysicsCalculationError
from rydberg_aoa.utils.numerical import extract_if_component
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool, oracle_record
from rydberg_aoa.simulation import runner as module
from rydberg_aoa.doa.snapshots import extract_if_snapshots

ROOT = Path(__file__).resolve().parents[1]


def noise(*, enabled=True, source=True, sigma=.05, seed=20261012):
    return parse_noise_config({'schema_version': 1, 'enabled': enabled, 'seed': seed,
        'trials': 1, 'sources': {'readout_gaussian': {'enabled': source, 'sigma_observable': sigma}}})


@pytest.mark.parametrize('config', [None, noise(enabled=False), noise(source=False), noise(sigma=0)])
def test_noop_identity_and_no_rng(monkeypatch, config):
    traces = np.ones((2, 20))
    def forbidden(*args, **kwargs):
        pytest.fail('no-op readout must consume no randomness')
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    measured, added, metadata = apply_gaussian_readout(traces, config)
    assert measured is traces
    if config is not None and config.active_sources:
        assert not metadata['noise_applied']
        assert np.array_equal(added, np.zeros_like(traces))
    else:
        assert added is metadata is None


def test_gaussian_statistics_reproducibility_and_input_not_mutated():
    traces = np.full((3, 50000), .4)
    original = traces.copy()
    measured, added, meta = apply_gaussian_readout(traces, noise(sigma=.1))
    assert np.array_equal(traces, original)
    assert np.array_equal(measured, traces + added)
    assert np.array_equal(measured, apply_gaussian_readout(traces, noise(sigma=.1))[0])
    assert not np.array_equal(added, apply_gaussian_readout(traces, noise(sigma=.1, seed=9))[1])
    # Analytic Gaussian sampling-error bounds, fixed before measurement.
    assert np.all(abs(added.mean(axis=1)) < 6*.1/np.sqrt(traces.shape[1]))
    assert np.all(abs(added.var(axis=1)/.01 - 1) < 6*np.sqrt(2/(traces.shape[1]-1)))
    assert abs(np.corrcoef(added)[0, 1]) < 6/np.sqrt(traces.shape[1])
    assert meta['trials_executed'] == 1 and meta['noise_applied']


@pytest.mark.parametrize('bad', [np.ones(20), np.ones((2, 0)), np.ones((2, 5), dtype=bool),
    np.full((2, 5), '1'), np.ones((2, 5), dtype=complex), [[1], [1, 2]], np.full((2, 5), np.nan)])
def test_invalid_clean_batch_fails_without_repair(bad):
    with pytest.raises(PhysicsCalculationError):
        apply_gaussian_readout(bad, noise())


def test_rng_failure_is_typed_and_preserves_cause(monkeypatch):
    cause = RuntimeError('injected random failure')
    def failed(*args, **kwargs):
        raise cause
    monkeypatch.setattr(np.random, 'SeedSequence', failed)
    with pytest.raises(PhysicsCalculationError) as error:
        apply_gaussian_readout(np.ones((2, 20)), noise())
    assert error.value.__cause__ is cause


@pytest.mark.parametrize('method', ['phase_beamforming', 'bartlett', 'music', 'mvdr', 'capon', 'esprit', 'root_music'])
@pytest.mark.parametrize('if_sign', [1, -1])
def test_every_doa_method_consumes_same_measured_trace(isolation, monkeypatch, method, if_sign):
    req = request(isolation)
    if if_sign < 0:
        from rydberg_aoa.config.rf_configuration import configure_rf_fields
        configure_rf_fields(lo=req.lo, signal=replace(req.signal, frequency_hz=req.lo.frequency_hz-1e5))
        req = request(isolation)
    cfg = plan(doa_method=method)
    tasks = []
    fake_pool(monkeypatch, tasks)
    runner = module.SimulationRunner()
    out = runner.run_laboratory_simulation(req, cfg, noise=noise())
    measured = np.asarray(out['time_traces'])
    clean = out['clean_time_traces']
    assert np.array_equal(measured, clean + out['additive_readout_noise'])
    assert not np.array_equal(measured, clean)
    assert runner.results is out['worker_records']
    assert out['readout_noise']['spectrum_readout_noise_applied'] is False
    for index, record in enumerate(out['worker_records']):
        amp, xf, yf = extract_if_component(measured[index], cfg.tlist_us, req.signed_if_mhz, cfg.transient_delay_us)
        assert np.array_equal(record[1], xf) and np.array_equal(record[2], yf)
        expected = np.conj(amp)  # Reference ladder is energy-downward.
        assert out['complex_responses'][index] == expected
        assert np.array_equal(out['clean_worker_records'][index][4], oracle_record(tasks[index])[4])
    raw_snapshots = extract_if_snapshots(cfg.tlist_us, measured, req.signed_if_mhz,
        start_us=cfg.transient_delay_us, cycles_per_snapshot=cfg.cycles_per_snapshot)
    assert np.array_equal(out['if_snapshots'].X, np.conj(raw_snapshots.X))
    assert out['doa_result'].algorithm == ('mvdr' if method == 'capon' else method)
    assert out['doa_result'].diagnostics['settings_source'] == 'doa_config'
    # A gross sign mirror would move this high-SNR synthetic case by 60 degrees.
    assert abs(out['estimated_aoa'] - cfg.target_azimuth_deg) < 10


@pytest.mark.parametrize('config', [None, noise(enabled=False), noise(source=False), noise(sigma=0)])
def test_runner_no_noise_preserves_baseline(isolation, monkeypatch, config):
    req, cfg = request(isolation), plan()
    fake_pool(monkeypatch, [])
    runner = module.SimulationRunner()
    clean = runner.run_laboratory_simulation(req, cfg)
    other = runner.run_laboratory_simulation(req, cfg, noise=config)
    assert np.array_equal(clean['complex_responses'], other['complex_responses'])
    assert np.array_equal(clean['time_traces'], other['time_traces'])
    assert np.array_equal(clean['if_snapshots'].X, other['if_snapshots'].X)
    assert np.array_equal(clean['power'], other['power'])


def test_worker_failure_cannot_be_replaced_by_noise(isolation, monkeypatch):
    def failed(task):
        raise PhysicsCalculationError('injected worker failure')
    fake_pool(monkeypatch, [], worker=failed)
    runner = module.SimulationRunner()
    runner.results = ['stale']
    with pytest.raises(PhysicsCalculationError, match='injected worker failure'):
        runner.run_laboratory_simulation(request(isolation), plan(), noise=noise())
    assert runner.results is None


def test_measured_if_failure_never_publishes_clean_fallback(isolation, monkeypatch):
    # Build valid clean worker records before replacing the canonical IF helper.
    records = []
    fake_pool(monkeypatch, records)
    def failed(*args, **kwargs):
        raise ValueError('injected measured IF failure')
    monkeypatch.setattr('rydberg_aoa.utils.numerical.extract_if_component', failed)
    runner = module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError, match='measured IF extraction failed') as error:
        runner.run_laboratory_simulation(request(isolation), plan(), noise=noise())
    assert isinstance(error.value.__cause__, ValueError)
    assert runner.results is None


def test_noise_cannot_invent_a_signal_from_zero_clean_responses(isolation, monkeypatch):
    def zero(task):
        amp, xf, yf, index, trace = oracle_record(task)
        return 0j, xf, np.zeros_like(yf), index, np.zeros_like(trace)
    fake_pool(monkeypatch, [], worker=zero)
    def forbidden(*args, **kwargs):
        pytest.fail('noise must not be drawn to fabricate a signal')
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    runner = module.SimulationRunner()
    with pytest.raises(PhysicsCalculationError, match='noise cannot invent'):
        runner.run_laboratory_simulation(request(isolation), plan(), noise=noise())
    assert runner.results is None


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_actual_noisy_profile_entry_and_saved_provenance(tmp_path, mode):
    data = json.loads((ROOT / f'examples/noise002/{mode}_gaussian.json').read_text())
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(data))
    process = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(profile),
        '--no-plots', '--output-dir', str(tmp_path)], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8', errors='replace', timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    assert report['noise_included'] and report['noise_status'] == 'applied'
    assert report['readout_noise']['trials_executed'] == 1
    assert report['applied_fields'] == data['fields']
    assert not report['readout_noise']['spectrum_readout_noise_applied']
    with np.load(tmp_path / 'simulation_arrays.npz', allow_pickle=False) as arr:
        assert np.array_equal(arr['time_traces'], arr['clean_time_traces'] + arr['additive_readout_noise'])
        assert not np.array_equal(arr['time_traces'], arr['clean_time_traces'])
        for index in range(len(arr['time_traces'])):
            amp, xf, yf = extract_if_component(arr['time_traces'][index], arr['tlist'], .1,
                                               data['execution']['transient_delay_us'])
            assert arr['complex_responses'][index] == np.conj(amp)
            assert np.array_equal(arr[f'fft_complex_path_{index}'], yf)
