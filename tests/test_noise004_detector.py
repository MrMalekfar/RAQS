"""Explicit detector units, Poisson oracle, PSD scaling and production wiring."""

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.constants import elementary_charge as Q_E, Boltzmann as K_B

from tests.noise_profile_fixtures import legacy_noise_profile

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.config.detector import validate_detector_sampling
from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.simulation.detector_readout import DetectorContext, detector_forward, apply_pin_readout
from tests.test_lab005_laboratory_execution import isolation, request, plan, fake_pool, oracle_record
from rydberg_aoa.simulation import runner as module
from rydberg_aoa.utils.numerical import extract_if_component

ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return legacy_noise_profile((ROOT/'examples/noise004/prescribed_pin.json'))['noise']


def config(active=(), **detector):
    data = declaration()
    data['trials'] = 2
    data['detector'].update(detector)
    for name in data['sources']:
        data['sources'][name]['enabled'] = name in active
    return parse_noise_config(data)


@pytest.mark.parametrize('field,bad', [('responsivity_a_per_w', True), ('collection_efficiency', 2),
    ('dark_current_a', -1), ('integration_time_us', 0), ('output_units', 'mV'),
    ('aperture_model', 'unspecified'), ('transimpedance_v_per_a', '1'),
    ('output_units', np.array(['V'])), ('integration_time_us', 1e-320)])
def test_hardware_invalid_even_when_disabled(field, bad):
    data = declaration()
    data['enabled'] = False
    data['detector'][field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_incompatible_units_unknown_sources_and_missing_specs_fail():
    data = declaration()
    for change in ('gaussian', 'missing', 'unknown'):
        changed = copy.deepcopy(data)
        if change == 'gaussian':
            changed['sources']['readout_gaussian'] = {'enabled': True, 'sigma_observable': .1}
        elif change == 'missing':
            del changed['detector']['responsivity_a_per_w']
        else:
            changed['sources']['adc'] = {'enabled': False}
        with pytest.raises(PhysicsConfigurationError):
            parse_noise_config(changed)


def test_wrong_readout_api_never_accepts_incompatible_units():
    from rydberg_aoa.simulation.readout_noise import apply_gaussian_readout
    with pytest.raises(PhysicsConfigurationError, match='schema 1 only'):
        apply_gaussian_readout([[0.]], config(('photon_shot',)))
    for bad in (None, object(), True):
        with pytest.raises(PhysicsConfigurationError):
            apply_pin_readout([[0.]], bad, DetectorContext(1e-8, 1, -1, 780.241476e-9))


def test_sampling_mismatch_fails_before_atomic_initialization(isolation, monkeypatch):
    from rydberg_aoa.config import atomic_config as ac
    monkeypatch.setattr(ac, 'initialize_atomic_physics', lambda **k: pytest.fail('invalid detector sampling reached ARC'))
    with pytest.raises(PhysicsConfigurationError, match='sample step'):
        module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(),
            noise=config(integration_time_us=.2), detector_context=DetectorContext(1e-7, 1, -1, 780.241476e-9))


def test_context_missing_cannot_use_global_optical_defaults(isolation):
    with pytest.raises(PhysicsConfigurationError, match='DetectorContext'):
        module.SimulationRunner().run_laboratory_simulation(request(isolation), plan(), noise=config())


def test_forward_optical_current_voltage_independent_oracle():
    cfg = config(collection_efficiency=.4, responsivity_a_per_w=.6, dark_current_a=2e-9,
                 transimpedance_v_per_a=100.)
    context = DetectorContext(1e-6, 2., -3., 780.241476e-9)
    y = np.array([[-.1, 0., .1]])
    power, current, voltage, transmission = detector_forward(y, cfg.detector, context)
    expected = np.exp(12*y)
    assert np.array_equal(y, [[-.1, 0., .1]])
    assert transmission == pytest.approx(expected)
    assert power == pytest.approx(1e-6*.4*expected)
    assert current == pytest.approx(.6*power+2e-9)
    assert voltage == pytest.approx(100*current)
    # Transmission >1 is not clipped; this declared optical model can predict gain.
    assert transmission[0, -1] > 1


def test_two_level_absorption_sign_without_coherence_conjugation_guess():
    import qutip as qt
    g, e = qt.basis(2, 0), qt.basis(2, 1)
    s21 = e*g.dag()
    omega, gamma = .02, 2.
    rho = qt.steadystate(-.5*omega*(s21+s21.dag()), [np.sqrt(gamma)*s21.dag()])
    y = qt.expect(s21, rho).imag
    expected_y = -omega*gamma/(gamma**2+2*omega**2)
    assert y == pytest.approx(expected_y, abs=1e-15)
    _, _, _, transmission = detector_forward([[y]], config().detector,
        DetectorContext(1e-8, 1, -1, 780.241476e-9))
    assert transmission[0, 0] == pytest.approx(np.exp(2*expected_y))
    assert 0 < transmission[0, 0] < 1


@pytest.mark.parametrize('control', ['master', 'detector'])
def test_detector_full_disable_preserves_old_clean_trace(isolation, monkeypatch, control):
    fake_pool(monkeypatch, [])
    cfg = config(('photon_shot',))
    cfg = replace(cfg, enabled=False) if control == 'master' else replace(cfg, detector=replace(cfg.detector, enabled=False))
    req, run = request(isolation), plan()
    clean = module.SimulationRunner().run_laboratory_simulation(req, run)
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('disabled physical RNG'))
    result = module.SimulationRunner().run_laboratory_simulation(req, run, noise=cfg)
    assert 'detector_arrays' not in result
    assert np.array_equal(result['time_traces'], clean['time_traces'])
    assert np.array_equal(result['complex_responses'], clean['complex_responses'])


def test_poisson_count_mean_variance_and_current_conversion():
    cfg = config(('photon_shot',), collection_efficiency=1, responsivity_a_per_w=.5,
                 integration_time_us=1., dark_current_a=0, output_units='A')
    y = np.zeros((1, 80000))
    _, added, meta, payload = apply_pin_readout(y, cfg, DetectorContext(1e-12, 1, -1, 780.241476e-9))
    counts = payload['photon_shot_electron_counts'][0]
    mean = .5e-12*1e-6/Q_E
    assert abs(counts.mean()-mean) < 6*np.sqrt(mean/len(counts))
    assert abs(counts.var(ddof=1)/mean-1) < 6*np.sqrt((2+1/mean)/len(counts))
    assert added[0] == pytest.approx(counts*Q_E/1e-6-.5e-12)
    assert meta['variance_current_a2']['photon_shot']['min_a2'] == pytest.approx(Q_E*.5e-12/1e-6)


@pytest.mark.parametrize('source', ['electronic_current', 'johnson_current'])
def test_white_current_psd_and_bandwidth_scaling(source):
    cfg = config((source,), output_units='A', integration_time_us=1.)
    y = np.zeros((1, 50000))
    _, _, meta, payload = apply_pin_readout(y, cfg, DetectorContext(1e-12, 1, -1, 780.241476e-9))
    psd = 1e-28 if source == 'electronic_current' else 4*K_B*300/1e9
    variance = psd/(2e-6)
    samples = payload[source+'_current_noise_a'][0]
    assert meta['variance_current_a2'][source]['a2'] == pytest.approx(variance, rel=1e-12, abs=0)
    assert abs(samples.var(ddof=1)/variance-1) < 6*np.sqrt(2/(len(samples)-1))
    wider = replace(cfg, detector=replace(cfg.detector, integration_time_us=.5))
    _, _, _, b = apply_pin_readout(y, wider, DetectorContext(1e-12, 1, -1, 780.241476e-9))
    assert b[source+'_current_noise_a'] == pytest.approx(np.sqrt(2)*payload[source+'_current_noise_a'])


def test_source_toggles_do_not_redraw_other_sources():
    y = np.full((2, 100), -.1)
    context = DetectorContext(1e-8, 1, -1, 780.241476e-9)
    names = tuple(declaration()['sources'])
    _, _, _, all_sources = apply_pin_readout(y, config(names), context)
    for name in names:
        _, _, _, single = apply_pin_readout(y, config((name,)), context)
        assert np.array_equal(single[name+'_current_noise_a'], all_sources[name+'_current_noise_a'])


def test_noiseless_detector_zero_variance_and_no_rng(monkeypatch):
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('disabled detector RNG'))
    y = np.array([[-.1, -.2]])
    measured, added, meta, payload = apply_pin_readout(y, config(), DetectorContext(1e-8, 1, -1, 780.241476e-9))
    assert not meta['noise_applied']
    assert np.array_equal(measured, payload['clean_detector_time_traces'])
    assert np.array_equal(added, np.zeros_like(added))


def test_overflow_is_visible_without_clipping():
    with pytest.raises(PhysicsCalculationError):
        detector_forward([[1e300]], config().detector, DetectorContext(1e-8, 1, -1, 780.241476e-9))


def test_pin_responsivity_cannot_invent_quantum_efficiency_above_one():
    with pytest.raises(PhysicsConfigurationError, match='quantum efficiency'):
        detector_forward([[0.]], config(responsivity_a_per_w=1.).detector,
                         DetectorContext(1e-8, 1, -1, 780.241476e-9))


def test_dark_poisson_variance_and_negative_electronics_are_preserved():
    context = DetectorContext(1e-12, 1, -1, 780.241476e-9)
    cfg = config(('dark_shot',), output_units='A')
    _, _, _, payload = apply_pin_readout(np.zeros((1, 30000)), cfg, context)
    counts = payload['dark_shot_electron_counts'][0]
    mean = cfg.detector.dark_current_a*.1e-6/Q_E
    assert abs(counts.mean()-mean) < 6*np.sqrt(mean/len(counts))
    assert abs(counts.var(ddof=1)/mean-1) < 6*np.sqrt((2+1/mean)/len(counts))
    data = declaration()
    data['sources']['electronic_current']['asd_a_per_sqrt_hz'] = 1e-6
    for name in data['sources']:
        data['sources'][name]['enabled'] = name == 'electronic_current'
    measured, _, _, _ = apply_pin_readout(np.zeros((1, 100)), parse_noise_config(data), context)
    assert (measured < 0).any()  # No physical-value clipping of electronic output.


def test_pin_generation_preserves_rng_failure_cause(monkeypatch):
    cause = RuntimeError('injected PIN RNG error')
    def failed(*a, **k):
        raise cause
    monkeypatch.setattr(np.random, 'SeedSequence', failed)
    with pytest.raises(PhysicsCalculationError) as error:
        apply_pin_readout([[0.]], config(('photon_shot',)), DetectorContext(1e-8, 1, -1, 780.241476e-9))
    assert error.value.__cause__ is cause


@pytest.mark.parametrize('kind', ['atomic_zero', 'zero_sensitivity'])
def test_detector_noise_cannot_invent_signal(isolation, monkeypatch, kind):
    def zero(task):
        record = oracle_record(task)
        return (0j, *record[1:])
    if kind == 'atomic_zero':
        fake_pool(monkeypatch, [], worker=zero)
    else:
        fake_pool(monkeypatch, [])
    monkeypatch.setattr(np.random, 'SeedSequence', lambda *a, **k: pytest.fail('noise fabricated signal'))
    runner = module.SimulationRunner()
    context = DetectorContext(1e-7, 0 if kind == 'zero_sensitivity' else .1, -.5, 780.241476e-9)
    with pytest.raises(PhysicsCalculationError, match='cannot invent|no resolved temporal'):
        runner.run_laboratory_simulation(request(isolation), plan(), noise=config(('photon_shot',)), detector_context=context)
    assert runner.results is None


@pytest.mark.parametrize('method', ['phase_beamforming', 'bartlett', 'music', 'mvdr', 'esprit', 'root_music'])
def test_detector_trace_is_shared_by_if_and_all_doa(isolation, monkeypatch, method):
    fake_pool(monkeypatch, [])
    cfg = config(('photon_shot', 'dark_shot', 'electronic_current', 'johnson_current'))
    runner = module.SimulationRunner()
    out = runner.run_laboratory_simulation(request(isolation), plan(doa_method=method), noise=cfg,
        detector_context=DetectorContext(1e-7, .1, -.5, 780.241476e-9))
    for trial in out['noise_trial_results']:
        assert trial['readout_noise']['units'] == 'V'
        assert np.array_equal(trial['time_traces'], trial['clean_time_traces']+trial['additive_readout_noise'])
        for i, trace in enumerate(trial['time_traces']):
            amp, xf, yf = extract_if_component(trace, out['tlist'], .1, 4.)
            assert trial['complex_responses'][i] == np.conj(amp)
            assert np.array_equal(trial['worker_records'][i][2], yf)
    assert out['noise_campaign']['trace_units'] == 'V'


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_real_detector_profile_and_si_saved_arrays(tmp_path, mode):
    data = json.loads((ROOT/f'examples/noise004/{mode}_pin.json').read_text())
    data['noise']['trials'] = 2
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    args = [sys.executable, '-B', 'main.py', '--profile', str(path), '--output-dir', str(tmp_path)]
    if mode == 'experimental':
        args.append('--no-plots')
    proc = subprocess.run(args, cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'),
        capture_output=True, text=True, errors='replace', timeout=180)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    report = json.loads((tmp_path/'run_report.json').read_text(encoding='utf-8'))
    assert report['sensing_trace_units'] == 'V'
    assert report['readout_noise']['boxcar_enbw_hz'] == 5e6
    assert report['applied_fields'] == data['fields']
    assert report['noise_campaign']['trials_completed'] == 2
    with np.load(tmp_path/'simulation_arrays.npz', allow_pickle=False) as a:
        assert a['trial_measured_current_a'].shape[0] == 2
        assert np.array_equal(a['time_traces'], a['clean_detector_time_traces']+a['additive_readout_noise'])
        assert a['noiseless_current_a'] == pytest.approx(.5*a['clean_optical_power_w']+1e-10)
        assert a['measured_voltage_v'] == pytest.approx(a['measured_current_a']*20000)
        assert a['trial_photon_shot_electron_counts'].dtype.kind in 'iu'
