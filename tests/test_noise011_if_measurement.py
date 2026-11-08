"""Independent finite-window power, phase, covariance and pipeline oracles."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from rydberg_aoa.exceptions import PhysicsCalculationError, PhysicsConfigurationError
from rydberg_aoa.utils.numerical import extract_if_component, if_fit_noise_weights
from rydberg_aoa.simulation.if_measurement import (
    measure_readout_if, pooled_if_snr, relative_phase_errors, measure_sensing_result)
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.config.profile_document import profile_document
from rydberg_aoa.simulation.study_metrics import summarize_study_point
from rydberg_aoa.simulation.study_statistics import analyze_study_statistics

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('frequency', [.7, -.7])
def test_integer_cycles_peak_amplitude_and_white_enbw(frequency):
    times = np.arange(1000)*.01
    selected, weights, meta = if_fit_noise_weights(times, frequency)
    y = 12 + 2*np.cos(2*np.pi*frequency*times+.4)
    assert weights@y == pytest.approx(2*np.exp(.4j), abs=2e-14)
    assert np.array_equal(selected, np.arange(1000))
    # Each quadrature variance = 2*v/N, sum = 4*v/N.
    assert meta['complex_peak_amplitude_white_enbw_hz'] == pytest.approx(200000.)
    assert np.asarray(meta['iq_white_enbw_matrix_hz']) == pytest.approx(np.diag([100000.]*2), abs=1e-9)


@pytest.mark.parametrize('frequency', [.113, -.113])
def test_noninteger_transient_window_same_operator_and_original_fit(frequency):
    times = np.linspace(0, 23, 461)
    rng = np.random.default_rng(70)
    real = rng.normal(size=len(times)) + 5
    complex_trace = real + 1j*rng.normal(size=len(times))
    indices, weights, _ = if_fit_noise_weights(times, frequency, 4.23)
    retained = times[times >= 4.23]
    phase = 2*np.pi*frequency*(retained-retained[0])
    design = np.column_stack([np.ones(len(retained)), np.cos(phase), np.sin(phase)])
    # Independent normal-equation oracle only for this well-conditioned case.
    oracle = np.array([0, 1, -1j]) @ np.linalg.inv(design.T@design) @ design.T
    assert weights == pytest.approx(oracle, abs=1e-16)
    for trace in (real, complex_trace):
        old = np.linalg.lstsq(design, trace[indices], rcond=None)[0]
        new = extract_if_component(trace, times, frequency, 4.23)[0]
        assert new == old[1]-1j*old[2]  # unchanged production fit, bit for bit
        assert weights@trace[indices] == pytest.approx(new, abs=1e-14)


def test_white_variance_and_colored_covariance_need_different_noise_laws():
    t = np.arange(200)*.1
    _, w, meta = if_fit_noise_weights(t, .2)
    rng = np.random.default_rng(114)
    samples = rng.normal(size=(6000, len(t)))
    power = np.mean(np.abs(samples@w)**2)
    prediction = np.sum(np.abs(w)**2)
    # Fixed-seed sample test with ~1/sqrt(6000) relative standard error.
    assert power == pytest.approx(prediction, rel=.05)
    assert prediction == pytest.approx(meta['complex_peak_amplitude_white_enbw_hz']*2/1e7)
    covariance = .8**np.abs(np.arange(len(t))[:, None]-np.arange(len(t))[None, :])
    # Full population covariance oracle: diagonal-white ENBW cannot predict OU.
    colored = (w@covariance@w.conj()).real
    assert colored > 3*prediction
    assert np.mean(np.abs((samples@np.linalg.cholesky(covariance).T)@w)**2) == pytest.approx(colored, rel=.05)


def test_if_snr_uses_same_band_and_not_total_trace_noise_power():
    t = np.arange(1000)*.01
    clean = 100 + 2*np.cos(2*np.pi*t+.3)
    noise = .1*np.cos(2*np.pi*t-.5) + 20*np.cos(4*np.pi*t)
    before = (clean.copy(), noise.copy())
    result = measure_readout_if([clean], [noise], t, 1., 0.)
    assert result['if_snr_db_per_path'] == pytest.approx([20*np.log10(20)])
    from rydberg_aoa.simulation.noise_campaign import trace_snr_db
    trace = trace_snr_db([clean], [noise], t, 0.)[0][0]
    assert trace < -19 and result['if_snr_db_per_path'][0] > 26
    assert np.array_equal(clean, before[0]) and np.array_equal(noise, before[1])


def test_pooled_power_not_mean_db_and_zero_trials_not_dropped():
    value, why = pooled_if_snr([[0., 0.], [0., 2.], [None, None]])
    assert value == pytest.approx(10*np.log10(2/101)) and why is None
    assert value != pytest.approx(-10.)
    assert pooled_if_snr([[0., None]]) == (None, 'zero_additive_if_power')
    assert pooled_if_snr([[None, 0.]]) == (None, 'zero_clean_if_power')
    assert pooled_if_snr([[0., 0.], None]) == (None, 'missing_additive_if_evidence')
    assert pooled_if_snr([[600., -600.]])[0] == 12000.


@pytest.mark.parametrize('bad', [[], [[True, 0.]], [[np.nan, 0.]], [['1', 0.]], [[1+0j, 0.]], [[10**1000, 0.]], [[0.]]])
def test_power_evidence_fails_loud(bad):
    with pytest.raises(PhysicsCalculationError):
        pooled_if_snr(bad)


@pytest.mark.parametrize('kind', ['complex', 'bool', 'object', 'nan', 'shape', 'short', 'grid', 'zero_if'])
def test_invalid_if_inputs_never_fallback(kind):
    t = np.arange(100)*.1
    clean = np.ones((2, 100)); noise = np.zeros_like(clean); frequency = .2
    if kind == 'complex': clean = clean.astype(complex)
    if kind == 'bool': clean = clean.astype(bool)
    if kind == 'object': clean = clean.astype(object)
    if kind == 'nan': clean[0, 3] = np.nan
    if kind == 'shape': noise = noise[:, :-1]
    if kind == 'short': t = t[:5]; clean = clean[:, :5]; noise = noise[:, :5]
    if kind == 'grid': t[6] = t[5]
    if kind == 'zero_if': frequency = 0.
    with pytest.raises(PhysicsCalculationError):
        measure_readout_if(clean, noise, t, frequency, 0.)


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('frequency', [-.17, .17])
@pytest.mark.parametrize('reference', [0, 2])
def test_relative_phase_cancels_common_gain_time_origin_both_rf_directions(direction, frequency, reference):
    # Independent x-baseline geometry oracle, using the SIGNAL carrier.
    x = np.arange(3)*.001
    phi = 2*np.pi*132.6491e9/299792458*x*np.sin(np.deg2rad(-30.))
    t = np.linspace(0, 90, 901)
    traces = [7 + 3*np.cos(direction*2*np.pi*frequency*t+p+.93) for p in phi]
    raw = np.array([extract_if_component(y, t, frequency, 23.4)[0] for y in traces])
    canonical = raw.conj() if direction < 0 else raw
    before = canonical.copy()
    measured = relative_phase_errors(canonical, phi, reference)
    assert measured['error_deg_per_path'] == pytest.approx([0.]*3, abs=2e-12)
    assert np.array_equal(canonical, before)
    wrong = relative_phase_errors(canonical.conj(), phi, reference)
    assert max(abs(v) for v in wrong['error_deg_per_path']) > 20  # detects mirror


def test_relative_phase_keeps_differential_error_and_zero_is_undefined():
    phi = np.deg2rad([179., -179., 20.])
    a = np.exp(1j*(phi+.8+np.deg2rad([0., 3., -4.])))
    assert relative_phase_errors(a, phi, 0)['error_deg_per_path'] == pytest.approx([0., 3., -4.])
    a[1] = 0
    assert relative_phase_errors(a, phi, 0)['error_deg_per_path'][1] is None
    assert relative_phase_errors(a, phi, 1)['error_deg_per_path'] == [None]*3


@pytest.mark.parametrize('bad', [True, np.bool_(False), -1, 3, 1., '0'])
def test_invalid_phase_reference(bad):
    with pytest.raises(PhysicsCalculationError):
        relative_phase_errors([1., 1., 1.], [0., 0., 0.], bad)


def study_profile():
    data = json.loads((ROOT/'examples/noise009/prescribed_readout_statistics.json').read_text())
    data['study'].update(metrics=['readout_if_snr_db_per_path', 'relative_if_phase_rmse_deg_per_path'],
                         trials_per_point=2, phase_reference_path=1,
                         statistics={'trial_checkpoints': [1, 2], 'confidence_level': .95,
                                     'bootstrap_replicates': 40, 'seed': 75})
    return parse_run_profile(data)


def synthetic_point(profile):
    t = np.arange(1000)*.01
    phi = np.array([0., -.4, -.8])
    clean = np.array([2*np.cos(2*np.pi*t+p) for p in phi])
    config = SimpleNamespace(transient_delay_us=0.)
    nominal = {'complex_responses': 2*np.exp(1j*phi), 'element_phases_rad': phi,
               'signed_if_mhz': 1., 'time_traces': clean, 'tlist': t,
               'run_config': config, 'true_aoa': -30., 'estimated_aoa': -30.}
    trials = []
    for index, amp in enumerate([.1, .3]):
        additive = np.tile(amp*np.cos(2*np.pi*t), (3, 1))
        trial = dict(nominal, time_traces=clean+additive, clean_time_traces=clean,
                     additive_readout_noise=additive, readout_noise={'trial_index': index},
                     complex_responses=nominal['complex_responses']+amp)
        trials.append(trial)
    return nominal, dict(trials[0], noise_config=replace(profile.noise, trials=2), noise_trial_results=trials)


def test_study_report_bootstrap_and_ordinary_api_share_power_evidence():
    profile = study_profile()
    assert parse_run_profile(profile_document(profile)) == profile
    nominal, result = synthetic_point(profile)
    summary = summarize_study_point(nominal, result, profile.study)
    expected = 10*np.log10(8/(.01+.09))
    assert summary['metrics']['readout_if_snr_db_per_path'] == pytest.approx([expected]*3)
    assert summary['metrics']['relative_if_phase_rmse_deg_per_path'][1] == 0.
    ordinary = measure_sensing_result(result, 1)
    assert ordinary['pooled_readout_if_snr_db_per_path'] == pytest.approx([expected]*3)
    from dataclasses import asdict
    from hashlib import sha256
    points = []
    for i, value in enumerate(profile.study.values):
        seed = profile.study.seed if profile.study.random_policy == 'paired' else int.from_bytes(
            sha256(f'NOISE-008-v1:{profile.study.seed}:{i}'.encode()).digest()[:8], 'big') & (2**53-1)
        points.append(dict(index=i, value=value, effective_seed=seed, effective_trials=2, summary=summary))
    report = dict(schema_version=1, study=asdict(profile.study), status='completed',
                  points_requested=len(points), points_completed=len(points), points=points)
    analysis = analyze_study_statistics(json.loads(json.dumps(report)))
    estimate = analysis['checkpoints'][-1]['points'][0]['metrics']['readout_if_snr_db_per_path'][0]
    assert estimate['estimate'] == pytest.approx(expected)
    assert estimate['interval'][0] <= expected <= estimate['interval'][1]
    assert analysis['checkpoints'][-1]['points'][0]['metrics']['relative_if_phase_rmse_deg_per_path'][1]['estimate'] == 0.


def test_study_no_additive_layer_and_no_nominal_phase_calibration():
    p = study_profile(); nominal, result = synthetic_point(p)
    for trial in result['noise_trial_results']:
        trial.pop('readout_noise')
        trial['complex_responses'] = 2*np.exp(1j*(trial['element_phases_rad']+np.deg2rad([7., 0., -8.])))
    summary = summarize_study_point(nominal, result, p.study)
    assert summary['metrics']['readout_if_snr_db_per_path'] == [None]*3
    assert summary['metrics']['relative_if_phase_rmse_deg_per_path'] == pytest.approx([7., 0., 8.])


@pytest.mark.parametrize('bad', [True, -1, 99, 1., '1'])
def test_profile_reference_rejected_before_initialization(bad):
    data = profile_document(study_profile()); data['study']['phase_reference_path'] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


def test_incomplete_campaign_is_not_certified():
    p = study_profile(); _, result = synthetic_point(p)
    result['noise_trial_results'].pop()
    with pytest.raises(PhysicsCalculationError, match='incomplete'):
        measure_sensing_result(result)


@pytest.mark.parametrize('damage', ['if', 'geometry', 'order'])
def test_stale_or_reordered_measurement_evidence_rejected(damage):
    p = study_profile(); nominal, result = synthetic_point(p)
    trial = result['noise_trial_results'][1]
    if damage == 'if': trial['signed_if_mhz'] = -.5
    if damage == 'geometry': trial['element_phases_rad'] = trial['element_phases_rad']+.1
    if damage == 'order': trial['readout_noise']['trial_index'] = 0
    with pytest.raises(PhysicsCalculationError):
        measure_sensing_result(result)
    if damage != 'order':
        with pytest.raises(PhysicsCalculationError):
            summarize_study_point(nominal, result, p.study)


def test_bootstrap_keeps_valid_pooled_estimate_if_resample_has_zero_power():
    from rydberg_aoa.simulation.study_statistics import _estimate, _interval
    estimate, draws, why = _estimate([[0., None], [0., 0.]], None, np.array([[0, 0], [0, 1]]))
    assert estimate == pytest.approx(10*np.log10(2))
    assert draws is None and why == 'undefined_bootstrap_power'
    interval = _interval(estimate, draws, why, 2, study_profile().study.statistics)
    assert interval['estimate'] == estimate and interval['interval'] is None


def test_offline_phase_and_statistics_do_not_import_arc_or_solver():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('arc', 'qutip') or name.endswith(('parameters', 'atomic_config', 'noise_random')):
            raise AssertionError('offline IF statistics imported physics: '+name)
sys.meta_path.insert(0, Guard())
from rydberg_aoa.simulation.study_statistics import _estimate
from rydberg_aoa.simulation.if_measurement import relative_phase_errors, measure_readout_if
import numpy as np
assert relative_phase_errors([1,1j], [0,np.pi/2], 0)['error_deg_per_path']==[0.,0.]
assert _estimate([[0.,0.],[0.,1.]],None,np.array([[0,1]]))[0] < 0
t=np.arange(100)*.1
assert measure_readout_if([np.cos(2*np.pi*.2*t)],[np.zeros(100)],t,.2,0)['if_snr_db_per_path']==[None]
'''
    child = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr


def test_relative_phase_graph_identifies_selected_reference(monkeypatch):
    from rydberg_aoa.simulation import study_execution as engine
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    p = study_profile(); nominal, result = synthetic_point(p)
    monkeypatch.setattr(engine, '_execute_isolated', lambda point: ({}, nominal if
        all(not source.specification.enabled for source in point.noise.sources) else result))
    output = engine.execute_study(p)
    figures = create_study_figures(output)
    figure = figures['study_relative_if_phase_rmse_deg_per_path']
    assert 'Path 2 (reference)' in [line.get_label() for line in figure.axes[0].lines]
    assert any('reference path 2 is zero by definition' in text.get_text() for text in figure.texts)
    for figure in figures.values():
        plt.close(figure)


def _run_profile(profile):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    return execute_profile(profile)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_actual_profile_reports_if_measurement_without_touching_physics(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    data = json.loads((ROOT/f'examples/noise004/{mode}_pin.json').read_text())
    data['noise']['trials'] = 2
    profile = parse_run_profile(data)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        report, result = executor.submit(_run_profile, profile).result()
    measured = report['if_measurement']
    assert len(measured['trials']) == 2 and measured['hardware_calibration_verified'] is False
    assert all(v is not None and np.isfinite(v) for v in measured['pooled_readout_if_snr_db_per_path'])
    for row in measured['trials']:
        assert row['relative_phase']['error_deg_per_path'][0] == 0.
        assert np.isfinite(row['relative_phase']['error_deg_per_path']).all()
    _, _, operator = if_fit_noise_weights(result['tlist'], result['signed_if_mhz'], profile.run_config.transient_delay_us)
    assert measured['fit'] == operator
    assert report['input_provenance']['experimental_calibration_verified'] is False


def test_actual_if_study_exports_metrics_uncertainty_and_graphs(tmp_path):
    from rydberg_aoa.simulation.study_execution import execute_study
    from rydberg_aoa.simulation.study_output import save_study_results
    from rydberg_aoa.visualization.study_figures import create_study_figures
    import matplotlib.pyplot as plt
    data = json.loads((ROOT/'examples/noise011/prescribed_if_study.json').read_text())
    data['study'].update(values=[0., 1e-5], trials_per_point=2, phase_reference_path=1,
                         statistics={'trial_checkpoints': [1, 2], 'confidence_level': .95,
                                     'bootstrap_replicates': 20, 'seed': 9011})
    p = parse_run_profile(data)
    output = execute_study(p)
    saved = output.report()
    for point in saved['points']:
        assert point['summary']['metrics']['relative_if_phase_rmse_deg_per_path'][1] == 0.
    assert saved['points'][0]['summary']['metrics']['readout_if_snr_db_per_path'] == [None]*3
    assert np.isfinite(saved['points'][1]['summary']['metrics']['readout_if_snr_db_per_path']).all()
    save_study_results(tmp_path/'study', output)
    document = json.loads((tmp_path/'study/study_report.json').read_text())
    assert document['study']['phase_reference_path'] == 1
    assert analyze_study_statistics(document) == document['uncertainty']
    figures = create_study_figures(output)
    assert 'study_readout_if_snr_db_per_path' in figures
    assert 'study_relative_if_phase_rmse_deg_per_path' in figures
    phase_figure = figures['study_relative_if_phase_rmse_deg_per_path']
    assert any('reference' in line.get_label() for line in phase_figure.axes[0].lines)
    assert any('zero by definition' in text.get_text() for text in phase_figure.texts)
    for figure in figures.values():
        plt.close(figure)
