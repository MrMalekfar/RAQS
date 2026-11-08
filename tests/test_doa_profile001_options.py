"""Strict profile options and actual estimators against independent wave phases."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rydberg_aoa.config import parameters as params, doa_config
from rydberg_aoa.config.doa_options import parse_doa_options
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.simulation.doa_pipeline import estimate_doa_from_results
from tests.test_lab007_profile_integration import isolated  # explicit full state isolation

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('phase_beamforming', 'bartlett', 'music', 'mvdr', 'capon', 'esprit', 'root_music')


def declaration(method):
    result = {}
    if method not in ('esprit', 'root_music'):
        result['azimuth_scan'] = {'min_deg': -80., 'max_deg': 80., 'point_count': 641}
    if method in ('music', 'esprit', 'root_music'):
        result['n_sources'] = 1
    if method in ('mvdr', 'capon'):
        result['loading_rel'] = 1e-4
    return result


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
@pytest.mark.parametrize('method', METHODS)
def test_parse_and_thermal_resolution_preserve_options(mode, method):
    data = json.loads((ROOT / 'examples/lab007' / f'{mode}_full.json').read_text())
    data['execution'].update(doa_method=method, doa_options=declaration(method))
    before = copy.deepcopy(data)
    profile = parse_run_profile(data)
    assert data == before
    plan = profile.run_config
    resolved = plan.resolve((.01, .02, .03), .4)
    assert resolved.doa_options is plan.doa_options
    assert resolved.doa_options.estimator_kwargs(method)['elevation_deg'] == 0.


@pytest.mark.parametrize('change', ['extra', 'missing', 'search_free_grid', 'unused_loading',
                                  'bool_count', 'string_angle', 'nan_angle', 'complex_angle',
                                  'reverse', 'outside', 'one_point', 'bool_sources', 'multi_source',
                                  'negative_loading', 'bool_loading', 'overflow_loading', 'null'])
def test_invalid_or_unused_options_fail_before_initialization(change):
    method = 'music'
    value = declaration(method)
    if change == 'extra':
        value['unknown'] = 1
    elif change == 'missing':
        del value['n_sources']
    elif change == 'search_free_grid':
        method = 'esprit'
    elif change == 'unused_loading':
        value['loading_rel'] = .1
    elif change in ('bool_count', 'one_point'):
        value['azimuth_scan']['point_count'] = True if change == 'bool_count' else 1
    elif change in ('string_angle', 'nan_angle', 'complex_angle'):
        value['azimuth_scan']['min_deg'] = {'string_angle': '-80', 'nan_angle': np.nan, 'complex_angle': 1j}[change]
    elif change == 'reverse':
        value['azimuth_scan']['max_deg'] = -85.
    elif change == 'outside':
        value['azimuth_scan']['min_deg'] = -100.
    elif change in ('bool_sources', 'multi_source'):
        value['n_sources'] = True if change == 'bool_sources' else 2
    elif change in ('negative_loading', 'bool_loading', 'overflow_loading'):
        method = 'mvdr'
        value = declaration(method)
        value['loading_rel'] = {'negative_loading': -.1, 'bool_loading': True,
                                'overflow_loading': 10**1000}[change]
    else:
        value = None
    with pytest.raises(PhysicsConfigurationError):
        parse_doa_options(value, method)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('angle', [-30.25, 30.25])
def test_real_estimators_follow_explicit_options_and_independent_phase(method, angle, monkeypatch):
    frequency = 1e9
    from scipy.constants import c
    positions = np.column_stack((np.arange(6)*c/frequency/4, np.zeros((6, 2))))
    times = np.linspace(0., 60., 601)
    # Independent plane-wave phase; no production steering helper as oracle.
    spatial = 2*np.pi*frequency/c*positions[:, 0]*np.sin(np.deg2rad(angle))
    responses = np.exp(1j*spatial)
    traces = np.cos(2*np.pi*.1*times[None, :] + spatial[:, None])
    monkeypatch.setattr(params, 'RF_SIGNAL_FREQUENCY', frequency)
    monkeypatch.setattr(params, 'IF_FREQ', .1)
    monkeypatch.setattr(doa_config, 'DOA_AZIMUTH_MIN_DEG', 0.)
    monkeypatch.setattr(doa_config, 'DOA_AZIMUTH_MAX_DEG', 1.)
    monkeypatch.setattr(doa_config, 'DOA_NUM_SOURCES', 100)
    monkeypatch.setattr(doa_config, 'DOA_MVDR_LOADING_REL', .8)
    monkeypatch.setattr(doa_config, 'DOA_ELEVATION_DEG', 42.)
    options = parse_doa_options(declaration(method), method)
    result, _ = estimate_doa_from_results(positions, responses, times, traces,
                                        method=method, doa_options=options)
    assert result.azimuth_deg == pytest.approx(angle, abs=1e-6)
    assert result.diagnostics['settings_source'] == 'profile'
    settings = result.diagnostics['effective_settings']
    assert settings['elevation_deg'] == 0.
    if method in ('esprit', 'root_music'):
        assert result.spectrum is None and result.scan_grid is None
        assert 'scan_azimuth_deg' not in settings
    else:
        assert np.array_equal(result.scan_grid, np.linspace(-80., 80., 641))
        assert settings['scan_azimuth_deg'] == tuple(result.scan_grid)
    if method in ('mvdr', 'capon'):
        assert settings['loading_rel'] == 1e-4


def test_legacy_profile_uses_existing_defaults_without_numeric_change(monkeypatch):
    from rydberg_aoa.simulation.doa_pipeline import build_estimator_kwargs
    assert np.array_equal(build_estimator_kwargs('music')['scan_azimuth_deg'],
                          doa_config.azimuth_scan_grid_deg())
    for folder in ('lab006', 'lab007'):
        name = 'experimental_near_resonant.json' if folder == 'lab006' else 'experimental_full.json'
        profile = parse_run_profile(json.loads((ROOT / 'examples' / folder / name).read_text()))
        assert profile.run_config.doa_options is None


def test_explicit_historical_defaults_preserve_legacy_spectrum_bitwise(monkeypatch):
    from scipy.constants import c
    frequency = 1e9
    positions = np.column_stack((np.arange(6)*c/frequency/4, np.zeros((6, 2))))
    times = np.linspace(0., 60., 601)
    spatial = 2*np.pi*frequency/c*positions[:, 0]*np.sin(np.deg2rad(-30.))
    responses = np.exp(1j*spatial)
    traces = np.cos(2*np.pi*.1*times[None, :] + spatial[:, None])
    monkeypatch.setattr(params, 'RF_SIGNAL_FREQUENCY', frequency)
    monkeypatch.setattr(params, 'IF_FREQ', .1)
    options = parse_doa_options({'azimuth_scan': {'min_deg': -90., 'max_deg': 90., 'point_count': 181}},
                                'phase_beamforming')
    old, _ = estimate_doa_from_results(positions, responses, times, traces, method='phase_beamforming')
    new, _ = estimate_doa_from_results(positions, responses, times, traces,
                                       method='phase_beamforming', doa_options=options)
    assert old.azimuth_deg == new.azimuth_deg
    assert np.array_equal(old.spectrum, new.spectrum)
    assert np.array_equal(old.scan_grid, new.scan_grid)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_real_profile_runner_consumes_options_and_reports_effective_settings(mode, monkeypatch, isolated):
    from rydberg_aoa.simulation.profile_execution import execute_profile
    data = json.loads((ROOT / 'examples/lab007' / f'{mode}_full.json').read_text())
    data['spectrum'] = None  # isolate the real temporal/DoA chain from spectral scans
    data['execution'].update(doa_method='music', target_azimuth_deg=-30.25,
                             doa_options=declaration('music'))
    monkeypatch.setattr(doa_config, 'DOA_NUM_SOURCES', 100)
    monkeypatch.setattr(doa_config, 'DOA_ELEVATION_DEG', 42.)
    report, output = execute_profile(parse_run_profile(data))
    settings = report['sensing']['doa_effective_settings']
    assert report['sensing']['doa_settings_source'] == 'profile'
    assert settings['n_sources'] == 1
    assert settings['elevation_deg'] == 0.
    assert len(settings['scan_azimuth_deg']) == 641
    assert output['estimated_aoa'] == pytest.approx(-30.25, abs=1e-12)
    assert np.array_equal(output['angles'], np.linspace(-80., 80., 641))
    json.dumps(report)  # the effective settings must survive the CLI report format
