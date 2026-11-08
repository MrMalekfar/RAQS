"""Actual ARC initialization and canonical profile rates at the diagnostic entry point.

Transport spies stop before campaigns; the final test runs the real Case A worker
and production anchor. No convergence claim is inferred from the spy tests.
"""

import copy
from dataclasses import replace
import importlib.util
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rydberg_aoa.config import atomic_config as ac
from rydberg_aoa.config.constants import k_B
from rydberg_aoa.config.run_profile import load_run_profile
from rydberg_aoa.exceptions import PhysicsCalculationError
from rydberg_aoa.simulation.laboratory_execution import LaboratoryRunConfig
from tests.test_lab007_profile_integration import isolated  # explicit state isolation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diag():
    path = ROOT / 'tests/manual_diagnostics/num_conv001_temporal_if.py'
    spec = importlib.util.spec_from_file_location('numconv_profile_connection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_path(tmp_path, kind):
    source = ROOT / ('examples/lab006/experimental_near_resonant.json'
                     if kind == 'v1' else 'examples/lab007/experimental_full.json')
    data = json.loads(source.read_text(encoding='utf-8'))
    if kind == 'v2_explicit':
        data['environment'].update(rate_source='explicit',
                                   linewidth_rates_per_us=[0., 0., 0.],
                                   linewidth_enabled=False, collision_enabled=False,
                                   transit_enabled=False)
        data['execution'].update(dephasing_rates_per_us=[.02, .03, .03],
                                 transit_rate_per_us=.5)
    path = tmp_path / f'{kind}.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


class DispatchReached(BaseException):
    """Stop before any campaign, outside main's case-failure catcher."""


@pytest.mark.parametrize('kind', ['v1', 'v2_explicit', 'v2_thermal'])
def test_entry_point_resolves_actual_profile_rates_before_dispatch(diag, isolated, tmp_path, monkeypatch, kind):
    path = profile_path(tmp_path, kind)
    profile = load_run_profile(path)
    before = copy.deepcopy(profile)
    monkeypatch.setattr('rydberg_aoa.config.run_profile.load_run_profile', lambda _: profile)
    captured = {}

    def stop(case, request, config, runtime, phases, grid):
        captured.update(config=config, request=request, runtime=runtime, phases=phases)
        raise DispatchReached()

    monkeypatch.setattr(diag, 'run_case', stop)
    with pytest.raises(DispatchReached):
        diag.main(['--profile', str(path), '--out', str(tmp_path / 'report'), '--processes', '2'])
    config = captured['config']
    assert isinstance(config, LaboratoryRunConfig)
    assert config.processes == 2
    assert profile == before  # the declared plan/fields are never rewritten
    assert captured['request'] is profile.request
    assert np.isfinite(captured['phases']).all()
    if kind != 'v2_thermal':
        assert replace(config, processes=profile.run_config.processes) == profile.run_config
    else:
        env = profile.environment
        mass = ac.ATOM.mass
        density = ac.ATOM.getNumberDensity(env.temperature_k)
        n = ac.ATOMIC_LEVELS['rydberg1']['n']
        collision = density * 1e-18 * (n / 50.)**4 * math.sqrt(8*k_B*env.temperature_k/(math.pi*mass)) / 1e6
        transit = math.sqrt(math.pi*k_B*env.temperature_k/(2*mass)) / (math.pi*profile.probe.beam_waist_mm*1e-3/2) / 1e6
        assert config.dephasing_rates_per_us == pytest.approx([v + collision for v in env.linewidth_rates_per_us])
        assert config.transit_rate_per_us == pytest.approx(transit)
        assert config.transit_rate_per_us > 0
        assert not hasattr(profile.run_config, 'dephasing_rates_per_us')
    assert not (tmp_path / 'report').exists()


def test_resolution_failure_preserves_error_and_prevents_dispatch(diag, isolated, tmp_path, monkeypatch):
    path = profile_path(tmp_path, 'v2_thermal')
    failure = PhysicsCalculationError('injected missing thermal data')
    cause = RuntimeError('atomic database unavailable')
    failure.__cause__ = cause

    def reject(*args):
        raise failure

    def forbidden(*args):
        pytest.fail('worker dispatched after rate-resolution failure')

    monkeypatch.setattr('rydberg_aoa.simulation.profile_spectrum.resolve_profile_medium', reject)
    monkeypatch.setattr(diag, 'run_case', forbidden)
    with pytest.raises(PhysicsCalculationError) as caught:
        diag.main(['--profile', str(path), '--out', str(tmp_path / 'report')])
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert not (tmp_path / 'report').exists()


def test_failed_campaign_reports_actual_thermal_budget_without_success(diag, isolated, tmp_path, monkeypatch):
    path = profile_path(tmp_path, 'v2_thermal')
    configs = []

    def reject(case, request, config, *args):
        configs.append(config)
        raise PhysicsCalculationError('intentional transport failure')

    monkeypatch.setattr(diag, 'run_case', reject)
    out = tmp_path / 'report'
    assert diag.main(['--profile', str(path), '--out', str(out)]) == 1
    report = json.loads((out / 'num_conv001_temporal_if.json').read_text(encoding='utf-8'))
    assert report['exit_status'] == 'failed'
    assert report['failures'] == ['A', 'B', 'C', 'D']
    assert all(case['status'] == 'failed' for case in report['cases'])
    assert report['production_anchor']['status'] == 'skipped'
    assert report['rate_source'] == 'thermal_budget'
    assert report['resolved_rate_budget_per_us'] == {
        'pure_dephasing': list(configs[0].dephasing_rates_per_us),
        'transit_reset': configs[0].transit_rate_per_us}
    assert report['resolved_medium']['atom_mass_kg'] == ac.ATOM.mass
    assert all(config == configs[0] for config in configs)


def test_thermal_case_a_matches_real_production_anchor(diag, isolated, tmp_path):
    # A real spawn process must import the worker by its actual module path,
    # rather than the temporary alias used by the entry-point spies above.
    diag = importlib.import_module('tests.manual_diagnostics.num_conv001_temporal_if')
    from rydberg_aoa.simulation.profile_execution import execute_profile
    from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium
    from rydberg_aoa.signal_processing.array_geometry import get_array_phases
    profile = load_run_profile(profile_path(tmp_path, 'v2_thermal'))
    execute_profile(profile, diagnostics_only=True)
    _, config = resolve_profile_medium(profile, ac)
    request = profile.request
    phases = get_array_phases(np.asarray(config.positions_m), request.signal.frequency_hz, config.target_azimuth_deg)
    summary = diag.run_case(diag.CONV_CASES[0], request, config,
                            ac.get_laboratory_runtime(), phases, config.tlist_us)
    anchor = diag.run_production_anchor(request, config, summary)
    assert summary['certified_worker_records'] is True
    assert len(summary['responses']) == len(config.positions_m)
    assert anchor['max_abs_delta_vs_case_a'] == 0.
    assert anchor['aoa_delta_vs_case_a_deg'] == 0.
