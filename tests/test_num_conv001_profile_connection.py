"""Connect declared profile rates to the released temporal execution path."""

import copy
from dataclasses import replace
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
from rydberg_aoa.simulation.profile_execution import execute_profile
from rydberg_aoa.simulation.profile_spectrum import resolve_profile_medium
from rydberg_aoa.simulation.runner import SimulationRunner
from tests.test_lab007_profile_integration import isolated  # explicit state isolation

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize('kind', ['v1', 'v2_explicit', 'v2_thermal'])
def test_entry_point_resolves_actual_profile_rates_before_dispatch(isolated, tmp_path, monkeypatch, kind):
    profile = replace(load_run_profile(profile_path(tmp_path, kind)), spectrum=None)
    before = copy.deepcopy(profile)
    captured = {}

    def capture(self, request, config, **kwargs):
        captured.update(config=config, request=request)
        if profile.environment is not None and kind == 'v2_thermal':
            captured['physical_inputs'] = (
                ac.ATOM.mass, ac.ATOM.getNumberDensity(profile.environment.temperature_k),
                ac.ATOMIC_LEVELS['rydberg1']['n'])
        raise dispatch_reached

    dispatch_reached = PhysicsCalculationError('dispatch boundary reached')
    monkeypatch.setattr(SimulationRunner, 'run_laboratory_simulation', capture)
    with pytest.raises(PhysicsCalculationError) as caught:
        execute_profile(profile)
    assert caught.value is dispatch_reached
    config = captured['config']
    assert isinstance(config, LaboratoryRunConfig)
    assert config.processes == profile.run_config.processes
    assert profile == before  # the declared plan/fields are never rewritten
    assert captured['request'] is profile.request
    if kind != 'v2_thermal':
        assert config == profile.run_config
    else:
        env = profile.environment
        mass, density, n = captured['physical_inputs']
        collision = density * 1e-18 * (n / 50.)**4 * math.sqrt(8*k_B*env.temperature_k/(math.pi*mass)) / 1e6
        transit = math.sqrt(math.pi*k_B*env.temperature_k/(2*mass)) / (math.pi*profile.probe.beam_waist_mm*1e-3/2) / 1e6
        assert config.dephasing_rates_per_us == pytest.approx([v + collision for v in env.linewidth_rates_per_us])
        assert config.transit_rate_per_us == pytest.approx(transit)
        assert config.transit_rate_per_us > 0
        assert not hasattr(profile.run_config, 'dephasing_rates_per_us')
    assert not ac.is_atomic_physics_initialized()


def test_resolution_failure_preserves_error_and_prevents_dispatch(isolated, tmp_path, monkeypatch):
    profile = load_run_profile(profile_path(tmp_path, 'v2_thermal'))
    failure = PhysicsCalculationError('injected missing thermal data')
    cause = RuntimeError('atomic database unavailable')
    failure.__cause__ = cause

    def reject(*args):
        raise failure

    def forbidden(*args, **kwargs):
        pytest.fail('worker dispatched after rate-resolution failure')

    monkeypatch.setattr('rydberg_aoa.simulation.profile_spectrum.resolve_profile_medium', reject)
    monkeypatch.setattr(SimulationRunner, 'run_laboratory_simulation', forbidden)
    with pytest.raises(PhysicsCalculationError) as caught:
        execute_profile(profile)
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert not ac.is_atomic_physics_initialized()


def test_thermal_profile_matches_direct_production_runner(isolated, tmp_path):
    profile = replace(load_run_profile(profile_path(tmp_path, 'v2_thermal')), spectrum=None)
    execute_profile(profile, diagnostics_only=True)
    medium, config = resolve_profile_medium(profile, ac)
    anchor = SimulationRunner().run_laboratory_simulation(profile.request, config)
    ac.reset_atomic_physics()
    report, result = execute_profile(profile)
    assert report['resolved_rate_budget_per_us'] == {
        'pure_dephasing': config.dephasing_rates_per_us,
        'transit_reset': config.transit_rate_per_us,
    }
    assert report['resolved_medium']['atom_mass_kg'] == medium.atom_mass_kg
    assert len(result['worker_records']) == len(config.positions_m)
    np.testing.assert_array_equal(result['complex_responses'], anchor['complex_responses'])
    assert result['estimated_aoa'] == anchor['estimated_aoa']
