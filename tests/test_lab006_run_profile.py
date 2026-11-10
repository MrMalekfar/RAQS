"""Public entry-point contracts, strict profile parsing and failed-run isolation."""

import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from rydberg_aoa.config.run_profile import load_run_profile, parse_run_profile
from rydberg_aoa.simulation import profile_execution as execution
from rydberg_aoa.exceptions import PhysicsConfigurationError, PhysicsCalculationError
from tests.test_lab004_live_laboratory_runtime import live_request

ROOT = Path(__file__).resolve().parents[1]


def document(mode='experimental'):
    return json.loads((ROOT / 'examples/lab006' / f'{mode}_reference.json').read_text())


def test_profiles_are_explicit_and_detached():
    data = document()
    original = copy.deepcopy(data)
    profile = parse_run_profile(data)
    assert data == original
    data['preparation']['rydberg1_mj'] = 1.5
    assert profile.request.preparation.rydberg1_mj == -.5
    assert profile.request.selection_carrier_hz == 132649000000.
    assert profile.run_config.tlist_us[-1] == 60.
    assert len(profile.run_config.tlist_us) == 601
    assert len(profile.run_config.positions_m) == 3


@pytest.mark.parametrize('bad', [True, '1', None, 3, 1.0])
def test_schema_version_is_exact_supported_integer(bad):
    data = document()
    data['schema_version'] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('change', ['no_mode', 'no_mj', 'extra_field', 'unknown_method', 'bad_axis',
                                    'dc_if', 'nyquist', 'too_short', 'wrong_signal_q', 'nonarray_positions'])
def test_invalid_physical_contract_rejected_before_execution(change):
    data = document()
    if change == 'no_mode':
        del data['mode']
    elif change == 'no_mj':
        del data['preparation']['rydberg1_mj']
    elif change == 'extra_field':
        data['fields']['probe']['power_w'] = 1
    elif change == 'unknown_method':
        data['execution']['doa_method'] = 'guess'
    elif change == 'bad_axis':
        data['preparation']['quantization_axis'] = [0, 0, 2]
    elif change == 'dc_if':
        data['fields']['signal']['frequency_hz'] = data['fields']['lo']['frequency_hz']
    elif change == 'nyquist':
        data['fields']['signal']['frequency_hz'] += 10e6
    elif change == 'too_short':
        data['execution']['duration_us'] = 21
    elif change == 'wrong_signal_q':
        data['fields']['signal']['polarization'] = 1
    else:
        data['execution']['positions_m'] = '0,1'
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('bad', [True, '132649000000', float('nan'), float('inf'), 10**1000])
def test_applied_rf_no_coercion_or_overflow_fallback(bad):
    data = document()
    data['fields']['lo']['frequency_hz'] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('text', ['{"mode":"experimental","mode":"prescribed"}',
                                 '{"mode":NaN}', '{"mode":Infinity}', '{broken'])
def test_invalid_json_is_typed_and_not_repaired(tmp_path, text):
    file = tmp_path / 'bad.json'
    file.write_text(text)
    with pytest.raises(PhysicsConfigurationError):
        load_run_profile(file)


def test_missing_file_preserves_cause(tmp_path):
    with pytest.raises(PhysicsConfigurationError) as error:
        load_run_profile(tmp_path / 'absent.json')
    assert isinstance(error.value.__cause__, OSError)


def test_unrepresentable_sample_count_is_rejected_without_allocation():
    data = document()
    data['execution']['sample_count'] = 10**1000
    with pytest.raises(PhysicsConfigurationError, match='array-index capacity'):
        parse_run_profile(data)


def test_prescribed_declares_levels_without_bounds_or_search():
    profile = parse_run_profile(document('prescribed'))
    assert profile.request.level_mappings()['rydberg1']['mj'] == -.5
    assert profile.request.level_mappings()['rydberg2']['n'] == 27
    assert profile.run_config is None


@pytest.mark.parametrize('change', ['missing_mj', 'string_j', 'other_policy', 'rf_q', 'extra_execution'])
def test_prescribed_ambiguous_contracts_rejected(change):
    data = document('prescribed')
    if change == 'missing_mj':
        del data['ladder']['rydberg1']['mj']
    elif change == 'string_j':
        data['ladder']['excited']['j'] = '1.5'
    elif change == 'other_policy':
        data['optical_energy_convention'] = 'low_level_hfs'
    elif change == 'rf_q':
        data['fields']['lo']['polarization'] = data['fields']['signal']['polarization'] = 1
    else:
        data['execution'] = {}
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


def test_prescribed_sensing_is_explicitly_unavailable_before_importing_physics():
    with pytest.raises(PhysicsConfigurationError, match='diagnostics-only'):
        execution.execute_profile(parse_run_profile(document('prescribed')))


def test_main_import_has_no_log_plot_or_physics_side_effect(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('main import must not create logs or figures')
    monkeypatch.setattr('logging.FileHandler', forbidden)
    module = importlib.import_module('main')
    importlib.reload(module)


@pytest.mark.parametrize('entry', ['main', 'cli'])
@pytest.mark.parametrize('case', ['missing', 'conflict', 'abbreviated_conflict', 'invalid'])
def test_public_entry_rejects_before_execution(monkeypatch, tmp_path, entry, case):
    import cli
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(document()))
    args = ['--profile', str(path), '--diagnostics-only', '--no-plots']
    if case == 'missing':
        args = []
    elif case == 'conflict':
        args += ['--rf-frequency=132000000000']
    elif case == 'abbreviated_conflict':
        args += ['--rf-freq=132000000000']
    else:
        path.write_text('{"mode": "experimental"}')
    monkeypatch.setattr(sys, 'argv', [f'{entry}.py', *args])
    def forbidden(*args, **kwargs):
        pytest.fail('invalid public input must not reach execution')
    monkeypatch.setattr(execution, 'execute_profile', forbidden)
    with pytest.raises(PhysicsConfigurationError):
        importlib.import_module(entry).main()


def test_failed_initialization_restores_every_applied_view_and_nested_alias(monkeypatch):
    from rydberg_aoa.config import atomic_config as ac, parameters as params
    from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state
    state, pipeline = snapshot_published_atomic_state(), ac._snapshot_pipeline_transaction()
    names = ('RF_LO_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_FREQUENCY', 'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')
    scalars = {key: getattr(params, key) for key in names}
    roots = (ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS)
    before = copy.deepcopy(roots)
    probe, coupling = ac.LASER_PARAMS['probe'], ac.LASER_PARAMS['coupling']
    original = PhysicsCalculationError('injected ARC failure')
    def fail(**kwargs):
        assert ac.RF_PARAMS['frequency_hz'] == 132648900000.
        assert ac.LASER_PARAMS['coupling']['wavelength_nm'] == 483.3
        raise original
    monkeypatch.setattr(ac, 'initialize_atomic_physics', fail)
    ac.reset_atomic_physics()
    try:
        data = document()
        data['fields']['lo']['frequency_hz'] = 132648900000.
        data['fields']['coupling']['wavelength_nm'] = 483.3
        with pytest.raises(PhysicsCalculationError) as error:
            execution.execute_profile(parse_run_profile(data), diagnostics_only=True)
        assert error.value is original
        assert roots == before
        assert ac.LASER_PARAMS['probe'] is probe and ac.LASER_PARAMS['coupling'] is coupling
        assert {key: getattr(params, key) for key in names} == scalars
        assert not ac.is_atomic_physics_initialized()
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        restore_published_atomic_state(state)
        for key, value in scalars.items():
            setattr(params, key, value)


def test_plot_has_only_actual_data_and_handles_search_free_doa(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = dict(tlist=np.arange(10.), time_traces=[np.arange(10.)], xf_0=np.arange(5.),
                  yf_0=np.array([1, 2j, 3, 4j, 5]), complex_responses=np.array([1, 1j]),
                  power=None, angles=None, estimated_aoa=-30.)
    figure = execution.plot_profile_sensing(result)
    try:
        assert len(figure.axes) == 4
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), result['time_traces'][0])
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), np.abs(result['yf_0']))
        assert len(figure.axes[3].lines) == 0
        assert '-30' in figure.axes[3].texts[0].get_text()
    finally:
        plt.close(figure)


def test_worker_failure_after_real_initialization_restores_inputs_and_caches(live_request, monkeypatch):
    from rydberg_aoa.config import atomic_config as ac, parameters as params
    from rydberg_aoa.simulation.runner import SimulationRunner
    from tests.test_lab004_laboratory_runtime import applied_snapshot, RF_SCALARS, CACHES
    scalars = {name: getattr(params, name) for name in RF_SCALARS}
    before = applied_snapshot()
    cache_snapshots = {name: dict(getattr(ac, name)) for name in CACHES}
    original = PhysicsCalculationError('injected failure after valid ARC initialization')
    def fail(self, request, config):
        ac.initialize_atomic_physics(resolution_request=request)
        assert ac.is_atomic_physics_initialized()
        assert ac.get_laboratory_runtime().selection.winner.rydberg1.mj == -.5
        raise original
    monkeypatch.setattr(SimulationRunner, 'run_laboratory_simulation', fail)
    data = document()
    data['fields']['lo']['frequency_hz'] -= 100000.
    data['fields']['signal']['frequency_hz'] -= 100000.
    data['fields']['coupling']['wavelength_nm'] = 483.3
    with pytest.raises(PhysicsCalculationError) as error:
        execution.execute_profile(parse_run_profile(data))
    assert error.value is original
    assert applied_snapshot() == before
    assert {name: getattr(params, name) for name in RF_SCALARS} == scalars
    assert all(getattr(ac, name) == cache_snapshots[name] for name in CACHES)
    assert not ac.is_atomic_physics_initialized()
    assert ac._INITIALIZED_REQUEST is ac._INITIALIZED_RESOLUTION_IDENTITY is None


def test_existing_published_run_is_not_reset_or_reconfigured(monkeypatch):
    from rydberg_aoa.config import atomic_config as ac
    monkeypatch.setattr(ac, '_ATOMIC_PHYSICS_INITIALIZED', True)
    before = copy.deepcopy((ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS))
    with pytest.raises(PhysicsConfigurationError, match='already published'):
        execution.execute_profile(parse_run_profile(document()), diagnostics_only=True)
    assert ac.is_atomic_physics_initialized()
    assert (ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS) == before
