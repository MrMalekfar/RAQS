"""NOISE-001 switches, strict declarations and spawn-independent streams."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.config.run_profile import load_run_profile, parse_run_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.simulation.noise_random import NoiseRandomPlan
from rydberg_aoa.simulation.profile_execution import execute_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated():
    # Import only in the parent fixture: RNG spawn children need no ARC.
    from rydberg_aoa.config import atomic_config as ac, parameters as params
    from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state
    state, pipeline = snapshot_published_atomic_state(), ac._snapshot_pipeline_transaction()
    names = ('RF_LO_FREQUENCY', 'RF_LO_E_FIELD', 'RF_SIGNAL_FREQUENCY',
             'RF_SIGNAL_E_FIELD', 'RF_FREQUENCY', 'IF_FREQ')
    scalars = {key: getattr(params, key) for key in names}
    ac.reset_atomic_physics()
    try:
        yield
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        for key, value in scalars.items():
            setattr(params, key, value)
        restore_published_atomic_state(state)


def block(*, master=True, source=True):
    return {'schema_version': 1, 'enabled': master, 'seed': 20261012, 'trials': 3,
            'sources': {'readout_gaussian': {'enabled': source, 'sigma_observable': .001}}}


def _sample_key(key):
    trial, path = key
    rng = NoiseRandomPlan(parse_noise_config(block())).generator(
        'readout_gaussian', trial=trial, path=path)
    return rng.standard_normal(32)


@pytest.mark.parametrize('master,source,active', [(True, True, True), (False, True, False),
                                                  (True, False, False), (False, False, False)])
def test_independent_source_and_master_switch(master, source, active):
    data = block(master=master, source=source)
    before = copy.deepcopy(data)
    config = parse_noise_config(data)
    assert bool(config.active_sources) is active
    assert config.sources[0].specification.sigma_observable == .001
    assert data == before


@pytest.mark.parametrize('bad', [True, np.bool_(False), '0.1', None, 1j, np.array(.1),
                                float('nan'), float('inf'), -.1, 10**1000])
def test_sigma_rejects_invalid_input_without_coercion(bad):
    data = block()
    data['sources']['readout_gaussian']['sigma_observable'] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


@pytest.mark.parametrize('field,bad', [('enabled', 1), ('enabled', np.bool_(True)),
    ('seed', True), ('seed', -1), ('seed', 2.), ('seed', '2'),
    ('trials', 0), ('trials', True), ('trials', 1.2), ('schema_version', True),
    ('schema_version', 2)])
def test_invalid_block_fields(field, bad):
    data = block()
    data[field] = bad
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


@pytest.mark.parametrize('change', ['unknown_source', 'empty_sources', 'source_extra',
                                  'block_extra', 'missing_seed', 'source_flag', 'null_sources'])
def test_strict_keys_and_sources_even_when_disabled(change):
    data = block(master=False, source=False)
    if change == 'unknown_source':
        data['sources']['shot_noise'] = {'enabled': False}
    elif change == 'empty_sources':
        data['sources'] = {}
    elif change == 'null_sources':
        data['sources'] = None
    elif change == 'source_extra':
        data['sources']['readout_gaussian']['snr_db'] = 20
    elif change == 'source_flag':
        data['sources']['readout_gaussian']['enabled'] = 0
    elif change == 'block_extra':
        data['snr_db'] = 20
    else:
        del data['seed']
    with pytest.raises(PhysicsConfigurationError):
        parse_noise_config(data)


def test_immutable_detached_picklable_configuration():
    data = block()
    config = parse_noise_config(data)
    data['sources']['readout_gaussian']['sigma_observable'] = 10
    assert config.sources[0].specification.sigma_observable == .001
    with pytest.raises(FrozenInstanceError):
        config.sources[0].specification.enabled = False
    assert pickle.loads(pickle.dumps(config)) == config
    assert parse_noise_config({**block(), 'seed': np.int64(4) }).seed == 4


def test_zero_sigma_is_valid_not_an_invented_floor():
    data = block()
    data['sources']['readout_gaussian']['sigma_observable'] = 0
    assert parse_noise_config(data).sources[0].specification.sigma_observable == 0.


@pytest.mark.parametrize('master,source', [(False, True), (True, False), (False, False)])
def test_disabled_path_constructs_no_rng_and_preserves_global_rng(monkeypatch, master, source):
    plan = NoiseRandomPlan(parse_noise_config(block(master=master, source=source)))
    state = np.random.get_state()

    def forbidden(*args, **kwargs):
        pytest.fail('disabled path must not construct or consume RNG')

    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    monkeypatch.setattr(np.random, 'Generator', forbidden)
    assert plan.generator('readout_gaussian', trial=0, path=0) is None
    assert not plan.metadata()['active_sources']
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_stable_namespaces_and_dispatch_order():
    keys = [(0, None), (0, 0), (0, 1), (1, 0)]
    forward = [_sample_key(key) for key in keys]
    reverse = list(reversed([_sample_key(key) for key in reversed(keys)]))
    for a, b in zip(forward, reverse):
        assert np.array_equal(a, b)
    for i, a in enumerate(forward):
        for b in forward[i+1:]:
            assert not np.array_equal(a, b)


def test_stream_uses_documented_seedsequence_key_independent_oracle():
    actual = _sample_key((2, 4))
    reference = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(20261012, spawn_key=(1, 1, 2, 1, 4))))
    assert np.array_equal(actual, reference.standard_normal(32))


def test_real_spawn_matches_serial_streams():
    keys = [(0, 0), (1, 1), (0, None)]
    serial = [_sample_key(key) for key in keys]
    with multiprocessing.get_context('spawn').Pool(processes=2) as pool:
        parallel = pool.map_async(_sample_key, list(reversed(keys))).get(timeout=30)
    for a, b in zip(serial, reversed(parallel)):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('trial,path', [(True, 0), (-1, 0), (3, 0), (0, True), (0, -1), (0, .5)])
def test_invalid_stream_key_rejected_even_when_disabled(trial, path):
    plan = NoiseRandomPlan(parse_noise_config(block(master=False)))
    with pytest.raises(PhysicsConfigurationError):
        plan.generator('readout_gaussian', trial=trial, path=path)


@pytest.mark.parametrize('mode', ['experimental', 'prescribed'])
@pytest.mark.parametrize('folder', ['lab006', 'lab007'])
def test_optional_block_preserves_physics_request_and_execution(mode, folder):
    suffix = 'reference' if folder == 'lab006' else 'full'
    data = json.loads((ROOT / f'examples/{folder}/{mode}_{suffix}.json').read_text())
    old = parse_run_profile(data)
    original = copy.deepcopy(data)
    data['noise'] = block(master=False)
    new = parse_run_profile(data)
    assert old.noise is None
    assert replace(new, noise=None) == old
    assert new.noise.sources[0].specification.enabled is True
    assert {k: v for k, v in data.items() if k != 'noise'} == original


@pytest.mark.parametrize('mode', ['experimental', 'prescribed'])
@pytest.mark.parametrize('diagnostics_only', [False, True])
def test_invalid_noise_config_fails_before_configure_or_initialization(monkeypatch, mode, diagnostics_only):
    data = json.loads((ROOT / f'examples/lab007/{mode}_full.json').read_text())
    data['noise'] = block()
    profile = replace(parse_run_profile(data), noise=object())

    def forbidden(*args, **kwargs):
        pytest.fail('invalid noise must fail before physics writes')

    monkeypatch.setattr('rydberg_aoa.config.atomic_config.initialize_atomic_physics', forbidden)
    monkeypatch.setattr('rydberg_aoa.config.rf_configuration.configure_rf_fields', forbidden)
    with pytest.raises(PhysicsConfigurationError, match='NOISE-001'):
        execute_profile(profile, diagnostics_only=diagnostics_only)


def test_duplicate_noise_keys_are_rejected_by_file_loader(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"noise":{"enabled":true,"enabled":false}}')
    with pytest.raises(PhysicsConfigurationError, match='duplicate JSON key'):
        load_run_profile(path)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_real_disabled_execution_matches_clean_physics_and_report(isolated, monkeypatch, mode):
    from rydberg_aoa.config import atomic_config as ac
    data = json.loads((ROOT / f'examples/lab007/{mode}_full.json').read_text())
    clean, result = execute_profile(parse_run_profile(data), diagnostics_only=True)
    assert result is None
    ac.reset_atomic_physics()
    data['noise'] = block(master=False, source=True)

    def forbidden(*args, **kwargs):
        pytest.fail('disabled noise execution must never initialize an RNG')

    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    monkeypatch.setattr(np.random, 'Generator', forbidden)
    disabled, result = execute_profile(parse_run_profile(data), diagnostics_only=True)
    assert result is None
    assert disabled['noise_status'] == 'configured_disabled'
    assert disabled['noise_included'] is False
    assert disabled['noise_random_plan']['active_sources'] == ()
    for key in ('noise_declaration', 'noise_status', 'noise_random_plan'):
        disabled.pop(key)
    assert disabled == clean


def test_parse_and_invalid_execution_guard_without_atomic_imports():
    code = '''
import importlib.abc, json, sys
from dataclasses import replace
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('arc', 'rydberg_aoa.config.parameters', 'rydberg_aoa.config.atomic_config'):
            raise AssertionError('Unexpected atomic import: ' + fullname)
sys.meta_path.insert(0, Guard())
from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.simulation.profile_execution import execute_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError
with open('examples/lab007/prescribed_full.json') as file:
    data = json.load(file)
data['noise'] = {'schema_version':1, 'enabled':True, 'seed':4, 'trials':3,
    'sources':{'readout_gaussian':{'enabled':True, 'sigma_observable':.001}}}
profile = replace(parse_run_profile(data), noise=object())
try:
    execute_profile(profile)
except PhysicsConfigurationError as error:
    assert 'NOISE-001' in str(error)
else:
    raise AssertionError('Unsupported active noise was accepted')
'''
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


def test_direct_prescribed_runner_cannot_bypass_invalid_config_guard(monkeypatch):
    from rydberg_aoa.simulation.runner import SimulationRunner
    data = json.loads((ROOT / 'examples/lab007/prescribed_full.json').read_text())
    data['noise'] = block()
    profile = replace(parse_run_profile(data), noise=object())

    def forbidden(*args, **kwargs):
        pytest.fail('noise guard must precede model/runtime checks')

    monkeypatch.setattr(SimulationRunner, '_validate_explicit_sensing', forbidden)
    with pytest.raises(PhysicsConfigurationError, match='NOISE-001'):
        SimulationRunner().run_prescribed_simulation(profile, None)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_direct_spectrum_rejects_invalid_noise_config(mode):
    from rydberg_aoa.simulation.profile_spectrum import run_profile_spectrum
    data = json.loads((ROOT / f'examples/lab007/{mode}_full.json').read_text())
    data['noise'] = block()
    # No atomic runtime or medium exists: invalid declaration fails first.
    with pytest.raises(PhysicsConfigurationError, match='NOISE-001'):
        run_profile_spectrum(replace(parse_run_profile(data), noise=object()), None, None, None)
