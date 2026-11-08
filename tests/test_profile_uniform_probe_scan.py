"""Uniform spectrum declarations and genuine profile entry-point consumers."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.exceptions import PhysicsConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def document(mode='prescribed'):
    return json.loads((ROOT / 'examples/lab007' / f'{mode}_uniform_scan.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_exact_uniform_grid_and_immutable_input(mode):
    data = document(mode)
    data['spectrum']['probe_scan'] = dict(start_offset_mhz=-2, stop_offset_mhz=2, point_count=5)
    before = copy.deepcopy(data)
    profile = parse_run_profile(data)
    assert profile.spectrum.probe_offsets_mhz == (-2., -1., 0., 1., 2.)
    assert parse_run_profile(data).spectrum == profile.spectrum
    assert data == before


def test_uniform_and_manual_declarations_resolve_identically():
    data = document()
    data['spectrum']['probe_scan'] = dict(start_offset_mhz=np.float64(-1), stop_offset_mhz=1., point_count=np.int64(3))
    uniform = parse_run_profile(data)
    del data['spectrum']['probe_scan']
    data['spectrum']['probe_offsets_mhz'] = [-1., 0., 1.]
    assert parse_run_profile(data) == uniform


@pytest.mark.parametrize('key,value', [
    ('start_offset_mhz', True), ('start_offset_mhz', '0'),
    ('start_offset_mhz', np.nan), ('start_offset_mhz', np.inf),
    ('start_offset_mhz', 1j), ('start_offset_mhz', np.array(0.)),
    ('stop_offset_mhz', -20), ('stop_offset_mhz', -21),
    ('stop_offset_mhz', np.nan), ('stop_offset_mhz', False),
    ('point_count', 0), ('point_count', 1), ('point_count', -1),
    ('point_count', 5.), ('point_count', '5'), ('point_count', True),
    ('point_count', np.bool_(False)), ('point_count', np.array(5)),
    ('point_count', 10**100),
])
def test_invalid_declaration_rejected(key, value):
    data = document()
    data['spectrum']['probe_scan'][key] = value
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('case', ['both', 'neither', 'extra', 'missing', 'not_object'])
def test_strict_schema(case):
    data = document()
    spec = data['spectrum']
    if case == 'both':
        spec['probe_offsets_mhz'] = [0.]
    elif case == 'neither':
        del spec['probe_scan']
    elif case == 'extra':
        spec['probe_scan']['step_mhz'] = .1
    elif case == 'missing':
        del spec['probe_scan']['point_count']
    else:
        spec['probe_scan'] = [0, 1, 2]
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


def test_grid_arithmetic_overflow_has_cause():
    data = document()
    data['spectrum']['probe_scan'] = dict(start_offset_mhz=-1e308, stop_offset_mhz=1e308, point_count=3)
    with pytest.raises(PhysicsConfigurationError) as caught:
        parse_run_profile(data)
    assert isinstance(caught.value.__cause__, FloatingPointError)


def test_finite_endpoints_cannot_hide_duplicate_grid_points():
    data = document()
    data['spectrum']['probe_scan'] = dict(start_offset_mhz=1., stop_offset_mhz=np.nextafter(1., 2.), point_count=3)
    with pytest.raises(PhysicsConfigurationError, match='strictly increasing'):
        parse_run_profile(data)


def test_allocation_failure_is_not_a_shorter_grid(monkeypatch):
    from rydberg_aoa.config import profile_effects
    failure = MemoryError('injected allocation failure')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(profile_effects.np, 'linspace', fail)
    with pytest.raises(PhysicsConfigurationError) as caught:
        parse_run_profile(document())
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize('offsets', [[0.], [2., -1., 2.]])
def test_manual_offsets_contract_is_preserved(offsets):
    data = document()
    del data['spectrum']['probe_scan']
    data['spectrum']['probe_offsets_mhz'] = offsets
    assert parse_run_profile(data).spectrum.probe_offsets_mhz == tuple(offsets)


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_annotated_new_examples_match_json(mode):
    path = ROOT / 'examples/lab007' / f'{mode}_uniform_scan.annotated.jsonc'
    clean = '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                      if not line.lstrip().startswith('//'))
    assert json.loads(clean) == document(mode)
    assert len(parse_run_profile(document(mode)).spectrum.probe_offsets_mhz) == 201


def test_uniform_parser_has_no_atomic_imports():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('arc', 'rydberg_aoa.config.parameters', 'rydberg_aoa.config.atomic_config'):
            raise AssertionError('Unexpected atomic import: ' + fullname)
sys.meta_path.insert(0, Guard())
from rydberg_aoa.config.run_profile import load_run_profile
for mode in ('prescribed', 'experimental'):
    load_run_profile('examples/lab007/' + mode + '_uniform_scan.json')
'''
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_live_uniform_grid_is_computed_and_saved(tmp_path, mode):
    data = document(mode)
    data['spectrum']['probe_scan'] = dict(start_offset_mhz=-.5, stop_offset_mhz=.5, point_count=5)
    # Own an explicit estimator reference, independent of editable example preferences.
    data['execution']['doa_method'] = 'phase_beamforming'
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(data), encoding='utf-8')
    process = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(profile), '--no-plots',
                              '--output-dir', str(tmp_path)], cwd=ROOT, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    expected = [-.5, -.25, 0., .25, .5]
    assert report['spectrum_declaration']['probe_offsets_mhz'] == expected
    assert report['applied_fields'] == data['fields']
    assert report['mode'] == mode and report['spectrum_computed']
    assert report['sensing']['estimated_aoa_deg'] == -30.
    with np.load(tmp_path / 'simulation_arrays.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['spectrum_probe_offsets_mhz'], expected)
        for key in ('rf_off', 'rf_on', 'rf_off_analytical', 'rf_on_analytical'):
            assert arrays['spectrum_' + key].shape == (5,)
            assert np.isfinite(arrays['spectrum_' + key]).all()
        assert not np.array_equal(arrays['spectrum_rf_off'], arrays['spectrum_rf_on'])
        assert arrays['time_traces'].shape == (3, 601)
    if mode == 'prescribed':
        assert 'Scanning RF' not in process.stdout
