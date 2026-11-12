"""Template density preservation and genuine nonuniform spectrum execution."""
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
ANCHORS = [-5., -2., -1., -.5, 0., .5, 1., 2., 5.]


def document(mode='prescribed'):
    return json.loads((ROOT / 'examples/lab007' / f'{mode}_nonuniform_scan.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_original_nine_offsets_are_recovered_exactly(mode):
    data = document(mode)
    original = json.loads((ROOT / 'examples/lab007' / f'{mode}_full.json').read_text(encoding='utf-8'))
    assert data['spectrum']['probe_scan']['anchor_offsets_mhz'] == original['spectrum']['probe_offsets_mhz'] == ANCHORS
    data['spectrum']['probe_scan']['point_count'] = 9
    before = copy.deepcopy(data)
    assert parse_run_profile(data) == parse_run_profile(original)
    assert data == before


def test_seventeen_offsets_follow_original_segment_density():
    data = document()
    data['spectrum']['probe_scan']['point_count'] = np.int64(17)
    expected = (-5., -3.5, -2., -1.5, -1., -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.5, 2., 3.5, 5.)
    assert parse_run_profile(data).spectrum.probe_offsets_mhz == expected


@pytest.mark.parametrize('count', [2, 4, 10, 201, 401])
def test_requested_count_endpoints_order_and_determinism(count):
    data = document()
    data['spectrum']['probe_scan']['point_count'] = count
    grid = parse_run_profile(data).spectrum.probe_offsets_mhz
    assert len(grid) == count and grid[0] == -5. and grid[-1] == 5.
    assert np.isfinite(grid).all() and (np.diff(grid) > 0).all()
    assert grid == parse_run_profile(data).spectrum.probe_offsets_mhz
    if count in (201, 401):
        stride = (count - 1) // 8
        assert grid[::stride] == tuple(ANCHORS)
        assert grid[1] - grid[0] > grid[count//2 + 1] - grid[count//2]


@pytest.mark.parametrize('anchors', [[], [0.], [0., 0.], [1., 0.], [0., True],
                                    [0., '1'], [0., 1j], [0., np.nan], [0., np.inf],
                                    [0., np.array(1.)], '0,1', None])
def test_invalid_templates_are_loud(anchors):
    data = document()
    data['spectrum']['probe_scan']['anchor_offsets_mhz'] = anchors
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('count', [True, np.bool_(False), '17', 17., 0, 1, -1, np.inf, np.array(17), 10**100])
def test_invalid_count_is_loud(count):
    data = document()
    data['spectrum']['probe_scan']['point_count'] = count
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


@pytest.mark.parametrize('change', ['both_grids', 'mixed_scan', 'extra', 'missing'])
def test_ambiguous_or_partial_schema_rejected(change):
    data = document()
    scan = data['spectrum']['probe_scan']
    if change == 'both_grids':
        data['spectrum']['probe_offsets_mhz'] = [0.]
    elif change == 'mixed_scan':
        scan.update(start_offset_mhz=-5., stop_offset_mhz=5.)
    elif change == 'extra':
        scan['smoothing'] = True
    else:
        del scan['point_count']
    with pytest.raises(PhysicsConfigurationError):
        parse_run_profile(data)


def test_generated_overflow_keeps_cause():
    data = document()
    data['spectrum']['probe_scan'] = dict(anchor_offsets_mhz=[-1e308, 1e308], point_count=3)
    with pytest.raises(PhysicsConfigurationError) as caught:
        parse_run_profile(data)
    assert isinstance(caught.value.__cause__, FloatingPointError)


def test_repeated_generated_values_are_not_repaired():
    data = document()
    data['spectrum']['probe_scan'] = dict(anchor_offsets_mhz=[1., np.nextafter(1., 2.)], point_count=3)
    with pytest.raises(PhysicsConfigurationError, match='strictly increasing'):
        parse_run_profile(data)


def test_allocation_failure_is_not_a_different_count(monkeypatch):
    from rydberg_aoa.config import profile_effects
    failure = MemoryError('injected')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(profile_effects.np, 'linspace', fail)
    with pytest.raises(PhysicsConfigurationError) as caught:
        parse_run_profile(document())
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_annotated_example_is_exact(mode):
    path = ROOT / 'examples/lab007' / f'{mode}_nonuniform_scan.annotated.jsonc'
    clean = '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                      if not line.lstrip().startswith('//'))
    assert json.loads(clean) == document(mode)


def test_parser_does_not_import_atomic_physics():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('arc', 'rydberg_aoa.config.parameters', 'rydberg_aoa.config.atomic_config'):
            raise AssertionError('Unexpected atomic import: ' + fullname)
sys.meta_path.insert(0, Guard())
from rydberg_aoa.config.run_profile import load_run_profile
for mode in ('prescribed', 'experimental'):
    load_run_profile('examples/lab007/' + mode + '_nonuniform_scan.json')
'''
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_live_nonuniform_grid_reaches_spectrum_and_saved_arrays(tmp_path, mode):
    data = document(mode)
    data['spectrum']['probe_scan'] = dict(anchor_offsets_mhz=[-.5, 0., .25], point_count=5)
    data['execution']['doa_method'] = 'phase_beamforming'
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(data), encoding='utf-8')
    process = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(profile), '--no-plots',
                              '--output-dir', str(tmp_path / 'run')], cwd=ROOT, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run' / 'run_report.json').read_text(encoding='utf-8'))
    expected = [-.5, -.25, 0., .125, .25]
    assert report['spectrum_declaration']['probe_offsets_mhz'] == expected
    assert report['applied_fields'] == data['fields']
    assert report['mode'] == mode and report['spectrum_computed']
    assert report['sensing']['estimated_aoa_deg'] == -30.
    with np.load(tmp_path / 'run' / 'simulation_arrays.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['spectrum_probe_offsets_mhz'], expected)
        for key in ('rf_off', 'rf_on', 'rf_off_analytical', 'rf_on_analytical'):
            assert arrays['spectrum_' + key].shape == (5,)
            assert np.isfinite(arrays['spectrum_' + key]).all()
        assert not np.array_equal(arrays['spectrum_rf_off'], arrays['spectrum_rf_on'])
        assert arrays['time_traces'].shape == (3, 601)
    if mode == 'prescribed':
        assert 'Scanning RF' not in process.stdout
