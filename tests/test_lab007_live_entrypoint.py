"""Actual prescribed/experimental ARC, spectrum, certified BDF and AoA runs."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_actual_full_entry_has_spectrum_and_preserves_if_arrays(tmp_path, mode):
    source = ROOT / 'examples/lab007' / f'{mode}_full.json'
    data = json.loads(source.read_text())
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(data))
    process = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(profile), '--no-plots',
                              '--output-dir', str(tmp_path)], cwd=ROOT, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    assert report['applied_fields'] == data['fields']
    assert report['mode'] == mode and report['spectrum_computed']
    assert report['doppler_ensemble_included'] and not report['temporal_doppler_ensemble_included']
    assert report['sensing']['estimated_aoa_deg'] == report['sensing']['true_aoa_deg'] == -30.
    assert report['detunings_mhz'][1] == pytest.approx(.25, abs=1e-6)
    assert report['resolved_rate_budget_per_us']['pure_dephasing'] == report['sensing']['execution']['dephasing_rates_per_us']
    with np.load(tmp_path / 'simulation_arrays.npz', allow_pickle=False) as arrays:
        assert arrays['time_traces'].shape == (3, 601)
        assert arrays['if_snapshots'].shape[0] == 3
        assert arrays['sample_covariance'].shape == (3, 3)
        assert np.iscomplexobj(arrays['complex_responses'])
        assert np.iscomplexobj(arrays['fft_complex_path_2'])
        assert arrays['spectrum_rf_off'].shape == (2,)
        assert arrays['spectrum_rf_on_analytical'].shape == (2,)
        assert np.isfinite(arrays['spectrum_rf_on']).all()
        assert not np.array_equal(arrays['spectrum_rf_on'], arrays['spectrum_rf_off'])
        assert arrays['spectrum_transit_rate_per_us'] == report['resolved_rate_budget_per_us']['transit_reset']
    assert not (tmp_path / 'sensing.png').exists() and not (tmp_path / 'spectrum.png').exists()
    if mode == 'prescribed':
        assert 'no level search' in process.stdout
        assert 'Scanning RF' not in process.stdout


def test_new_declarations_are_validated_without_arc_import():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('arc', 'rydberg_aoa.config.parameters', 'rydberg_aoa.config.atomic_config'):
            raise AssertionError('Unexpected atomic import: ' + fullname)
sys.meta_path.insert(0, Guard())
from rydberg_aoa.config.run_profile import load_run_profile
for mode in ('prescribed', 'experimental'):
    load_run_profile('examples/lab007/' + mode + '_full.json')
'''
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
