"""Fresh-process public entry points, actual ARC and bounded spawn sensing."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    return subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT,
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)


def test_declarations_and_help_need_no_arc_physics_or_worker_import():
    script = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('arc', 'rydberg_aoa.config.parameters',
                        'rydberg_aoa.config.atomic_config', 'rydberg_aoa.simulation.time_domain'):
            raise AssertionError('Forbidden pre-validation import: ' + fullname)
sys.meta_path.insert(0, Guard())
import main, cli
from rydberg_aoa.config.run_profile import load_run_profile
load_run_profile('examples/lab006/experimental_reference.json')
load_run_profile('examples/lab006/prescribed_reference.json')
sys.argv = ['main.py', '--help']
main.main()
'''
    process = run('-c', script)
    assert process.returncode == 0, process.stdout + process.stderr
    assert '--profile' in process.stdout


@pytest.mark.parametrize('entry,profile', [('main.py', 'experimental_reference.json'),
                                         ('cli.py', 'prescribed_reference.json')])
def test_actual_entry_diagnostics_keep_inputs_and_prepared_channel(tmp_path, entry, profile):
    source = ROOT / 'examples/lab006' / profile
    process = run(entry, '--profile', str(source), '--diagnostics-only', '--no-plots',
                  '--output-dir', str(tmp_path))
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    assert report['diagnostics_only'] is True
    assert report['applied_fields'] == json.loads(source.read_text())['fields']
    assert report['levels']['rydberg1']['mj'] == report['levels']['rydberg2']['mj'] == -.5
    assert report['levels']['rydberg1']['n'] == 26 and report['levels']['rydberg2']['n'] == 27
    assert report['rf_orientation']['signed_frequency_hz'] < 0
    assert report['rf_orientation']['direction'] == -1
    assert abs(report['detunings_rad_per_us'][1]) > 1e6
    assert report.get('sensing') is None
    assert not (tmp_path / 'sensing.png').exists()


def test_actual_main_arc_spawn_if_and_doa(tmp_path):
    process = run('main.py', '--profile', 'examples/lab006/experimental_near_resonant.json',
                  '--no-plots', '--output-dir', str(tmp_path))
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    assert report['diagnostics_only'] is False
    assert report['sensing']['paths'] == 3 and report['sensing']['samples_per_path'] == 601
    assert report['sensing']['true_aoa_deg'] == report['sensing']['estimated_aoa_deg'] == -30.
    assert report['signed_if_mhz'] == .1
    assert abs(report['detunings_rad_per_us'][0]) < 1e-6
    assert abs(report['detunings_rad_per_us'][1]) < 1e-6
    assert not report['zeeman_included'] and not report['noise_included']
    assert not report['doppler_ensemble_included']


def test_invalid_profile_and_missing_profile_do_not_touch_arc_or_logs(tmp_path):
    invalid = tmp_path / 'invalid.json'
    invalid.write_text('{"mode":"experimental"}')
    process = run('main.py', '--profile', str(invalid), '--diagnostics-only')
    assert process.returncode != 0
    assert 'PhysicsConfigurationError' in process.stderr
    assert 'Atomic database' not in process.stdout + process.stderr
    missing = run('main.py')
    assert missing.returncode != 0
    assert '--profile is required' in missing.stderr
    assert 'Atomic database' not in missing.stdout + missing.stderr
