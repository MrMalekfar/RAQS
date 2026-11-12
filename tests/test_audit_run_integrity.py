import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import cli
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.simulation import profile_execution


@pytest.mark.parametrize('case', ['existing', 'export_error', 'report_error', 'success'])
def test_single_profile_publication(tmp_path, monkeypatch, case):
    source = tmp_path / 'input.json'
    raw = Path('examples/lab006/prescribed_reference.json').read_bytes()
    source.write_bytes(raw)
    directory = tmp_path / 'run'
    directory.mkdir()
    arrays = directory / 'simulation_arrays.npz'
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--profile', str(source), '--no-plots',
                                      '--output-dir', str(directory)])
    if case == 'existing':
        arrays.write_bytes(b'old arrays')
        monkeypatch.setattr(profile_execution, 'execute_profile', lambda *a, **k: pytest.fail('executed'))
        with pytest.raises(FileExistsError):
            cli.main()
        assert arrays.read_bytes() == b'old arrays'
        assert not (directory / 'profile.json').exists()
    else:
        def execute(*args, **kwargs):
            source.write_bytes(b'changed after parsing')
            monkeypatch.setenv('OMP_NUM_THREADS', '2')
            return {'diagnostics_only': False}, {}

        def save(*args):
            if case == 'export_error':
                raise OSError('export failed')
            arrays.write_bytes(b'new arrays')
            if case == 'report_error':
                (directory / 'run_report.json').mkdir()

        monkeypatch.setattr(profile_execution, 'execute_profile', execute)
        monkeypatch.setattr(profile_execution, 'save_profile_results', save)
        if case == 'success':
            cli.main()
            assert (directory / 'profile.json').read_bytes() == raw
            receipt = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
            assert receipt['state'] == 'completed'
            assert receipt['numerical_environment']['OMP_NUM_THREADS'] == '1'
            assert receipt['files']['profile.json'] == hashlib.sha256(raw).hexdigest()
            assert set(receipt['files']) == {'profile.json', 'run_report.json', 'simulation_arrays.npz'}
            assert receipt['files'][arrays.name] == hashlib.sha256(arrays.read_bytes()).hexdigest()
            return
        with pytest.raises(OSError):
            cli.main()
    assert not (directory / 'run_report.json').is_file()
    assert not (directory / 'manifest.json').exists()


@pytest.mark.parametrize('gammas,dephasing,transit', [
    ([-1., 0., 0.], [0., 0., 0.], 0.),
    ([0., 0., 0.], [float('nan'), 0., 0.], 0.),
    ([0., 0., 0.], [0., 0., 0.], -1.),
    ([False, 0., 0.], [0., 0., 0.], 0.),
    ([0., 0.], [0., 0., 0.], 0.),
])
def test_model_rejects_invalid_dissipation(gammas, dephasing, transit):
    from rydberg_aoa.physics.hamiltonian import create_sensor_model

    with pytest.raises(PhysicsConfigurationError):
        create_sensor_model(1., 2., 3., 0., 0., 0., gammas, dephasing, transit)


def test_explicit_model_import_without_arc():
    process = subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['arc'] = None
from rydberg_aoa.config import atomic_config as ac, parameters
from rydberg_aoa.exceptions import PhysicsCalculationError
from rydberg_aoa.physics.hamiltonian import create_sensor_model
assert ac.ATOM is parameters.CALCULATED_N0 is parameters.SI_Constants.N0 is None
H, jumps, _ = create_sensor_model(1., 2., 3., 0., 0., 0., [0.] * 3, [0.] * 3)
assert H.isherm and not jumps
try:
    ac.initialize_atomic_physics()
except PhysicsCalculationError as exc:
    assert 'ARC construction failed' in str(exc)
else:
    raise AssertionError('Explicit initialization succeeded without ARC')
assert ac.ATOM is parameters.CALCULATED_N0 is parameters.SI_Constants.N0 is None
'''], capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
