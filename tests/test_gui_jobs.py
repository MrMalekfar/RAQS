"""Optional-UI contracts, independent of Qt and live numerical fixtures."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from raqs_gui.jobs import digest, parse_snapshot, prepare_run, read_result, write_json
from rydberg_aoa.exceptions import PhysicsConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def raw_profile():
    return (ROOT / 'examples/lab006/prescribed_reference.json').read_bytes()


def test_snapshot_preserved_and_existing_directory_refused(tmp_path):
    directory = tmp_path / 'new run'
    raw = raw_profile()
    prepare_run(directory, raw, True)
    assert (directory / 'profile.json').read_bytes() == raw
    with pytest.raises(FileExistsError):
        prepare_run(directory, raw, True)
    assert (directory / 'profile.json').read_bytes() == raw


def test_invalid_or_unsupported_inputs_create_no_run(tmp_path):
    for raw in (b'{"mode":"prescribed"}', b'{"mode":1,"mode":2}'):
        with pytest.raises(PhysicsConfigurationError):
            prepare_run(tmp_path / 'absent', raw, True)
        assert not (tmp_path / 'absent').exists()
    with pytest.raises(ValueError, match='no sensing execution'):
        prepare_run(tmp_path / 'absent', raw_profile(), False)
    with pytest.raises(ValueError, match='single profiles'):
        parse_snapshot((ROOT / 'examples/noise008/prescribed_readout_sweep.json').read_bytes())
    profile = parse_snapshot((ROOT / 'examples/noise013/prescribed_signal_pin.json').read_bytes())
    assert profile.measurement.mode == 'if_only'


def receipt(directory, *, sensing=False):
    (directory / 'profile.json').write_bytes(raw_profile())
    write_json(directory / 'run_report.json', {'diagnostics_only': not sensing})
    manifest = {'schema_version': 1, 'state': 'completed', 'files':
                {name: digest(directory / name) for name in ('profile.json', 'run_report.json')}}
    write_json(directory / 'manifest.json', manifest)
    return manifest


def test_results_require_final_receipt_and_unchanged_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_result(tmp_path)
    receipt(tmp_path)
    report, images = read_result(tmp_path)
    assert report['diagnostics_only'] is True and images == []
    (tmp_path / 'profile.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='modified'):
        read_result(tmp_path)


def test_boolean_manifest_version_is_not_an_integer_version(tmp_path):
    manifest = receipt(tmp_path)
    manifest['schema_version'] = True
    write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Unsupported'):
        read_result(tmp_path)


def test_sensing_requires_arrays_and_paths_cannot_escape(tmp_path):
    manifest = receipt(tmp_path, sensing=True)
    with pytest.raises(ValueError, match='numerical arrays'):
        read_result(tmp_path)
    manifest['files']['../outside'] = 'bad'
    write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='artifact path'):
        read_result(tmp_path)


def test_worker_failure_leaves_no_completed_receipt(tmp_path):
    from raqs_gui.worker import run
    (tmp_path / 'profile.json').write_text('{}')
    assert run(tmp_path, True) == 1
    assert not (tmp_path / 'manifest.json').exists()
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'failed'


def test_ui_contract_imports_do_not_initialize_arc():
    script = '''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('arc','rydberg_aoa.config.atomic_config','rydberg_aoa.config.parameters'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
import raqs_gui,raqs_gui.jobs,raqs_gui.worker
from pathlib import Path
raqs_gui.jobs.parse_snapshot(Path('examples/lab006/prescribed_reference.json').read_bytes())
'''
    process = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
