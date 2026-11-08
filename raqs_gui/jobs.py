"""File contracts shared by the optional UI and isolated worker; no Qt imports."""
import hashlib
import json
from pathlib import Path
import tempfile


def parse_snapshot(raw):
    # Use the canonical strict loader on the exact bytes shown by the UI.
    from rydberg_aoa.config.run_profile import load_run_profile
    with tempfile.TemporaryDirectory(prefix='raqs-profile-') as temporary:
        path = Path(temporary) / 'profile.json'
        path.write_bytes(raw)
        profile = load_run_profile(path)
    if profile.study is not None or profile.magnetic_core is not None:
        raise ValueError('This prototype supports ordinary single profiles only; study and magnetic-core profiles are not executed.')
    if profile.measurement is not None and profile.measurement.mode != 'aoa':
        raise ValueError('This prototype supports AoA profiles only; IF-only results require a separate viewer.')
    return profile


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_run(directory, raw, diagnostics):
    profile = parse_snapshot(raw)
    if not diagnostics and profile.run_config is None:
        raise ValueError('This profile has no sensing execution. Use Resolve Diagnostics or choose a sensing profile.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'profile.json').write_bytes(raw)
    write_json(directory / 'status.json', {'schema_version': 1, 'state': 'starting'})


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_result(directory):
    """Only GUI manifests with complete, unmodified files are accepted."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('state') != 'completed':
        raise ValueError('Unsupported or incomplete GUI run manifest.')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'profile.json', 'run_report.json'} <= files.keys():
        raise ValueError('The run is missing its input or report receipt.')
    for name, checksum in files.items():
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid run artifact path.')
        path = directory / name
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError(f'Missing or modified run artifact: {name}')
    report = json.loads((directory / 'run_report.json').read_text(encoding='utf-8'))
    if type(report.get('diagnostics_only')) is not bool:
        raise ValueError('Invalid run report: diagnostics_only must be boolean.')
    if not report['diagnostics_only'] and 'simulation_arrays.npz' not in files:
        raise ValueError('A sensing run is missing its numerical arrays.')
    return report, [directory / name for name in files if name.endswith('.png')]
