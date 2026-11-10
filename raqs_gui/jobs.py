"""File contracts shared by the optional UI and isolated worker; no Qt imports."""
import hashlib
import json
from pathlib import Path
import tempfile


def auxiliary_document(raw, source, repository):
    """Describe research metadata; never invent or convert a physical profile."""
    document = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(document, dict) or 'mode' in document or 'fields' in document:
        return None
    if not ({'task', 'campaign', 'study', 'production_sha256'} & document.keys()
            or Path(source).name in ('design.json', 'design_v2.json', 'manifest.json', 'profile_manifest.json', 'input_manifest.json', 'source_guard.json', 'validation_evidence.json')):
        return None
    references = []
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.endswith('.json'):
            references.append(value)
    visit(document)
    profiles, missing = [], []
    for reference in dict.fromkeys(references):
        reference_path = Path(reference)
        candidates = [reference_path] if reference_path.is_absolute() else [Path(source).parent / reference_path, Path(repository) / reference_path]
        existing = next((p for p in candidates if p.is_file()), None)
        if existing is None:
            missing.append(reference)
            continue
        try:
            candidate = json.loads(existing.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        if isinstance(candidate, dict) and 'fields' in candidate and 'mode' in candidate:
            profiles.append(existing.resolve())
    message = ('This JSON is a research design, manifest or evidence document, not a run profile. '
               'A run profile must declare mode, fields and its atomic preparation/ladder. No automatic conversion was applied.')
    if profiles:
        message += ' Use Choose Referenced Profile to open a declared input.'
    if missing:
        message += ' Missing referenced JSON files: ' + ', '.join(missing[:4])
    if not profiles:
        message += ' The original research workflow may require tools or input files not present in this checkout.'
    return message, list(dict.fromkeys(profiles))


def load_snapshot(raw):
    # Use the canonical strict loader on the exact bytes shown by the UI.
    from rydberg_aoa.config.run_profile import load_run_profile
    with tempfile.TemporaryDirectory(prefix='raqs-profile-') as temporary:
        path = Path(temporary) / 'profile.json'
        path.write_bytes(raw)
        profile = load_run_profile(path)
    return profile


def parse_snapshot(raw):
    profile = load_snapshot(raw)
    if profile.study is not None or profile.magnetic_core is not None:
        raise ValueError('This prototype supports ordinary single profiles only; study and magnetic-core profiles are not executed.')
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


def read_result(directory, *, capture=False):
    """Verify a GUI manifest. Capture binds the UI to the bytes verified here."""
    directory = Path(directory)
    if (directory / 'interrupted.json').exists():
        raise ValueError('This run was interrupted. Its partial artifacts are not a completed result.')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('state') != 'completed':
        raise ValueError('Unsupported or incomplete GUI run manifest.')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'profile.json', 'run_report.json'} <= files.keys():
        raise ValueError('The run is missing its input or report receipt.')
    content = {}
    images = []
    for name, checksum in files.items():
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid run artifact path.')
        path = directory / name
        if path.is_symlink():
            raise ValueError(f'Missing or modified run artifact: {name}')
        is_image = name.endswith('.png')
        if name == 'run_report.json' or (capture and (name == 'profile.json' or is_image)):
            data = path.read_bytes()
            content[name] = data
            actual = hashlib.sha256(data).hexdigest()
        else:
            actual = digest(path)
        if actual != checksum:
            raise ValueError(f'Missing or modified run artifact: {name}')
        if is_image:
            images.append(path)
    report = json.loads(content['run_report.json'].decode('utf-8'))
    if type(report.get('diagnostics_only')) is not bool:
        raise ValueError('Invalid run report: diagnostics_only must be boolean.')
    if not report['diagnostics_only'] and 'simulation_arrays.npz' not in files:
        raise ValueError('A sensing run is missing its numerical arrays.')
    if capture:
        return report, content['profile.json'], {path.name: content[path.name] for path in images}
    return report, images
