"""Shared completion receipt for single-profile runs."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def capture_run_provenance(*, extra_packages=()):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {'started_utc': datetime.now(timezone.utc).isoformat(),
            'numerical_environment': {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'python': sys.version, 'git_commit': git('rev-parse', 'HEAD'),
            'git_status': git('status', '--porcelain'),
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'qutip', 'ARC-Alkali-Rydberg-Calculator', 'matplotlib', *extra_packages)}}


def write_run_receipt(directory, names, provenance):
    manifest = {'schema_version': 1, 'state': 'completed', **provenance,
                'files': {name: digest(directory / name) for name in names}}
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(directory / 'manifest.json', manifest)
