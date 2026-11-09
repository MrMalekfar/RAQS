"""Fresh-process orchestration using production APIs; never imported by Qt UI."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from uuid import uuid4

from .jobs import digest, parse_snapshot, write_json


def await_gui_start(directory):
    """Do not initialize physics or spawn workers until the GUI owns our job."""
    from .process_job import ProcessJob
    name = 'raqs-' + uuid4().hex
    job = ProcessJob(name, create=True)
    try:
        job.assign(os.getpid())
        write_json(Path(directory) / 'worker_ready.json',
                   {'pid': os.getpid(), 'parent_pid': os.getppid(), 'job_name': name})
        if sys.stdin.readline().strip() != 'RAQS_START':
            raise RuntimeError('The GUI did not authorize calculation startup.')
    finally:
        # On authorization the GUI holds the only remaining handle; otherwise closing kills this gated worker.
        job.close()


def run(directory, diagnostics):
    directory = Path(directory)
    status = directory / 'status.json'
    started_utc = datetime.now(timezone.utc).isoformat()
    if (directory / 'manifest.json').exists():
        print('A completed run cannot be executed again in the same folder.', file=sys.stderr)
        return 1
    try:
        raw = (directory / 'profile.json').read_bytes()
        profile = parse_snapshot(raw)
        write_json(status, {'schema_version': 1, 'state': 'running'})
        from rydberg_aoa.simulation.profile_execution import execute_profile, json_default, save_profile_results
        report, result = execute_profile(profile, diagnostics_only=diagnostics)
        write_json(status, {'schema_version': 1, 'state': 'saving'})
        (directory / 'run_report.json').write_text(
            json.dumps(report, default=json_default, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        names = ['profile.json', 'run_report.json']
        if result is not None:
            save_profile_results(directory, result)
            names.append('simulation_arrays.npz')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from rydberg_aoa.visualization.profile_figures import create_profile_figures
            try:
                for name, figure in create_profile_figures(report, result).items():
                    filename = name + '.png'
                    figure.savefig(directory / filename, dpi=140, bbox_inches='tight')
                    names.append(filename)
            finally:
                plt.close('all')
        root = Path(__file__).resolve().parents[1]
        def git(*args):
            try:
                return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            except (OSError, subprocess.SubprocessError):
                return None
        manifest = {'schema_version': 1, 'state': 'completed',
                    'started_utc': started_utc,
                    'finished_utc': datetime.now(timezone.utc).isoformat(),
                    'numerical_environment': {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                    'python': sys.version, 'git_commit': git('rev-parse', 'HEAD'),
                    'git_status': git('status', '--porcelain'),
                    'packages': {name: version(name) for name in ('numpy', 'scipy', 'qutip', 'ARC-Alkali-Rydberg-Calculator', 'matplotlib', 'PySide6-Essentials')},
                    'files': {name: digest(directory / name) for name in names}}
        write_json(status, {'schema_version': 1, 'state': 'completed'})
        write_json(directory / 'manifest.json', manifest)
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(status, {'schema_version': 1, 'state': 'failed', 'error': str(exc)})
        return 1


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4) or sys.argv[2] not in ('diagnostics', 'sensing') or (len(sys.argv) == 4 and sys.argv[3] != '--managed'):
        raise SystemExit('Usage: python -m raqs_gui.worker RUN_DIRECTORY diagnostics|sensing [--managed]')
    if len(sys.argv) == 4:
        await_gui_start(Path(sys.argv[1]))
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2] == 'diagnostics'))
