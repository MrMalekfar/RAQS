"""Fresh-process orchestration using production APIs; never imported by Qt UI."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .jobs import digest, parse_snapshot, write_json


def run(directory, diagnostics):
    directory = Path(directory)
    status = directory / 'status.json'
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
                    'finished_utc': datetime.now(timezone.utc).isoformat(),
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
    if len(sys.argv) != 3 or sys.argv[2] not in ('diagnostics', 'sensing'):
        raise SystemExit('Usage: python -m raqs_gui.worker RUN_DIRECTORY diagnostics|sensing')
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2] == 'diagnostics'))
