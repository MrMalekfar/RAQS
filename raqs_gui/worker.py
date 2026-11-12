"""Fresh-process orchestration using production APIs; never imported by Qt UI."""
import json
import os
from pathlib import Path
import sys
import traceback
from uuid import uuid4

from .jobs import parse_snapshot, write_json
from rydberg_aoa.simulation.run_receipt import capture_run_provenance, write_run_receipt


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
    if (directory / 'manifest.json').exists():
        print('A completed run cannot be executed again in the same folder.', file=sys.stderr)
        return 1
    try:
        raw = (directory / 'profile.json').read_bytes()
        profile = parse_snapshot(raw)
        provenance = capture_run_provenance(extra_packages=('PySide6-Essentials',))
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
        write_json(status, {'schema_version': 1, 'state': 'completed'})
        write_run_receipt(directory, names, provenance)
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
