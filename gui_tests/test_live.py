"""Real Qt process lifecycle and numerical equivalence to the unchanged CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from raqs_gui.jobs import read_result
from raqs_gui.window import MainWindow

ROOT = Path(__file__).resolve().parents[1]


def wait_for_worker(app, window):
    deadline = time.monotonic() + 180
    while window.active and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    assert not window.active, 'GUI worker exceeded the test deadline'
    assert window.result_directory is not None, window.notice.text() + '\n' + window.log.toPlainText()


def test_two_fresh_qt_jobs_and_cli_equivalence(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.output.setText(str(tmp_path / 'نتایج GUI'))
    window.show()
    app.processEvents()
    source = tmp_path / 'profile with spaces.json'
    source.write_bytes((ROOT / 'examples/lab006/prescribed_reference.json').read_bytes())
    window.load_profile(source)
    original = window.raw
    window.start_run(True)
    source.write_bytes(b'changed after run start')
    wait_for_worker(app, window)
    diagnostic_directory = window.result_directory
    report, images = read_result(diagnostic_directory)
    assert report['diagnostics_only'] and not images
    assert (diagnostic_directory / 'profile.json').read_bytes() == original
    window.load_profile(ROOT / 'examples/lab006/experimental_near_resonant.json')
    window.editor.controls[('fields', 'signal', 'electric_field_vm')].setText('0.006')
    window.start_run(False)
    assert not window.editor.isEnabled()
    wait_for_worker(app, window)
    assert window.editor.isEnabled()
    sensing_directory = window.result_directory
    assert sensing_directory != diagnostic_directory
    report, images = read_result(sensing_directory)
    assert len(images) >= 4 and not window.image.pixmap().isNull()
    assert report['sensing']['estimated_aoa_deg'] == -30.
    assert report['applied_fields']['signal']['electric_field_vm'] == .006
    cli_directory = tmp_path / 'cli'
    with (tmp_path / 'cli.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run([sys.executable, 'main.py', '--profile',
            str(sensing_directory / 'profile.json'), '--no-plots',
            '--output-dir', str(cli_directory)], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, timeout=180)
    assert completed.returncode == 0, (tmp_path / 'cli.log').read_text(encoding='utf-8')
    cli_report = json.loads((cli_directory / 'run_report.json').read_text(encoding='utf-8'))
    for key in ('levels', 'detunings_rad_per_us', 'rabi_rad_per_us', 'signed_if_mhz', 'sensing'):
        assert report[key] == cli_report[key]
    with np.load(sensing_directory / 'simulation_arrays.npz', allow_pickle=False) as gui_arrays, np.load(cli_directory / 'simulation_arrays.npz', allow_pickle=False) as cli_arrays:
        for key in ('tlist', 'time_traces', 'if_snapshots', 'sample_covariance'):
            np.testing.assert_array_equal(gui_arrays[key], cli_arrays[key])
    manifest = json.loads((sensing_directory / 'manifest.json').read_text())
    assert manifest['packages']['ARC-Alkali-Rydberg-Calculator'] == '3.9.0'
    window.dirty = False
    window.close()
