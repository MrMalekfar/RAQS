"""Regression for the exact Cs133 IF-only profile opened by the user."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from raqs_gui.window import MainWindow, ROOT
from raqs_gui.jobs import read_result
from gui_tests.test_live import wait_for_worker
import subprocess
import sys
import json
import numpy as np


def test_cs133_if_only_diagnostics_and_calculation(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_profile(ROOT / 'examples/atom001/cs133_prescribed.json')
    window.output.setText(str(tmp_path))
    assert window.diagnostic_button.isEnabled()
    assert window.sensing_button.isEnabled()
    for diagnostics in (True, False):
        window.start_run(diagnostics)
        wait_for_worker(app, window)
        report, images = read_result(window.result_directory)
        assert report['atom']['species'] == 'Cs133'
        assert report['diagnostics_only'] is diagnostics
        assert report['measurement']['mode'] == 'if_only'
        if not diagnostics:
            assert images
            assert 'DoA not requested' in window.result_label.text()
            assert report['sensing']['estimated_aoa_deg'] is None
    cli_directory = tmp_path / 'cli'
    with (tmp_path / 'cli.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, 'main.py', '--profile', 'examples/atom001/cs133_prescribed.json', '--no-plots', '--output-dir', str(cli_directory)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0
    cli_report = json.loads((cli_directory / 'run_report.json').read_text(encoding='utf-8'))
    assert cli_report['if_measurement'] == report['if_measurement']
    with np.load(cli_directory / 'simulation_arrays.npz', allow_pickle=False) as a, np.load(window.result_directory / 'simulation_arrays.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(a['time_traces'], b['time_traces'])
    window.close()
