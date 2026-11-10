"""Qt state checks; install requirements-gui.txt to run these tests."""
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QProcess
from raqs_gui.window import MainWindow

ROOT = Path(__file__).resolve().parents[1]


def test_profile_state_rejects_invalid_input_and_active_window_close(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert 'DejaVu Sans' in QFontDatabase.families()
    assert window.sensing_button.isEnabled()
    window.load_profile(ROOT / 'examples/lab006/prescribed_reference.json')
    assert window.diagnostic_button.isEnabled()
    assert not window.sensing_button.isEnabled()
    snapshot = window.raw
    window.load_profile(ROOT / 'examples/lab006/experimental_near_resonant.json')
    assert window.sensing_button.isEnabled() and window.raw != snapshot
    window.active = True
    window.update_controls()
    assert not window.open_button.isEnabled()
    assert not window.close()
    window.active = False
    window.update_controls()
    invalid = tmp_path / 'invalid.json'
    invalid.write_text('{}')
    snapshot = window.raw
    assert not window.load_profile(invalid)
    assert window.raw == snapshot and window.diagnostic_button.isEnabled()
    assert 'Error:' in window.notice.text()
    window.close()


def test_failed_process_start_does_not_leave_a_busy_or_successful_ui(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.run_directory = tmp_path
    window.active = True
    window.update_controls()
    window.process.start(str(tmp_path / 'missing-python.exe'), [])
    window.process.waitForFinished(2000)
    app.processEvents()
    assert window.process.state() == QProcess.ProcessState.NotRunning
    assert not window.active and window.result_directory is None
    assert 'Failed to start' in window.stage.text()
    assert 'Error:' in window.notice.text()
    window.close()
