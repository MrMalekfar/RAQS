"""Task workflow, explicit effects, invalid drafts, rendering and failure states."""
import json
import os
import time
import threading
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QPushButton, QFileDialog
from raqs_gui.window import MainWindow, ROOT


def test_task_sections_inline_error_and_preview_are_one_draft():
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.show(); app.processEvents()
    assert w.navigation.currentItem().text() == 'Configure'
    assert w.editor.sections.currentItem().text() == 'Probe Laser'
    control = w.editor.controls[('fields', 'probe', 'power_mw')]
    control.setFocus(); control.selectAll(); QTest.keyClicks(control, 'not-a-number')
    assert w.raw is None and not w.sensing_button.isEnabled()
    assert 'Not runnable' in w.eligibility.text()
    assert not w.editor.errors[('fields', 'probe', 'power_mw')].isHidden()
    control.selectAll(); QTest.keyClicks(control, '1e-4')
    assert w.sensing_button.isEnabled()
    assert json.loads(w.raw) == json.loads(w.profile_text.toPlainText())
    assert w.editor.errors[('fields', 'probe', 'power_mw')].isHidden()
    w.dirty = False; w.close()


def test_optional_effects_preserve_other_values_and_are_removable(monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    before = json.loads(w.raw)
    presets = w.option_presets()
    source = next(iter(presets['numerics']))
    answers = iter([('numerics', True), (source, True)])
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: next(answers))
    w.add_option()
    added = json.loads(w.editor.snapshot())
    assert added['numerics'] == presets['numerics'][source]
    assert added['fields'] == before['fields']
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('numerics', True))
    w.remove_option()
    assert json.loads(w.editor.snapshot()) == before
    w.dirty = False; w.close()


def test_import_save_and_reopen_through_the_actual_buttons(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.show(); app.processEvents()
    source = ROOT / 'examples/lab006/experimental_near_resonant.json'
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(source), ''))
    QTest.mouseClick(w.open_button, Qt.MouseButton.LeftButton)
    control = w.editor.controls[('fields', 'probe', 'power_mw')]
    control.setFocus(); control.selectAll(); QTest.keyClicks(control, '0.0002')
    target = tmp_path / 'saved.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), ''))
    QTest.mouseClick(w.copy_button, Qt.MouseButton.LeftButton)
    assert not w.dirty and json.loads(target.read_bytes())['fields']['probe']['power_mw'] == .0002
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(target), ''))
    QTest.mouseClick(w.open_button, Qt.MouseButton.LeftButton)
    assert json.loads(w.raw) == json.loads(w.profile_text.toPlainText()) == json.loads(target.read_bytes())
    w.close()


def test_incomplete_saved_run_is_rejected_asynchronously(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.request_result(tmp_path)
    deadline = time.monotonic() + 10
    while w.reading and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    assert not w.reading and w.result_directory is None
    assert 'Failed' in w.stage.text() and 'Error:' in w.notice.text()
    w.close()


def test_late_artifact_read_cannot_replace_a_changed_draft(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import raqs_gui.result_reader as reader
    gate = threading.Event()
    def delayed_read(directory, *, capture):
        assert gate.wait(5)
        return {'diagnostics_only': True}, b'{}', {}
    monkeypatch.setattr(reader, 'read_result', delayed_read)
    w = MainWindow(); w.request_result(tmp_path)
    w.editor.controls[('fields', 'probe', 'power_mw')].setText('0.0002')
    current = w.raw
    gate.set()
    deadline = time.monotonic() + 10
    while w.reading and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    assert not w.reading and w.result_directory is None
    assert w.raw == current and 'Error:' not in w.notice.text()
    w.dirty = False; w.close()


def test_vector_size_controls_never_silently_drop_values():
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.load_profile(ROOT / 'examples/lab007/prescribed_full.json')
    table = w.editor.controls[('spectrum', 'probe_offsets_mhz')]
    before = json.loads(w.editor.snapshot())['spectrum']['probe_offsets_mhz']
    buttons = {b.text(): b for b in table.parentWidget().findChildren(QPushButton)}
    table.setCurrentCell(0, 0)
    buttons['Duplicate value'].click()
    after = json.loads(w.editor.snapshot())['spectrum']['probe_offsets_mhz']
    assert after == before + [before[0]]
    table.setCurrentCell(0, table.columnCount() - 1)
    buttons['Remove value'].click()
    assert json.loads(w.editor.snapshot())['spectrum']['probe_offsets_mhz'] == before
    w.dirty = False; w.close()


def test_actual_storage_failure_never_publishes_completion(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import raqs_gui.window as window_module
    prepare = window_module.prepare_run
    def blocked_report(directory, raw, diagnostics):
        prepare(directory, raw, diagnostics)
        (directory / 'run_report.json').mkdir()
    monkeypatch.setattr(window_module, 'prepare_run', blocked_report)
    w = MainWindow(); w.output.setText(str(tmp_path))
    w.load_profile(ROOT / 'examples/lab006/prescribed_reference.json')
    w.start_run(True)
    deadline = time.monotonic() + 120
    while w.active and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.02)
    assert not w.active and not w.reading
    assert w.result_directory is None and not (w.run_directory / 'manifest.json').exists()
    assert 'Failed' in w.stage.text()
    assert 'Error:' in w.notice.text()
    w.close()
