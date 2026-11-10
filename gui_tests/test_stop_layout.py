"""Stop owns only its process tree; editing, plotting and logging stay visible."""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QProcess
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit
from PySide6.QtGui import QPixmap, QColor

from raqs_gui.window import MainWindow, ROOT
from raqs_gui.structure import StructureDialog
from raqs_gui.jobs import write_json, read_result


def pump(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    assert predicate(), 'Timed out waiting for the GUI/process state'


def test_structure_value_editor_is_visible_and_commits_a_scientific_number():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    d = StructureDialog({'power': .0001}, dict, w); d.show(); app.processEvents()
    item = d.tree.topLevelItem(0).child(0)
    rect = d.tree.visualRect(d.tree.indexFromItem(item, 1))
    QTest.mouseClick(d.tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    app.processEvents()
    editors = [x for x in d.tree.findChildren(QLineEdit) if x.isVisible()]
    assert len(editors) == 1
    edit = editors[0]
    assert edit.height() >= edit.fontMetrics().height() + 6
    assert '2px solid' in edit.styleSheet()
    edit.selectAll(); QTest.keyClicks(edit, '2.5e-4'); QTest.keyClick(edit, Qt.Key.Key_Return)
    app.processEvents()
    assert d.read(d.tree.topLevelItem(0)) == {'power': .00025}
    d.reject(); w.close()


def test_log_follows_last_line_and_manual_scroll_can_be_preserved(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.tabs.setCurrentIndex(2); w.show(); app.processEvents()
    w.run_directory = tmp_path
    write_json(tmp_path / 'status.json', {'state': 'running'})
    log = tmp_path / 'run.log'
    log.write_text('\n'.join(f'line {i}' for i in range(1000)))
    w.poll(); app.processEvents()
    bar = w.log.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    assert w.log.toPlainText().endswith('line 999')
    w.follow_log.setChecked(False); bar.setValue(0)
    with log.open('a') as stream: stream.write('\nnew final line')
    w.poll(); app.processEvents()
    assert bar.value() == 0
    w.follow_log.setChecked(True)
    assert bar.value() == bar.maximum()
    w.close()


def test_figure_first_fits_the_page_and_focus_reclaims_space(tmp_path):
    """Consumer layout fixture only; this does not validate physical output."""
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.show(); app.processEvents()
    (tmp_path / 'profile.json').write_bytes(w.raw)
    image = QPixmap(1200, 800); image.fill(QColor('white'))
    path = tmp_path / 'consumer.png'; assert image.save(str(path))
    report = {'diagnostics_only': False, 'signed_if_mhz': .1,
              'sensing': {'true_aoa_deg': -30, 'estimated_aoa_deg': -30}}
    w.display_result(tmp_path, report, w.raw, {path.name: path.read_bytes()}); app.processEvents(); app.processEvents()
    assert w.result_tabs.currentWidget() == w.figures_page
    assert w.figure_zoom.currentText() == 'Fit page'
    assert w.image.pixmap().height() <= w.figure_scroll.viewport().height()
    height = w.figure_scroll.viewport().height()
    w.figure_focus.setChecked(True); app.processEvents(); app.processEvents()
    assert w.figure_scroll.viewport().height() > height + 70
    assert not w.navigation.isVisible() and not w.bottom_status.isVisible()
    QTest.keyClick(w, Qt.Key.Key_Escape); app.processEvents()
    assert not w.figure_focus.isChecked() and w.navigation.isVisible()
    w.close()


def test_actual_profile_can_be_stopped_and_is_not_a_completed_run(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(); w.output.setText(str(tmp_path))
    try:
        w.start_run(False)
        pump(app, lambda: w.worker_ready)
        assert w.active and w.stop_button.isEnabled()
        QTest.mouseClick(w.stop_button, Qt.MouseButton.LeftButton)
        pump(app, lambda: not w.active)
        assert w.result_directory is None
        assert (w.run_directory / 'interrupted.json').exists()
        assert 'Stopped' in w.stage.text()
    finally:
        if w.job is not None: w.job.terminate(); w.job.close(); w.job = None
        if w.process.state() != QProcess.ProcessState.NotRunning:
            w.process.kill(); w.process.waitForFinished(5000)
        w.active = False; w.timer.stop(); w.close()


def test_calculation_does_not_start_when_gui_cannot_own_its_processes(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import raqs_gui.process_job as backend
    def denied(*args, **kwargs):
        raise PermissionError('ownership denied by test boundary')
    monkeypatch.setattr(backend, 'ProcessJob', denied)
    w = MainWindow(); w.output.setText(str(tmp_path))
    w.start_run(True)
    pump(app, lambda: not w.active)
    assert w.result_directory is None and not w.reading
    assert not (w.run_directory / 'manifest.json').exists()
    assert 'safe process ownership' in w.notice.text()
    assert w.process.state() == QProcess.ProcessState.NotRunning
    w.close()


def test_windows_stop_terminates_owned_descendants_only_and_allows_next_run(tmp_path):
    assert sys.platform == 'win32', 'This test verifies the Windows-only stop backend.'
    from raqs_gui.process_job import ProcessJob
    app = QApplication.instance() or QApplication([])
    unrelated = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(60)'], creationflags=subprocess.CREATE_NO_WINDOW)
    helper = tmp_path / 'owned_tree.py'
    helper.write_text(f'''import sys,subprocess,time
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from raqs_gui.worker import await_gui_start
from raqs_gui.jobs import write_json
directory=Path(sys.argv[1])
await_gui_start(directory)
child=subprocess.Popen([sys._base_executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
write_json(directory/'child.json',{{'pid':child.pid}})
time.sleep(60)
''')
    directory = tmp_path / 'owned'; directory.mkdir()
    write_json(directory / 'status.json', {'state': 'starting'})
    w = MainWindow(); observer = None
    try:
        w.run_directory = directory; w.active = True; w.update_controls()
        w.process.setWorkingDirectory(str(ROOT))
        w.process.setStandardOutputFile(str(directory / 'run.log'))
        w.timer.start()
        w.process.start(sys.executable, ['-u', str(helper), str(directory)])
        pump(app, lambda: (directory / 'child.json').exists())
        info = json.loads((directory / 'worker_ready.json').read_bytes())
        observer = ProcessJob(info['job_name'])
        assert observer.active_processes() >= 2
        QTest.mouseClick(w.stop_button, Qt.MouseButton.LeftButton)
        pump(app, lambda: not w.active)
        assert observer.active_processes() == 0
        assert unrelated.poll() is None
        assert (directory / 'interrupted.json').exists()
        assert w.result_directory is None and 'Stopped' in w.stage.text()
        assert w.process.state() == QProcess.ProcessState.NotRunning
        try: read_result(directory)
        except ValueError as exc: assert 'interrupted' in str(exc)
        else: raise AssertionError('A stopped folder was accepted as complete')
        w.output.setText(str(tmp_path / 'next'))
        w.load_profile(ROOT / 'examples/lab006/prescribed_reference.json')
        w.start_run(True)
        pump(app, lambda: not w.active and not w.reading, timeout=120)
        assert w.result_directory is not None, w.notice.text() + w.log.toPlainText()
    finally:
        if w.job is not None: w.job.terminate(); w.job.close(); w.job = None
        if w.process.state() != QProcess.ProcessState.NotRunning:
            w.process.kill(); w.process.waitForFinished(5000)
        w.active = False; w.timer.stop(); w.close()
        if observer is not None: observer.close()
        unrelated.terminate(); unrelated.wait(timeout=10)
