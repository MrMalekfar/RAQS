"""Regressions for input replacement, verified figures and compact workspace use."""
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from raqs_gui.jobs import digest, write_json
from raqs_gui.structure import StructureDialog
from raqs_gui.window import MainWindow, ROOT


def test_bad_import_reload_and_cancel_preserve_unsaved_draft(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    source = tmp_path / 'profile.json'
    source.write_bytes(w.raw)
    assert w.load_profile(source)
    control = w.editor.controls[('fields', 'probe', 'power_mw')]
    control.setText('0.0002')
    draft = w.raw
    broken = tmp_path / 'broken.json'
    broken.write_text('{}', encoding='utf-8')
    selected = iter(('', str(broken), str(source), str(source)))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (next(selected), ''))
    prompts = []
    def confirm(*args):
        prompts.append(args)
        return QMessageBox.StandardButton.Cancel if len(prompts) == 1 else QMessageBox.StandardButton.Discard
    monkeypatch.setattr(QMessageBox, 'question', confirm)
    for _ in range(2):
        w.open_profile()
        assert w.dirty and w.raw == draft and w.source == source
        assert w.editor.controls[('fields', 'probe', 'power_mw')] is control
    assert not prompts
    w.open_profile()
    assert len(prompts) == 1 and w.dirty and w.raw == draft
    w.open_profile()
    assert len(prompts) == 2 and not w.dirty and w.raw != draft
    w.editor.controls[('fields', 'probe', 'power_mw')].setText('0.0002')
    source.unlink()
    w.reload_profile()
    assert w.dirty and w.raw == draft and len(prompts) == 2
    w.dirty = False
    w.close()


def test_advanced_edit_rejects_missing_field_without_replacing_active_form(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    original = w.raw
    (tmp_path / 'examples').mkdir()
    (tmp_path / 'examples' / 'unrelated.json').write_text('{', encoding='utf-8')
    (tmp_path / 'examples' / 'profile.json').write_bytes(original)
    monkeypatch.setattr('raqs_gui.window.ROOT', tmp_path)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))

    def exercise(dialog):
        assert 'profile.json' in dialog.load_presets()['fields']
        root = dialog.tree.topLevelItem(0)
        fields = next(root.child(i) for i in range(root.childCount()) if root.child(i).text(0) == 'fields')
        signal = next(fields.child(i) for i in range(fields.childCount()) if fields.child(i).text(0) == 'signal')
        fields.removeChild(signal)
        dialog.apply()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert warnings and 'signal' in warnings[0]
        assert w.raw == original and w.editor.controls
        fields.addChild(signal)
        power = next(fields.child(i) for i in range(fields.childCount()) if fields.child(i).text(0) == 'probe')
        entry = next(power.child(i) for i in range(power.childCount()) if power.child(i).text(0) == 'power_mw')
        entry.setText(1, '0.0002')
        dialog.apply()
        return dialog.result()

    monkeypatch.setattr(StructureDialog, 'exec', exercise)
    w.edit_structure()
    assert json.loads(w.raw)['fields']['probe']['power_mw'] == .0002
    assert w.sensing_button.isEnabled()
    w.dirty = False
    w.close()


def test_verified_figure_stays_bound_to_bytes_and_export_feedback_is_visible(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'profile.json').write_bytes(w.raw)
    write_json(folder / 'run_report.json', {'diagnostics_only': False, 'signed_if_mhz': .1,
               'sensing': {'true_aoa_deg': -30, 'estimated_aoa_deg': -30}})
    (folder / 'simulation_arrays.npz').write_bytes(b'consumer-only fixture')
    image = QPixmap(50, 50)
    image.fill(QColor('blue'))
    figure = folder / 'figure.png'
    assert image.save(str(figure))
    original = figure.read_bytes()
    write_json(folder / 'manifest.json', {'schema_version': 1, 'state': 'completed',
               'files': {p.name: digest(p) for p in folder.iterdir()}})
    w.load_result(folder)
    assert w.current_result_matches_draft
    image.fill(QColor('red'))
    assert image.save(str(figure))
    w.figure_zoom.setCurrentText('100%')
    assert w.image.pixmap().toImage().pixelColor(0, 0) == QColor('blue')

    w.figure_focus.setChecked(True)
    exported = tmp_path / 'export.png'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(exported), ''))
    w.save_figure()
    assert exported.read_bytes() == original and w.figure_feedback.isVisible()
    w.save_figure()
    assert 'Could not save' in w.figure_feedback.text() and w.figure_feedback.isVisible()
    assert exported.read_bytes() == original

    w.editor.controls[('fields', 'probe', 'power_mw')].setText('0.0002')
    assert w.result_directory == folder and w.folder_button.isEnabled()
    assert 'Different input snapshot' in w.result_label.text()
    w.dirty = False
    w.close()


def test_archived_diagnostics_are_not_attributed_to_different_input(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    archived = (ROOT / 'examples/atom001/cs133_prescribed.json').read_bytes()
    w.display_result(tmp_path, {'diagnostics_only': True, 'atom': {'species': 'Cs133'}}, archived, {})
    assert w.tabs.currentIndex() == 3 and 'Different input snapshot' in w.result_label.text()
    assert not w.diagnostics_text.toPlainText() and 'Cs133' not in w.diagnostics_summary.text()
    w.load_profile(ROOT / 'examples/atom001/cs133_prescribed.json')
    assert w.current_result_matches_draft and 'Cs133' in w.diagnostics_summary.text()
    w.close()


def test_compact_configuration_supports_keyboard_and_accessible_controls():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.resize(1090, 610)
    w.show()
    app.processEvents()
    assert w.size().width() == 1090 and w.size().height() == 610
    assert not w.profile_text.isVisible() and w.eligibility.isVisible()
    w.editor.sections.setFocus()
    QTest.keyClick(w.editor.sections, Qt.Key.Key_Down)
    assert w.editor.sections.currentItem().text() == 'Coupling Laser'
    assert w.editor.controls[('fields', 'coupling', 'power_mw')].isVisible()
    w.preview_toggle.setChecked(True)
    app.processEvents()
    assert w.profile_text.isVisible()
    assert all(control.accessibleName() for control in (w.log, w.diagnostics_text, w.report_text, w.figure_zoom))
    w.close()
