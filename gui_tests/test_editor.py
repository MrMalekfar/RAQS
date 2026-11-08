"""Bidirectional form/JSON integration, without changing scientific validation."""
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from raqs_gui.jobs import prepare_run
from raqs_gui.window import MainWindow, ROOT


def make_window():
    app = QApplication.instance() or QApplication([])
    return app, MainWindow()


def discard(window):
    window.dirty = False
    window.close()


def test_new_templates_are_directly_editable_without_import():
    app, window = make_window()
    for index in range(window.template_choice.count()):
        window.template_choice.setCurrentIndex(index)
        window.new_profile()
        assert window.raw is not None, window.notice.text()
        assert window.source is None and not window.reload_button.isEnabled()
        assert window.editor.controls[('fields', 'probe', 'power_mw')].isEnabled()
        assert json.loads(window.editor.snapshot()) == json.loads(window.raw)
    discard(window)


def test_import_repopulates_all_controls_and_preserves_unedited_options():
    app, window = make_window()
    for filename in ('lab006/experimental_near_resonant.json', 'lab007/prescribed_full.json', 'noise007/prescribed_rin.json'):
        source = ROOT / 'examples' / filename
        window.load_profile(source)
        original = json.loads(source.read_bytes())
        assert json.loads(window.editor.snapshot()) == original
        control = window.editor.controls[('fields', 'coupling', 'power_mw')]
        assert control.text() == str(original['fields']['coupling']['power_mw'])
    window.load_profile(ROOT / 'examples/lab006/prescribed_reference.json')
    assert ('environment', 'temperature_k') not in window.editor.controls
    assert ('bounds', 'n_min') not in window.editor.controls
    discard(window)


def test_numeric_edits_update_preview_snapshot_and_save_without_touching_source(tmp_path, monkeypatch):
    app, window = make_window()
    source = tmp_path / 'source.json'
    source.write_bytes((ROOT / 'examples/lab006/experimental_near_resonant.json').read_bytes())
    original = source.read_bytes()
    window.load_profile(source)
    control = window.editor.controls[('fields', 'signal', 'electric_field_vm')]
    control.setText('1.25e-3')
    assert window.dirty and window.sensing_button.isEnabled()
    assert json.loads(window.raw)['fields']['signal']['electric_field_vm'] == .00125
    assert json.loads(window.profile_text.toPlainText()) == json.loads(window.raw)
    assert window.result_directory is None and not window.diagnostics_text.toPlainText()
    assert 'Inputs changed' in window.stage.text()
    prepare_run(tmp_path / 'job', window.raw, False)
    assert (tmp_path / 'job/profile.json').read_bytes() == window.raw
    saved = tmp_path / 'edited.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(saved), ''))
    window.save_copy()
    assert saved.read_bytes() == window.raw and source.read_bytes() == original
    assert not window.dirty
    discard(window)


def test_invalid_draft_blocks_stale_execution_and_recovers():
    app, window = make_window()
    window.load_profile(ROOT / 'examples/lab006/experimental_near_resonant.json')
    window.diagnostics_text.setPlainText('old diagnostic')
    control = window.editor.controls[('fields', 'probe', 'power_mw')]
    for invalid in ('', 'NaN', '-1'):
        control.setText(invalid)
        assert window.raw is None and window.profile is None
        assert not window.sensing_button.isEnabled()
        assert not window.diagnostic_button.isEnabled() and not window.copy_button.isEnabled()
        assert not window.diagnostics_text.toPlainText()
    control.setText('0.0002')
    assert window.raw is not None and window.sensing_button.isEnabled()
    discard(window)


def test_booleans_tables_and_quantum_numbers_use_core_validation():
    app, window = make_window()
    window.load_profile(ROOT / 'examples/lab007/prescribed_full.json')
    toggle = window.editor.controls[('environment', 'collision_enabled')]
    toggle.setCurrentIndex(toggle.findData(False))
    assert json.loads(window.raw)['environment']['collision_enabled'] is False
    table = window.editor.controls[('execution', 'positions_m')]
    table.item(1, 0).setText('0.0003')
    assert json.loads(window.raw)['execution']['positions_m'][1][0] == .0003
    # An integer-looking JSON value for mF is not an integer-only physical input.
    window.editor.controls[('ladder', 'ground', 'mf')].setText('0.0')
    assert window.raw is not None
    window.editor.controls[('execution', 'sample_count')].setText('601.5')
    assert window.raw is None
    discard(window)


def test_effective_atom_default_is_not_inserted_until_the_user_changes_it():
    app, window = make_window()
    window.new_profile()
    assert 'atom' not in json.loads(window.raw)
    species = window.editor.controls[('atom', 'species')]
    assert species.currentData() == 'Rb85'
    species.setCurrentIndex(species.findData('Rb87'))
    assert json.loads(window.raw)['atom'] == {'species': 'Rb87', 'isotope_fraction': 1.0}
    species.setCurrentIndex(species.findData('Rb85'))
    assert 'atom' not in json.loads(window.raw)
    discard(window)


def test_unsaved_changes_require_discard_before_replacing_or_closing(monkeypatch):
    app, window = make_window()
    window.new_profile()
    window.editor.controls[('fields', 'signal', 'electric_field_vm')].setText('0.007')
    edited = window.raw
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Cancel)
    window.new_profile()
    assert window.raw == edited
    assert not window.close()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Discard)
    assert window.close()
