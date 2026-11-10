import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from raqs_gui.window import MainWindow, ROOT
from raqs_gui.structure import StructureDialog


def test_start_and_preview_toggle_are_visible_without_finding_a_tab():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.sensing_button.text() == 'Start Calculation'
    assert window.sensing_button.isVisible() and window.sensing_button.isEnabled()
    assert window.preview_toggle.isVisible() and not window.profile_text.isVisible()
    window.preview_toggle.click()
    assert window.profile_text.isVisible()
    assert json.loads(window.profile_text.toPlainText()) == json.loads(window.raw)
    window.close()


def test_unsupported_execution_still_imports_all_inputs_and_preview():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = ROOT / 'examples/noise008/prescribed_readout_sweep.json'
    window.load_profile(path)
    assert json.loads(window.profile_text.toPlainText()) == json.loads(path.read_bytes())
    assert ('study', 'kind') in window.editor.controls
    assert not window.sensing_button.isEnabled()
    assert window.copy_button.isEnabled()
    window.close()


def test_structure_preserves_composites_and_edits_optional_sections():
    app = QApplication.instance() or QApplication([])
    document = {'schema_version': 2, 'mode': 'experimental', 'array': [{'value': None}, [1, 2]]}
    dialog = StructureDialog(document, {'example / noise': ('noise', {'enabled': False})})
    root = dialog.tree.topLevelItem(0)
    assert dialog.read(root) == document
    root.child(0).setText(1, '1')
    assert dialog.read(root)['schema_version'] == 1
    dialog.populate(root, 'noise', {'enabled': False})
    assert dialog.read(root)['noise']['enabled'] is False
    root.child(2).child(1).removeChild(root.child(2).child(1).child(0))
    assert dialog.read(root)['array'][1] == [2]
    dialog.reject()
