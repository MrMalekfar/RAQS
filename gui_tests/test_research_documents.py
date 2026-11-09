import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from raqs_gui.jobs import auxiliary_document
from raqs_gui.window import MainWindow, ROOT


def test_design_offers_explicit_existing_profile_without_conversion(tmp_path):
    profile = tmp_path / 'point_000.json'
    raw = (ROOT / 'examples/lab006/prescribed_reference.json').read_bytes()
    profile.write_bytes(raw)
    design = tmp_path / 'design.json'
    design.write_text(json.dumps({'task': 'research', 'profiles': ['point_000.json']}))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_profile(design)
    assert window.raw is None and not window.diagnostic_button.isEnabled()
    assert window.related_profiles == [profile.resolve()]
    assert window.related_button.isEnabled()
    assert 'not a run profile' in window.notice.text()
    assert json.loads(window.profile_text.toPlainText()) == json.loads(design.read_bytes())
    window.load_profile(profile)
    assert window.raw == raw and window.diagnostic_button.isEnabled()
    assert not window.related_profiles and not window.related_button.isEnabled()
    window.close()


def test_missing_research_reference_is_reported_without_substitution(tmp_path):
    raw = json.dumps({'task': 'analysis', 'profile': 'examples/old/profile.json'}).encode()
    message, paths = auxiliary_document(raw, tmp_path / 'design.json', tmp_path)
    assert paths == [] and 'Missing referenced' in message
    assert 'examples/old/profile.json' in message
    assert auxiliary_document(b'{}', tmp_path / 'invalid.json', tmp_path) is None
