"""English local workbench; process state stays separate from physics state."""
import json
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)

from .jobs import parse_snapshot, prepare_run, read_result
from .editor import ProfileEditor

ROOT = Path(__file__).resolve().parents[1]


def text_view():
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setFont(QFont('DejaVu Sans Mono', 10))
    return view


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Matplotlib ships these fonts, including on hosts without system-font discovery.
        from matplotlib import get_data_path
        for filename in ('DejaVuSans.ttf', 'DejaVuSansMono.ttf'):
            QFontDatabase.addApplicationFont(str(Path(get_data_path()) / 'fonts' / 'ttf' / filename))
        self.setFont(QFont('DejaVu Sans', 10))
        self.setWindowTitle('RAQS — Research Workbench')
        self.resize(1120, 780)
        self.setMinimumSize(800, 580)
        self.raw = None
        self.source = None
        self.profile = None
        self.run_directory = None
        self.result_directory = None
        self.images = []
        self.active = False
        self.dirty = False
        self.process = QProcess(self)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(12)
        heading = QLabel('RAQS Research Workbench')
        heading.setFont(QFont('DejaVu Sans', 20, QFont.Weight.DemiBold))
        layout.addWidget(heading)
        subtitle = QLabel('Rydberg sensing  /  Profile inspection, isolated execution and results')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self.notice = QLabel('Choose New Profile to enter inputs, or Open Profile to import JSON. The physical model is unchanged.')
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.notice)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        profiles = QWidget()
        column = QVBoxLayout(profiles)
        row = QHBoxLayout()
        self.template_choice = QComboBox()
        self.template_choice.setAccessibleName('New profile template')
        for label, filename in (
            ('Near-resonant experimental', 'lab006/experimental_near_resonant.json'),
            ('Prescribed diagnostics', 'lab006/prescribed_reference.json'),
            ('Prescribed thermal / spectrum', 'lab007/prescribed_full.json'),
            ('Experimental thermal / spectrum', 'lab007/experimental_full.json'),
        ):
            self.template_choice.addItem(label, filename)
        self.new_button = QPushButton('&New Profile')
        self.new_button.clicked.connect(self.new_profile)
        row.addWidget(self.template_choice)
        row.addWidget(self.new_button)
        column.addLayout(row)
        row = QHBoxLayout()
        self.open_button = QPushButton('&Open Profile…')
        self.open_button.clicked.connect(self.open_profile)
        self.reload_button = QPushButton('&Reload')
        self.reload_button.clicked.connect(self.reload_profile)
        self.copy_button = QPushButton('Save &Copy…')
        self.copy_button.clicked.connect(self.save_copy)
        for button in (self.open_button, self.reload_button, self.copy_button):
            row.addWidget(button)
        row.addStretch()
        column.addLayout(row)
        self.profile_label = QLabel('No profile selected')
        self.profile_label.setWordWrap(True)
        column.addWidget(self.profile_label)
        self.profile_text = text_view()
        self.profile_text.setPlaceholderText('Choose New Profile or import JSON. The preview follows the current form.')
        self.editor = ProfileEditor()
        self.editor.changed.connect(self.form_changed)
        self.profile_views = QTabWidget()
        self.profile_views.addTab(self.editor, 'Inputs')
        self.profile_views.addTab(self.profile_text, 'JSON Preview')
        column.addWidget(self.profile_views)
        self.tabs.addTab(profiles, 'Profiles')

        diagnostics = QWidget()
        column = QVBoxLayout(diagnostics)
        self.diagnostics_text = text_view()
        self.diagnostics_text.setPlaceholderText('Resolve Diagnostics runs ARC in a fresh process. A successful parse alone does not validate physical execution.')
        column.addWidget(self.diagnostics_text)
        self.tabs.addTab(diagnostics, 'Diagnostics')

        run_page = QWidget()
        column = QVBoxLayout(run_page)
        column.addWidget(QLabel('Output parent folder — each run creates a new subfolder'))
        row = QHBoxLayout()
        self.output = QLineEdit(str(ROOT / 'outputs' / 'gui'))
        self.output.setAccessibleName('Output parent folder')
        self.browse_button = QPushButton('Browse…')
        self.browse_button.clicked.connect(self.choose_output)
        row.addWidget(self.output, 1)
        row.addWidget(self.browse_button)
        column.addLayout(row)
        row = QHBoxLayout()
        self.diagnostic_button = QPushButton('Resolve &Diagnostics')
        self.diagnostic_button.clicked.connect(lambda: self.start_run(True))
        self.sensing_button = QPushButton('Run &Sensing')
        self.sensing_button.clicked.connect(lambda: self.start_run(False))
        row.addWidget(self.diagnostic_button)
        row.addWidget(self.sensing_button)
        row.addStretch()
        column.addLayout(row)
        self.stage = QLabel('Idle — no simulation is running')
        self.stage.setWordWrap(True)
        column.addWidget(self.stage)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        column.addWidget(self.progress)
        self.log = text_view()
        self.log.setPlaceholderText('Worker output appears here. The complete log is kept in the run folder.')
        column.addWidget(self.log)
        column.addWidget(QLabel('No forced stop: allow the active solver to finish before closing this window.'))
        self.tabs.addTab(run_page, 'Run')

        results = QWidget()
        column = QVBoxLayout(results)
        row = QHBoxLayout()
        self.open_result_button = QPushButton('Open Saved Run…')
        self.open_result_button.clicked.connect(self.open_result)
        self.folder_button = QPushButton('Open Result Folder')
        self.folder_button.clicked.connect(self.open_folder)
        row.addWidget(self.open_result_button)
        row.addWidget(self.folder_button)
        row.addStretch()
        column.addLayout(row)
        self.result_label = QLabel('No completed run loaded. Results are accepted only with a valid GUI manifest.')
        self.result_label.setWordWrap(True)
        column.addWidget(self.result_label)
        self.result_tabs = QTabWidget()
        self.report_text = text_view()
        self.report_text.setPlaceholderText('The full scientific report will appear here, with the model’s original field names and units.')
        self.result_tabs.addTab(self.report_text, 'Report')
        figures = QWidget()
        figure_column = QVBoxLayout(figures)
        self.figure_choice = QComboBox()
        self.figure_choice.setAccessibleName('Result figure')
        self.figure_choice.currentIndexChanged.connect(self.show_image)
        figure_column.addWidget(self.figure_choice)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image = QLabel('No figure available. Diagnostics do not generate sensing plots.')
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.image)
        figure_column.addWidget(scroll)
        self.result_tabs.addTab(figures, 'Figures')
        column.addWidget(self.result_tabs)
        self.tabs.addTab(results, 'Results')
        self.update_controls()

    def update_controls(self):
        for control in (self.open_button, self.new_button, self.template_choice, self.editor, self.output, self.browse_button, self.open_result_button):
            control.setEnabled(not self.active)
        for control in (self.copy_button, self.diagnostic_button):
            control.setEnabled(self.raw is not None and not self.active)
        self.reload_button.setEnabled(self.source is not None and not self.active)
        self.sensing_button.setEnabled(self.profile is not None and self.profile.run_config is not None and not self.active)
        self.folder_button.setEnabled(self.result_directory is not None)

    def error(self, exc):
        self.notice.setText(f'Error: {exc}')

    def open_profile(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Open RAQS Profile', str(ROOT / 'examples'), 'JSON profiles (*.json)')
        if path:
            self.load_profile(Path(path))

    def confirm_discard(self):
        return not self.dirty or QMessageBox.question(
            self, 'Unsaved profile changes', 'Discard the unsaved changes to the current profile?',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def new_profile(self):
        if self.active or not self.confirm_discard():
            return
        self.load_profile(ROOT / 'examples' / self.template_choice.currentData())
        if self.raw is not None:
            self.source = None
            self.profile_label.setText('New profile from ' + self.template_choice.currentText())
            self.notice.setText('Template loaded. Edit Inputs directly, then save a copy or run. Template values are simulation assumptions, not hardware calibration.')
            self.update_controls()

    def reload_profile(self):
        if self.confirm_discard():
            self.load_profile(self.source)

    def load_profile(self, path):
        if path is None or self.active:
            return
        self.raw = self.profile = None
        self.dirty = False
        self.source = Path(path)
        self.profile_text.clear()
        self.diagnostics_text.clear()
        self.clear_result()
        self.editor.clear_profile()
        try:
            raw = self.source.read_bytes()
            profile = parse_snapshot(raw)
            self.editor.load_document(json.loads(raw.decode('utf-8-sig')))
            self.profile_text.setPlainText(raw.decode('utf-8-sig'))
            self.raw, self.profile = raw, profile
            self.profile_label.setText(f'{self.source}\nMode: {profile.mode}  |  Atom: {profile.atom.species if profile.atom else "Rb85 (default)"}  |  Sensing: {"available" if profile.run_config else "not declared"}')
            self.notice.setText('Profile imported into Inputs. Edit values directly; the JSON preview and execution follow the current validated form.')
        except Exception as exc:
            self.profile_label.setText(f'Profile could not be loaded: {path}')
            self.error(exc)
        self.update_controls()

    def form_changed(self):
        if self.active:
            return
        self.dirty = True
        self.raw = self.profile = None
        self.diagnostics_text.clear()
        self.clear_result()
        self.profile_label.setText(f'{self.source or "New profile"} — unsaved edits')
        self.stage.setText('Inputs changed — previous diagnostics and results cleared')
        raw = None
        try:
            raw = self.editor.snapshot()
            self.profile_text.setPlainText(raw.decode('utf-8'))
            self.profile = parse_snapshot(raw)
            self.raw = raw
            self.notice.setText('Inputs validated by the profile parser. Unsaved changes will be included in the next run snapshot.')
        except Exception as exc:
            if raw is None:
                self.profile_text.setPlainText('Draft is not executable. Correct the inputs.\n' + str(exc))
            self.error(exc)
        self.update_controls()

    def save_copy(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Profile Copy — choose a new filename', 'profile-copy.json', 'JSON profiles (*.json)')
        if path:
            try:
                with Path(path).open('xb') as stream:
                    stream.write(self.raw)
                self.source = Path(path)
                self.dirty = False
                self.profile_label.setText(str(self.source))
                self.notice.setText(f'Current validated profile saved: {path}')
                self.update_controls()
            except OSError as exc:
                self.error(exc)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Parent', self.output.text())
        if path:
            self.output.setText(path)

    def clear_result(self):
        self.result_directory = None
        self.images = []
        self.figure_choice.clear()
        self.image.clear()
        self.image.setText('No figure available.')
        self.report_text.clear()
        self.result_label.setText('No completed run loaded.')

    def start_run(self, diagnostics):
        if self.active or self.raw is None:
            return
        try:
            parent = Path(self.output.text()).expanduser()
            if not self.output.text().strip():
                raise ValueError('Choose an output parent folder.')
            directory = parent.resolve() / ('run-' + uuid4().hex)
            prepare_run(directory, self.raw, diagnostics)
        except Exception as exc:
            self.error(exc)
            return
        self.clear_result()
        self.diagnostics_text.clear()
        self.run_directory = directory
        self.active = True
        self.update_controls()
        self.stage.setText('Starting isolated worker…')
        self.notice.setText(f'Input snapshot saved. Running {"diagnostics" if diagnostics else "sensing"}: {directory.name}')
        self.log.clear()
        self.progress.setRange(0, 0)
        self.tabs.setCurrentIndex(2)
        self.process.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONIOENCODING', 'utf-8')
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setStandardOutputFile(str(directory / 'run.log'))
        self.timer.start()
        self.process.start(sys.executable, ['-u', '-m', 'raqs_gui.worker', str(directory), 'diagnostics' if diagnostics else 'sensing'])

    def poll(self):
        try:
            status = json.loads((self.run_directory / 'status.json').read_text(encoding='utf-8'))
            self.stage.setText(status['state'].capitalize() + ' — ' + str(self.run_directory))
            with (self.run_directory / 'run.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16000))
                self.log.setPlainText(stream.read().decode('utf-8', errors='replace'))
        except (OSError, ValueError, KeyError):
            pass  # The process may not have created its first status/log yet.

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finish_ui()
            self.stage.setText('Failed to start worker')
            self.error(self.process.errorString())

    def finish_ui(self):
        self.poll()
        self.timer.stop()
        self.active = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.update_controls()

    def finished(self, code, exit_status):
        self.finish_ui()
        if code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self.stage.setText('Failed / interrupted — inspect the worker log')
            self.error('The worker did not complete. No successful result has been loaded.')
            return
        try:
            self.load_result(self.run_directory)
            self.stage.setText('Completed — ' + str(self.run_directory))
            self.progress.setValue(1)
        except Exception as exc:
            self.stage.setText('Failed — result receipt is incomplete or invalid')
            self.error(exc)

    def open_result(self):
        path = QFileDialog.getExistingDirectory(self, 'Open Completed GUI Run', self.output.text())
        if path:
            try:
                self.load_result(Path(path))
            except Exception as exc:
                self.clear_result()
                self.update_controls()
                self.error(exc)

    def load_result(self, directory):
        # shortcut: hashes are checked on the UI thread for small prototype runs; move to a worker before large campaigns.
        report, images = read_result(directory)
        self.clear_result()
        self.result_directory = Path(directory)
        self.images = images
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        self.report_text.setPlainText(encoded)
        if report['diagnostics_only']:
            self.diagnostics_text.setPlainText(encoded)
        sensing = report.get('sensing')
        summary = ('Diagnostics only — no sensing executed' if sensing is None else
                   f"AoA: {sensing['estimated_aoa_deg']} deg  |  Input: {sensing['true_aoa_deg']} deg  |  IF: {report['signed_if_mhz']} MHz")
        self.result_label.setText(summary + '\n' + str(directory))
        self.figure_choice.addItems([p.stem.replace('_', ' ').title() for p in images])
        self.notice.setText('Completed run verified against its artifact hashes. This is a simulation result, not experimental validation.')
        self.tabs.setCurrentIndex(1 if report['diagnostics_only'] else 3)
        self.update_controls()

    def show_image(self, index):
        if 0 <= index < len(self.images):
            pixmap = QPixmap(str(self.images[index]))
            if pixmap.isNull():
                self.image.setText('The saved figure could not be decoded.')
            else:
                self.image.setPixmap(pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation))

    def open_folder(self):
        if self.result_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_directory)))

    def closeEvent(self, event):
        if self.active:
            self.notice.setText('A simulation is still running. Wait for completion before closing the workbench.')
            event.ignore()
        else:
            if self.confirm_discard():
                event.accept()
            else:
                event.ignore()
