"""English local workbench; process state stays separate from physics state."""
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QThreadPool
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMessageBox, QPlainTextEdit,
                              QDialog, QInputDialog, QTableWidgetItem)

from rydberg_aoa.exceptions import PhysicsConfigurationError

from .jobs import auxiliary_document, load_snapshot, parse_snapshot, prepare_run, read_result, write_json

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
        self.raw = None
        self.source = None
        self.profile = None
        self.run_directory = None
        self.result_directory = None
        self.images = {}
        self.result_profile = None
        self.result_diagnostics = False
        self.result_summary = None
        self.active = False
        self.job = None
        self.stop_requested = False
        self.worker_ready = False
        self.startup_error = None
        self.dirty = False
        self.template_name = 'New profile'
        self.profile_context = 'No profile'
        self.execution_error = None
        self.related_profiles = []
        self.current_result_matches_draft = False
        self.reading = False
        self.read_generation = 0
        self.process = QProcess(self)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)

        from .workbench import build_workbench
        self.started_at = None
        build_workbench(self, ROOT, text_view)
        available = self.screen().availableGeometry()
        self.resize(min(1380, available.width() - 36), min(880, available.height() - 64))
        self.setMinimumSize(900, 555)
        self.update_controls()
        self.new_profile()

    def update_controls(self):
        for control in (self.open_button, self.new_button, self.template_choice, self.editor, self.structure_button, self.add_option_button, self.remove_option_button, self.output, self.browse_button, self.open_result_button):
            control.setEnabled(not self.active)
        self.copy_button.setEnabled(self.raw is not None and not self.active)
        self.diagnostic_button.setEnabled(self.raw is not None and not self.active and not self.reading and self.execution_error is None)
        self.reload_button.setEnabled(self.source is not None and not self.active)
        self.sensing_button.setEnabled(self.profile is not None and self.profile.run_config is not None and not self.active and not self.reading and self.execution_error is None)
        reason = ('A saved result is being verified; wait for verification.' if self.reading else 'A calculation is already running.' if self.active else
                  self.execution_error or ('Correct the invalid profile inputs.' if self.raw is None else ''))
        self.diagnostic_button.setToolTip(reason or 'Resolve the declared atomic physics without sensing.')
        self.sensing_button.setToolTip(reason or ('Run the current measurement profile.' if self.profile and self.profile.run_config else 'This profile has no sensing execution; use Run Diagnostics.'))
        if self.active and self.stop_requested:
            state = 'Stopping · waiting for the owned process group to exit. Partial output will not be accepted.'
        elif self.active:
            state = 'Running · input snapshot locked. You can inspect the monitor; configuration is disabled until completion.'
        elif reason:
            state = 'Not runnable · ' + reason
        elif self.profile and self.profile.run_config is None:
            state = 'Parsed · diagnostics ready. No sensing execution is declared; choose a calculation template to run sensing.'
        elif self.current_result_matches_draft:
            state = 'Executed / resolved · the displayed verified result matches the current input snapshot.'
        else:
            receiver = self.profile.measurement.mode if self.profile and self.profile.measurement else 'aoa'
            state = f'Parsed · ready for {receiver.upper()} calculation or diagnostics. Physical resolution has not yet been performed for this draft.'
        self.eligibility.setText(state)
        self.folder_button.setEnabled(self.result_directory is not None)
        self.export_button.setEnabled(bool(self.images))
        self.stop_button.setEnabled(self.active and sys.platform == 'win32' and not self.stop_requested)
        self.stop_button.setToolTip('Force-stop this calculation and its owned worker descendants; partial output will not be accepted.' if sys.platform == 'win32' else 'Verified process-tree stop is currently Windows-only.')
        self.related_button.setEnabled(bool(self.related_profiles) and not self.active)
        self.related_button.setVisible(bool(self.related_profiles))
        self.open_result_button.setEnabled(not self.active and not self.reading)

    def message(self, text, severity='info'):
        self.notice.setText(text)
        self.notice.setProperty('severity', severity)
        self.notice.style().unpolish(self.notice)
        self.notice.style().polish(self.notice)

    def error(self, exc):
        self.message(f'Error: {exc}', 'error')

    def open_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open RAQS Profile', str(ROOT / 'examples'), 'JSON profiles (*.json)')
        if path:
            self.load_profile(path, confirm=True)

    def confirm_discard(self):
        return not self.dirty or QMessageBox.question(
            self, 'Unsaved profile changes', 'Discard the unsaved changes to the current profile?',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def new_profile(self):
        if self.load_profile(ROOT / 'examples' / self.template_choice.currentData(), confirm=True, template=True):
            self.message('Template loaded. Edit the inputs, then save a copy or run. Template values are simulation assumptions.')

    def reload_profile(self):
        self.load_profile(self.source, confirm=True)

    def refresh_profile_label(self):
        name = self.source.name if self.source else self.template_name
        state = 'Unsaved edits' if self.dirty else 'Saved' if self.source else 'Template'
        if self.profile:
            species = self.profile.atom.species if self.profile.atom else 'Rb85 (default)'
            context = f'{self.profile.mode} · {species}'
        else:
            context = self.profile_context
        self.profile_label.setText(f'{name}  ·  {context}  ·  {state}')
        self.profile_label.setToolTip(str(self.source) if self.source else name)

    def load_profile(self, path, *, confirm=False, template=False):
        if path is None or self.active:
            return False
        source = Path(path)
        try:
            raw = source.read_bytes()
            preview = raw.decode('utf-8-sig')
            auxiliary = auxiliary_document(raw, source, ROOT)
            profile = None if auxiliary else load_snapshot(raw)
            document = json.loads(preview)
            execution_error = None
            if profile and (profile.study is not None or profile.magnetic_core is not None):
                try:
                    parse_snapshot(raw)
                except ValueError as exc:
                    execution_error = str(exc)
        except (OSError, ValueError, PhysicsConfigurationError) as exc:
            self.error(f'Could not open {source.name}: {exc}')
            return False
        if confirm and not self.confirm_discard():
            return False

        self.source = None if template else source
        self.template_name = self.template_choice.currentText()
        self.dirty = False
        self.raw = raw if profile else None
        self.profile = profile
        self.related_profiles = []
        self.execution_error = execution_error
        self.profile_text.setPlainText(preview)
        self.read_generation += 1
        if auxiliary:
            message, self.related_profiles = auxiliary
            self.profile_context = 'Research document'
            self.execution_error = 'Research document. Choose a referenced profile or import a run profile.'
            self.editor.clear_profile()
            self.preview_toggle.setChecked(True)
            self.message(message)
        else:
            self.editor.load_document(document)
            self.profile_context = f'{profile.mode} · {profile.atom.species if profile.atom else "Rb85 (default)"}'
            self.message(f'Profile loaded for editing. Execution unavailable: {execution_error}' if execution_error
                         else 'Profile loaded. Edit inputs, then save a copy or run.')
        self.refresh_profile_label()
        self.refresh_result_context()
        self.update_controls()
        return True

    def open_related_profile(self):
        if self.active or not self.related_profiles:
            return
        paths = [str(path) for path in self.related_profiles]
        selected, ok = QInputDialog.getItem(self, 'Referenced Run Profile', 'Select the input to open (no conversion):', paths, 0, False)
        if ok:
            self.load_profile(Path(selected), confirm=True)

    def form_changed(self):
        if self.active:
            return
        self.dirty = True
        self.notice.setProperty('severity', 'normal')
        self.execution_error = None
        self.raw = self.profile = None
        self.read_generation += 1
        if self.result_directory is None:
            self.stage.setText('Inputs changed — ready to validate')
        raw = None
        try:
            raw = self.editor.snapshot()
            self.profile_text.setPlainText(raw.decode('utf-8'))
            self.profile = load_snapshot(raw)
            self.raw = raw
            self.message('Inputs validated by the profile parser. Unsaved changes will be included in the next run snapshot.')
            if self.profile.study is not None or self.profile.magnetic_core is not None:
                try:
                    parse_snapshot(raw)
                except ValueError as exc:
                    self.execution_error = str(exc)
                    self.message(f'Valid editable profile; execution unavailable: {exc}')
            self.profile_context = f'{self.profile.mode} · {self.profile.atom.species if self.profile.atom else "Rb85 (default)"}'
        except Exception as exc:
            if raw is None:
                self.profile_text.setPlainText('Draft is not executable. Correct the inputs.\n' + str(exc))
            self.error(exc)
        self.refresh_profile_label()
        self.refresh_result_context()
        self.update_controls()

    def edit_structure(self):
        if self.active:
            return
        from .structure import StructureDialog
        try:
            document = json.loads(self.editor.snapshot())
        except (ValueError, TypeError) as exc:
            self.error(exc)
            return
        dialog = StructureDialog(document, lambda: self.option_presets(all_sections=True), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.editor.load_document(dialog.result_document)
            self.form_changed()

    def option_presets(self, *, all_sections=False):
        allowed = {'atom', 'environment', 'spectrum', 'execution', 'numerics', 'noise', 'temporal_doppler',
                   'radiative', 'lo_phase_reference', 'measurement', 'provenance', 'study', 'magnetic_core'}
        presets = {}
        for path in sorted((ROOT / 'examples').rglob('*.json')):
            try:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict) or 'fields' not in value or 'mode' not in value:
                continue
            keys = value.keys() if all_sections else allowed & value.keys()
            for key in keys:
                if all_sections or value[key] is not None:
                    presets.setdefault(key, {})[str(path.relative_to(ROOT / 'examples'))] = value[key]
        return presets

    def add_option(self):
        try:
            document = json.loads(self.editor.snapshot())
            presets = self.option_presets()
            section, ok = QInputDialog.getItem(self, 'Add or Replace Effect', 'Section (core validates compatibility):', sorted(presets), 0, False)
            if not ok: return
            source, ok = QInputDialog.getItem(self, 'Choose Explicit Starting Values', 'Source example; values are assumptions, not calibration:', sorted(presets[section]), 0, False)
            if not ok: return
            if section in document and QMessageBox.question(self, 'Replace Section', f'Replace the current {section} settings with the selected example?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            document[section] = presets[section][source]
            self.editor.load_document(document); self.form_changed()
        except (ValueError, TypeError) as exc:
            self.error(exc)

    def remove_option(self):
        try:
            document = json.loads(self.editor.snapshot())
            optional = sorted(set(document) & {'noise', 'numerics', 'temporal_doppler', 'radiative', 'lo_phase_reference', 'measurement', 'provenance', 'study', 'magnetic_core', 'atom', 'spectrum'})
            if not optional:
                self.message('No optional effect is currently configured.'); return
            section, ok = QInputDialog.getItem(self, 'Remove Optional Effect', 'Removing a required section may invalidate the draft:', optional, 0, False)
            if not ok: return
            if section == 'spectrum' and document.get('schema_version') == 2:
                document[section] = None
            else:
                del document[section]
            self.editor.load_document(document); self.form_changed()
        except (ValueError, TypeError) as exc:
            self.error(exc)

    def save_copy(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Profile Copy — choose a new filename', 'profile-copy.json', 'JSON profiles (*.json)')
        if path:
            try:
                with Path(path).open('xb') as stream:
                    stream.write(self.raw)
                self.source = Path(path)
                self.dirty = False
                self.refresh_profile_label()
                self.message(f'Current validated profile saved: {path}')
                self.update_controls()
            except OSError as exc:
                self.error(exc)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Parent', self.output.text())
        if path:
            self.output.setText(path)

    def clear_result(self):
        if self.figure_focus.isChecked(): self.figure_focus.setChecked(False)
        self.read_generation += 1
        self.current_result_matches_draft = False
        self.result_directory = None
        self.images = {}
        self.result_profile = None
        self.result_diagnostics = False
        self.result_summary = None
        self.figure_choice.clear()
        self.image.clear()
        self.image.setText('No figure available.')
        self.report_text.clear()
        self.summary_table.setRowCount(0)
        self.result_label.setText('No completed run loaded.')
        self.figure_feedback.clear()
        self.figure_feedback.hide()

    def start_run(self, diagnostics):
        if self.active or self.reading or self.raw is None:
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
        self.figure_focus.setChecked(False)
        self.run_directory = directory
        self.active = True
        self.stop_requested = False
        self.worker_ready = False
        self.startup_error = None
        self.started_at = time.monotonic()
        self.update_controls()
        self.stage.setText('Starting isolated worker…')
        self.message(f'Input snapshot saved. Running {"diagnostics" if diagnostics else "sensing"}: {directory.name}')
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
        arguments = ['-u', '-m', 'raqs_gui.worker', str(directory), 'diagnostics' if diagnostics else 'sensing']
        if sys.platform == 'win32': arguments.append('--managed')
        self.process.start(sys.executable, arguments)

    def authorize_worker(self):
        ready = self.run_directory / 'worker_ready.json'
        if not ready.exists(): return
        from .process_job import ProcessJob
        info = json.loads(ready.read_text(encoding='utf-8'))
        root_pid = self.process.processId()
        if type(info.get('pid')) is not int or (info['pid'] != root_pid and info.get('parent_pid') != root_pid):
            raise RuntimeError('Worker startup identity does not match this calculation.')
        job = ProcessJob(info['job_name'])
        try:
            if not job.contains(info['pid']):
                raise RuntimeError('The worker is not owned by the announced cleanup group.')
        except Exception:
            job.close(); raise
        self.job = job
        self.worker_ready = True
        if self.stop_requested:
            job.terminate()
        elif self.process.write(b'RAQS_START\n') < 0:
            raise RuntimeError('Could not authorize the owned calculation worker.')

    def stop_run(self):
        if not self.active or self.stop_requested or sys.platform != 'win32': return
        self.stop_requested = True
        self.update_controls()
        self.stage.setText('Stopping — waiting for owned processes to exit')
        self.message('Stop requested. Partial output is retained, but will not be accepted as a completed result.')
        if self.job is not None:
            try:
                self.job.terminate()
            except OSError as exc:
                self.stop_requested = False
                self.update_controls()
                self.error(f'Stop could not terminate the process group: {exc}')

    def check_stopped(self):
        if self.job is None: return
        try:
            if self.job.active_processes(): return
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()  # The owned job is empty; only its launcher can remain.
                return
            receipt_error = None
            try:
                write_json(self.run_directory / 'interrupted.json', {'state': 'stopped', 'reason': 'User requested forced stop; owned process group verified empty.'})
                write_json(self.run_directory / 'status.json', {'schema_version': 1, 'state': 'stopped'})
            except OSError as exc:
                receipt_error = str(exc)
            self.finish_ui()
            self.stage.setText('Stopped — owned calculation processes have exited')
            if receipt_error:
                self.error(f'Processes stopped, but interruption metadata could not be saved: {receipt_error}. Do not treat this folder as a completed run.')
            else:
                self.message('Calculation stopped. Partial files remain in the run folder and are not a completed result.')
        except (OSError, ValueError) as exc:
            self.error(f'Stop cleanup could not be verified: {exc}')

    def read_latest_log(self):
        if self.run_directory is None: return
        try:
            with (self.run_directory / 'run.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16000))
                tail = stream.read().decode('utf-8', errors='replace')
        except OSError:
            return
        if tail != self.log.toPlainText():
            scroll = self.log.verticalScrollBar()
            previous = scroll.value()
            self.log.setPlainText(tail)
            scroll.setValue(scroll.maximum() if self.follow_log.isChecked() else previous)

    def poll(self):
        self.read_latest_log()
        if self.started_at is not None:
            self.elapsed_label.setText(f'Elapsed: {time.monotonic() - self.started_at:.1f} s · measured wall time')
        if self.active and self.process.state() != QProcess.ProcessState.NotRunning and sys.platform == 'win32' and not self.worker_ready:
            try:
                self.authorize_worker()
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                self.process.closeWriteChannel()
                self.startup_error = f'Calculation startup could not establish safe process ownership: {exc}'
                self.error(self.startup_error)
                return
        try:
            if self.stop_requested:
                self.check_stopped()
                return
            status = json.loads((self.run_directory / 'status.json').read_text(encoding='utf-8'))
            self.stage.setText(status['state'].capitalize() + ' — ' + str(self.run_directory))
        except (OSError, ValueError, KeyError):
            pass  # The process may not have created its first status/log yet.

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finish_ui()
            self.stage.setText('Failed to start worker')
            self.error(self.process.errorString())

    def finish_ui(self):
        self.read_latest_log()
        if not self.stop_requested: self.poll()
        self.timer.stop()
        self.active = False
        self.started_at = None
        if self.job is not None:
            self.job.close(); self.job = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.update_controls()

    def finished(self, code, exit_status):
        if self.stop_requested and self.job is not None:
            self.check_stopped()
            return
        self.finish_ui()
        if code != 0 or exit_status != QProcess.ExitStatus.NormalExit or self.startup_error:
            self.stage.setText('Failed / interrupted — inspect the worker log')
            detail = self.startup_error or 'The worker did not complete. No successful result has been loaded.'
            try:
                detail = json.loads((self.run_directory / 'status.json').read_text(encoding='utf-8')).get('error', detail)
            except (OSError, ValueError):
                pass
            self.error(detail)
            return
        try:
            self.request_result(self.run_directory)
        except Exception as exc:
            self.stage.setText('Failed — result receipt is incomplete or invalid')
            self.error(exc)

    def open_result(self):
        path = QFileDialog.getExistingDirectory(self, 'Open Completed GUI Run', self.output.text())
        if path:
            try:
                self.request_result(Path(path))
            except Exception as exc:
                self.error(exc)

    def load_result(self, directory):
        """Synchronous entry for callers/tests; UI events use request_result."""
        self.display_result(directory, *read_result(directory, capture=True))

    def request_result(self, directory):
        if self.reading or self.active:
            return
        from .result_reader import ReadResult
        self.reading = True
        self.read_generation += 1
        generation = self.read_generation
        self.stage.setText('Verifying saved artifacts — ' + str(directory))
        self.message('Reading and verifying the completed run. No result is accepted before its files pass verification.')
        task = ReadResult(generation, directory)
        task.signals.complete.connect(self.result_verified)
        task.signals.failed.connect(self.result_rejected)
        self.reader_directory = directory
        self.reader_task = task
        self.update_controls()
        QThreadPool.globalInstance().start(task)

    def result_verified(self, generation, result):
        self.reading = False
        if generation != self.read_generation:
            self.update_controls(); return
        try:
            self.display_result(self.reader_directory, *result)
            self.stage.setText('Completed — ' + str(self.reader_directory))
            self.progress.setValue(1)
        except Exception as exc:
            self.error(exc)
            self.stage.setText('Failed — result could not be displayed')
        self.update_controls()

    def result_rejected(self, generation, message):
        self.reading = False
        if generation == self.read_generation:
            self.error(message)
            self.stage.setText('Failed — incomplete, incompatible or modified result')
        self.update_controls()

    def refresh_result_context(self):
        self.current_result_matches_draft = self.raw is not None and self.raw == self.result_profile
        if self.result_directory is not None:
            relation = 'Matches current input' if self.current_result_matches_draft else 'Different input snapshot'
            run_name = self.result_directory.name
            if run_name.startswith('run-'):
                run_name = run_name[:12]
            self.result_label.setText(f'{self.result_summary}  ·  {run_name}  ·  {relation}')
            self.result_label.setToolTip(str(self.result_directory))
        if self.result_diagnostics and self.current_result_matches_draft:
            self.diagnostics_text.setPlainText(self.report_text.toPlainText())
            self.diagnostics_summary.setText(f'{self.result_summary}. No sensing was executed.')
        else:
            self.diagnostics_text.clear()
            self.diagnostics_summary.setText('No diagnostics for the current input. Run Diagnostics to resolve its declared physics.')

    def display_result(self, directory, report, profile, images):
        sensing = report.get('sensing')
        summary = (f"Diagnostics · {report.get('atom', {}).get('species', 'Atom')} · IF {report.get('signed_if_mhz', 'not reported')} MHz" if sensing is None else
                   f"AoA: {sensing['estimated_aoa_deg']} deg  |  Input: {sensing['true_aoa_deg']} deg  |  IF: {report['signed_if_mhz']} MHz")
        if not report['diagnostics_only'] and report.get('measurement', {}).get('mode') == 'if_only':
            summary = f"IF-only measurement  |  IF: {report['signed_if_mhz']} MHz  |  DoA not requested"
        rows = [('Measurement', 'Diagnostics only' if report['diagnostics_only'] else report.get('measurement', {}).get('mode', 'aoa').upper()),
                ('Atomic species', report.get('atom', {}).get('species', 'Not reported')),
                ('Resolution mode', report.get('mode', 'Not reported')),
                ('Signed IF [MHz]', report.get('signed_if_mhz', 'Not reported'))]
        if sensing:
            rows.extend((('Paths', sensing.get('paths')), ('Samples per path', sensing.get('samples_per_path')),
                        ('Duration [us]', sensing.get('duration_us')), ('Sample step [us]', sensing.get('sample_step_us')),
                        ('Estimator', sensing.get('execution', {}).get('doa_method') if report.get('measurement', {}).get('mode') != 'if_only' else 'Not requested')))
            if report.get('measurement', {}).get('mode') != 'if_only':
                rows.extend((('Input azimuth [deg]', sensing.get('true_aoa_deg')), ('Estimated azimuth [deg]', sensing.get('estimated_aoa_deg'))))
        for key in ('optical_energy_convention', 'decay_rate_basis', 'noise_status', 'noise_included', 'zeeman_included', 'doppler_ensemble_included', 'temporal_doppler_ensemble_included', 'prepared_single_channel_assumption'):
            if key in report: rows.append((key.replace('_', ' ').capitalize(), report[key]))
        rows.extend((('Output verification', 'Required artifacts match the manifest hashes'),
                     ('Scientific scope', 'Simulation result; no experimental validation is inferred')))
        # Prepare the display before replacing the current result.
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        self.clear_result()
        self.result_directory = Path(directory)
        self.result_profile = profile
        self.result_diagnostics = report['diagnostics_only']
        self.result_summary = summary
        self.images = images
        self.report_text.setPlainText(encoded)
        self.refresh_result_context()
        self.summary_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.summary_table.setItem(index, 0, QTableWidgetItem(name))
            self.summary_table.setItem(index, 1, QTableWidgetItem(str(value)))
        self.summary_table.verticalHeader().setMinimumSectionSize(26)
        self.summary_table.verticalHeader().setDefaultSectionSize(28)
        self.figure_choice.addItems([Path(name).stem.replace('_', ' ').title() for name in images])
        self.message('Completed run verified against its artifact hashes. This is a simulation result, not experimental validation.')
        self.result_tabs.setCurrentIndex(0 if images else 1)
        self.tabs.setCurrentIndex(1 if self.result_diagnostics and self.current_result_matches_draft else 3)
        self.update_controls()

    def show_image(self, index):
        if 0 <= index < len(self.images):
            name = list(self.images)[index]
            pixmap = QPixmap()
            pixmap.loadFromData(self.images[name], 'PNG')
            if pixmap.isNull():
                self.image.setText('The saved figure could not be decoded.')
            else:
                zoom = self.figure_zoom.currentText()
                if zoom == 'Fit page':
                    available = self.figure_scroll.viewport().size()
                    available.setWidth(max(200, available.width() - 12)); available.setHeight(max(180, available.height() - 12))
                    displayed = pixmap.scaled(available, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                else:
                    width = max(240, self.figure_scroll.viewport().width() - 12) if zoom == 'Fit width' else int(pixmap.width() * int(zoom.rstrip('%')) / 100)
                    displayed = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
                self.image.setMinimumSize(0, 0) if zoom.startswith('Fit') else self.image.setMinimumSize(displayed.size())
                self.image.setPixmap(displayed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'figure_zoom') and self.figure_zoom.currentText().startswith('Fit'):
            QTimer.singleShot(0, lambda: self.show_image(self.figure_choice.currentIndex()))

    def set_figure_focus(self, focused):
        for control in (self.navigation, self.page_headings[3], self.result_actions, self.bottom_status):
            control.setVisible(not focused)
        QTimer.singleShot(0, lambda: self.show_image(self.figure_choice.currentIndex()))

    def save_figure(self):
        index = self.figure_choice.currentIndex()
        if not 0 <= index < len(self.images):
            return
        name = list(self.images)[index]
        path, _ = QFileDialog.getSaveFileName(self, 'Save Scientific Figure Copy', name, 'PNG image (*.png)')
        if path:
            try:
                with Path(path).open('xb') as stream:
                    stream.write(self.images[name])
                feedback = f'Figure saved to {Path(path).name}.'
                self.message(feedback)
                severity = 'info'
            except OSError as exc:
                detail = ('A file already exists. Choose another name.' if isinstance(exc, FileExistsError)
                          else exc.strerror or str(exc))
                feedback = f'Could not save figure. {detail}'
                self.error(feedback)
                severity = 'error'
            self.figure_feedback.setProperty('severity', severity)
            self.figure_feedback.style().unpolish(self.figure_feedback)
            self.figure_feedback.style().polish(self.figure_feedback)
            self.figure_feedback.setText(feedback)
            self.figure_feedback.show()
            self.show_image(index)

    def open_folder(self):
        if self.result_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_directory)))

    def closeEvent(self, event):
        if self.active or self.reading:
            self.message('A calculation or result verification is active. Wait for completion before closing the workbench.')
            event.ignore()
        else:
            if self.confirm_discard():
                event.accept()
            else:
                event.ignore()
