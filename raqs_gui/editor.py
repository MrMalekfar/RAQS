"""Typed profile controls; scientific validation remains in the core parser."""
import copy
import json
import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QStackedWidget, QPushButton, QHBoxLayout, QListWidget,
)


def unit_for(path):
    name = next((part for part in reversed(path) if isinstance(part, str)), '')
    for suffix, unit in (('_per_us', '1/us'), ('_mhz', 'MHz'), ('_hz', 'Hz'),
                         ('_nm', 'nm'), ('_mw', 'mW'), ('_mm', 'mm'),
                         ('_us', 'us'), ('_deg', 'deg'), ('_m', 'm'),
                         ('_k', 'K'), ('_vm', 'V/m')):
        if name.endswith(suffix):
            return unit
    return ''


def number(text):
    value = json.loads(text)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Enter a finite number using a decimal point or scientific notation.')
    return value


SECTION_NAMES = {'General': 'Profile contract', 'RF local oscillator': 'RF Local Oscillator', 'RF signal': 'RF Signal',
    'lo_phase_reference': 'LO phase reference', 'atom': 'Atom & isotope', 'preparation': 'Prepared channel',
    'ladder': 'Prepared atomic states', 'execution': 'Array & acquisition', 'bounds': 'Hardware search limits',
    'environment': 'Thermal & relaxation', 'spectrum': 'EIT spectrum', 'noise': 'Noise & detector',
    'numerics': 'Solver controls', 'temporal_doppler': 'Temporal Doppler', 'radiative': 'Radiative rates',
    'measurement': 'Measurement mode', 'quantization_axis': 'Quantization axis', 'provenance': 'Input provenance',
    'study': 'Parameter study (execution unavailable)', 'magnetic_core': 'Magnetic core (execution unavailable)'}


def field_label(key):
    exact = {'f': 'F', 'mf': 'mF', 'j': 'j', 'mj': 'mJ', 'n': 'n', 'l': 'l',
             'beam_waist_mm': 'Beam radius (1/e² intensity)', 'polarization': 'Polarization q',
             'rydberg1_mj': 'Rydberg 1 mJ', 'rydberg2_mj': 'Rydberg 2 mJ'}
    if key in exact:
        return exact[key]
    for suffix in ('_per_us', '_mhz', '_hz', '_nm', '_mw', '_mm', '_us', '_deg', '_m', '_k', '_vm'):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
            break
    return key.replace('_', ' ').capitalize()


class ProfileEditor(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.document = None
        self.readers = {}
        self.controls = {}
        self.default_atom = None
        self.errors = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.sections = QListWidget()
        self.sections.setObjectName('configurationSections')
        self.sections.setAccessibleName('Scientific configuration sections')
        self.sections.setFixedWidth(180)
        self.sections.setWordWrap(True)
        self.pages = QStackedWidget()
        self.sections.currentRowChanged.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.sections)
        layout.addWidget(self.pages, 1)

    def clear_profile(self):
        self.sections.clear()
        while self.pages.count():
            page = self.pages.widget(0)
            self.pages.removeWidget(page)
            page.deleteLater()
        self.document = None
        self.readers = {}
        self.controls = {}
        self.default_atom = None
        self.errors = {}

    def load_document(self, document):
        self.clear_profile()
        self.document = copy.deepcopy(document)
        sections = [('General', (), {key: value for key, value in document.items() if not isinstance(value, (dict, list))})]
        for key, value in document.items():
            if key == 'fields':
                sections.extend((title, ('fields', role), value[role]) for role, title in (
                    ('probe', 'Probe laser'), ('coupling', 'Coupling laser'), ('lo', 'RF local oscillator'), ('signal', 'RF signal')))
            elif isinstance(value, (dict, list)):
                sections.append((key, (key,), value))
        if 'atom' not in document:
            from rydberg_aoa.config.atom_species import LEGACY_ATOM
            self.default_atom = {'species': LEGACY_ATOM.species, 'isotope_fraction': LEGACY_ATOM.isotope_fraction}
            sections.append(('atom', ('atom',), self.default_atom))
        for name, path, value in sections:
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setContentsMargins(12, 12, 12, 12)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            if name == 'General':
                for key, scalar in value.items():
                    form.addRow(field_label(key), self.control_row((key,), scalar))
            else:
                self.add_fields(form, path, value)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
            self.sections.addItem(SECTION_NAMES.get(name, name.replace('_', ' ').title()))
        if self.pages.count() > 1:
            self.sections.setCurrentRow(1)

    def add_fields(self, form, path, value):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, key)
                if isinstance(child, dict):
                    group = QGroupBox(key.replace('_', ' ').title())
                    nested = QFormLayout(group)
                    self.add_fields(nested, child_path, child)
                    form.addRow(group)
                else:
                    form.addRow(field_label(key) + (f' [{unit_for(child_path)}]' if unit_for(child_path) else ''),
                                self.control_row(child_path, child))
        else:
            form.addRow(self.control_row(path, value))

    def control_row(self, path, value):
        control = self.make_control(path, value)
        wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(3)
        layout.addWidget(control)
        if isinstance(control, QTableWidget):
            row = QHBoxLayout()
            vector = bool(control.property('vector'))
            duplicate = QPushButton('Duplicate value' if vector else 'Duplicate row')
            remove = QPushButton('Remove value' if vector else 'Remove row')
            def duplicate_row():
                if vector:
                    index = control.columnCount()
                    selected = max(0, control.currentColumn())
                    text = control.item(0, selected).text() if index else '0'
                    control.blockSignals(True)
                    control.insertColumn(index)
                    control.setItem(0, index, QTableWidgetItem(text))
                    control.blockSignals(False); self.changed.emit()
                    return
                previous = max(0, control.currentRow())
                data = [control.item(previous, j).text() for j in range(control.columnCount())] if control.rowCount() else ['0'] * control.columnCount()
                control.blockSignals(True)
                index = control.rowCount(); control.insertRow(index)
                for j, text in enumerate(data): control.setItem(index, j, QTableWidgetItem(text))
                control.blockSignals(False); self.changed.emit()
            def remove_row():
                if vector:
                    if control.columnCount(): control.removeColumn(max(0, control.currentColumn())); self.changed.emit()
                elif control.rowCount(): control.removeRow(max(0, control.currentRow())); self.changed.emit()
            duplicate.clicked.connect(duplicate_row); remove.clicked.connect(remove_row)
            row.addWidget(duplicate); row.addWidget(remove); row.addStretch(); layout.addLayout(row)
        error = QLabel(); error.setObjectName('fieldError'); error.setWordWrap(True); error.hide()
        self.errors[path] = error; layout.addWidget(error)
        control.setToolTip('.'.join(map(str, path)))
        return wrapper

    def make_control(self, path, value):
        key = path[-1]
        choices = None
        if type(value) is bool:
            choices = [('Enabled', True), ('Disabled', False)]
        elif key == 'polarization':
            choices = [('q = -1', -1), ('q = 0', 0), ('q = +1', 1)]
        elif key == 'species':
            from rydberg_aoa.config.atom_species import species_capabilities
            choices = [(name, name) for name in species_capabilities()['species']]
        elif key == 'doa_method':
            choices = [(name, name) for name in ('phase_beamforming', 'bartlett', 'mvdr', 'capon', 'music', 'esprit', 'root_music')]
        elif path == ('environment', 'rate_source'):
            choices = [('Explicit rates', 'explicit'), ('Thermal budget', 'thermal_budget')]
        elif path == ('measurement', 'mode'):
            choices = [('Angle of arrival', 'aoa'), ('IF-only', 'if_only')]
        if choices is not None:
            control = QComboBox()
            for label, data in choices:
                control.addItem(label, data)
            index = next((i for i, (_, data) in enumerate(choices) if type(data) is type(value) and data == value), -1)
            if index < 0:
                control.addItem(str(value), value)
                index = control.count() - 1
            control.setCurrentIndex(index)
            self.readers[path] = control.currentData
            control.currentIndexChanged.connect(lambda _: self.changed.emit())
        elif isinstance(value, list) and value and all(type(v) in (int, float) for v in value):
            control = self.number_table(path, [value], vector=True)
        elif isinstance(value, list) and value and all(isinstance(row, list) and len(row) == len(value[0]) and all(type(v) in (int, float) for v in row) for row in value):
            control = self.number_table(path, value, vector=False)
        elif isinstance(value, (dict, list)):
            # shortcut: irregular composite arrays remain read-only until a dedicated structural editor is needed.
            control = QLabel(json.dumps(value, ensure_ascii=False))
            control.setWordWrap(True)
        else:
            control = QLineEdit(str(value) if value is not None else 'null')
            if path == ('noise', 'readout', 'model'):
                control.setReadOnly(True)
                control.setToolTip('Use Add / replace option or Advanced structure to change the model and its required parameters together.')
            elif key in ('mode', 'schema_version') or value is None:
                control.setReadOnly(True)
                control.setToolTip('Choose a different New Profile template or import a profile to change its structure.')
            else:
                self.readers[path] = (lambda c=control: number(c.text())) if type(value) in (int, float) else control.text
                control.textChanged.connect(lambda _: self.changed.emit())
        control.setAccessibleName('.'.join(map(str, path)))
        self.controls[path] = control
        return control

    def number_table(self, path, rows, *, vector):
        table = QTableWidget(len(rows), len(rows[0]))
        table.setProperty('vector', vector)
        table.setMinimumHeight(100)
        table.setMaximumHeight(200)
        table.horizontalHeader().setStretchLastSection(True)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))
        def read():
            values = [[number(table.item(i, j).text()) for j in range(table.columnCount())]
                      for i in range(table.rowCount())]
            return values[0] if vector and values else values
        self.readers[path] = read
        table.cellChanged.connect(lambda *_: self.changed.emit())
        return table

    def snapshot(self):
        if self.document is None:
            raise ValueError('Open a profile or choose New Profile first.')
        result = copy.deepcopy(self.document)
        values = {}
        failures = []
        for path, reader in self.readers.items():
            self.errors[path].hide()
            self.controls[path].setProperty('invalid', False)
            try:
                values[path] = reader()
            except (ValueError, OverflowError) as exc:
                failures.append(f'{".".join(map(str, path))}: {exc}')
                self.errors[path].setText('Enter a finite number; use a decimal point or scientific notation.')
                self.errors[path].show()
                self.controls[path].setProperty('invalid', True)
            self.controls[path].style().unpolish(self.controls[path]); self.controls[path].style().polish(self.controls[path])
        if failures:
            raise ValueError(failures[0])
        if self.default_atom is not None:
            atom = {key: values[('atom', key)] for key in self.default_atom}
            if atom != self.default_atom:
                result['atom'] = atom
        for path, value in values.items():
            if path[0] == 'atom' and 'atom' not in result:
                continue
            parent = result
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
        return (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
