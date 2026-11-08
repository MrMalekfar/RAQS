"""Typed profile controls; scientific validation remains in the core parser."""
import copy
import json
import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
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


class ProfileEditor(QTabWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.document = None
        self.readers = {}
        self.controls = {}
        self.default_atom = None

    def clear_profile(self):
        while self.count():
            page = self.widget(0)
            self.removeTab(0)
            page.deleteLater()
        self.document = None
        self.readers = {}
        self.controls = {}
        self.default_atom = None

    def load_document(self, document):
        self.clear_profile()
        self.document = copy.deepcopy(document)
        sections = [('General', {key: value for key, value in document.items()
                                 if not isinstance(value, (dict, list))})]
        sections.extend((key, value) for key, value in document.items() if isinstance(value, (dict, list)))
        if 'atom' not in document:
            from rydberg_aoa.config.atom_species import LEGACY_ATOM
            self.default_atom = {'species': LEGACY_ATOM.species, 'isotope_fraction': LEGACY_ATOM.isotope_fraction}
            sections.append(('atom', self.default_atom))
        for name, value in sections:
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setContentsMargins(16, 16, 16, 16)
            if name == 'General':
                for key, scalar in value.items():
                    form.addRow(key, self.make_control((key,), scalar))
            else:
                self.add_fields(form, (name,), value)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll, name.replace('_', ' ').title())
        if self.count() > 1:
            self.setCurrentIndex(1)

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
                    form.addRow(key + (f' [{unit_for(child_path)}]' if unit_for(child_path) else ''),
                                self.make_control(child_path, child))
        else:
            form.addRow(self.make_control(path, value))

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
            if key in ('mode', 'schema_version') or value is None:
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
        table.setMinimumHeight(100)
        table.setMaximumHeight(200)
        table.horizontalHeader().setStretchLastSection(True)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))
        def read():
            values = [[number(table.item(i, j).text()) for j, original in enumerate(row)]
                      for i, row in enumerate(rows)]
            return values[0] if vector else values
        self.readers[path] = read
        table.cellChanged.connect(lambda *_: self.changed.emit())
        return table

    def snapshot(self):
        if self.document is None:
            raise ValueError('Open a profile or choose New Profile first.')
        result = copy.deepcopy(self.document)
        values = {}
        for path, reader in self.readers.items():
            try:
                values[path] = reader()
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{".".join(map(str, path))}: {exc}') from exc
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
