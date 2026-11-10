"""Native structural editing for all JSON profile values, without editing JSON text."""
import copy
import json

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QStyledItemDelegate,
    QLineEdit, QAbstractItemView,
)


from rydberg_aoa.exceptions import PhysicsConfigurationError

from .jobs import load_snapshot


def reject_constant(token):
    raise ValueError('Nonfinite values are not allowed.')


class ParameterDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        tree = self.parent()
        item = tree.itemFromIndex(index)
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        if index.column() == 2 or (index.column() == 1 and kind in ('object', 'array')):
            return None
        if index.column() == 0 and (item.parent() is None or item.parent().data(0, Qt.ItemDataRole.UserRole) == 'array'):
            return None
        editor = QLineEdit(parent)
        editor.setObjectName('parameterCellEditor')
        editor.setStyleSheet('QLineEdit { background: white; color: #17293d; border: 2px solid #235f91; padding: 1px 4px; selection-background-color: #235f91; selection-color: white; }')
        return editor

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), option.fontMetrics.height() + 14))
        return size

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect.adjusted(1, 1, -1, -1))


class StructureDialog(QDialog):
    def __init__(self, document, load_presets, parent=None):
        super().__init__(parent)
        self.setWindowTitle('All Parameters and Options')
        self.resize(850, 650)
        self.result_document = None
        self.load_presets = load_presets
        layout = QVBoxLayout(self)
        hint = QLabel('Click a scalar value or press F2 to edit it. The blue-bordered input is the active editor. Core validation runs when you apply changes.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tree = QTreeWidget()
        self.tree.setItemDelegate(ParameterDelegate(self.tree))
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked | QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.tree.setHeaderLabels(['Parameter / index', 'Value', 'Type'])
        self.tree.setColumnWidth(0, 310)
        self.tree.setColumnWidth(1, 310)
        layout.addWidget(self.tree)
        self.populate(self.tree.invisibleRootItem(), 'profile', document)
        self.tree.expandToDepth(1)
        row = QHBoxLayout()
        for label, callback in (('Add Value / Entry', self.add_entry), ('Remove Selected', self.remove_entry), ('Add / Replace Option Preset', self.add_preset)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def populate(self, parent, key, value):
        kind = 'object' if isinstance(value, dict) else 'array' if isinstance(value, list) else 'string' if isinstance(value, str) else 'scalar'
        item = QTreeWidgetItem(parent, [str(key), '' if kind in ('object', 'array') else str(value) if kind == 'string' else json.dumps(value), kind])
        item.setData(0, Qt.ItemDataRole.UserRole, kind)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        if isinstance(value, (dict, list)):
            for name, child in (value.items() if isinstance(value, dict) else enumerate(value)):
                self.populate(item, name, child)
        return item

    def read(self, item):
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        children = [item.child(i) for i in range(item.childCount())]
        if kind == 'object':
            keys = [child.text(0) for child in children]
            if len(set(keys)) != len(keys) or any(not key for key in keys):
                raise ValueError('Object names must be nonempty and unique.')
            return {child.text(0): self.read(child) for child in children}
        if kind == 'array':
            return [self.read(child) for child in children]
        if kind == 'string':
            return item.text(1)
        return json.loads(item.text(1), parse_constant=reject_constant)

    def add_entry(self):
        parent = self.tree.currentItem()
        if parent is None or parent.data(0, Qt.ItemDataRole.UserRole) not in ('object', 'array'):
            return
        key = str(parent.childCount())
        if parent.data(0, Qt.ItemDataRole.UserRole) == 'object':
            key, ok = QInputDialog.getText(self, 'Parameter Name', 'JSON field name:')
            if not ok or not key:
                return
        kind, ok = QInputDialog.getItem(self, 'Value Type', 'Type:', ['number', 'string', 'boolean', 'null', 'object', 'array'], 0, False)
        if ok:
            value = {'number': 0, 'string': '', 'boolean': False, 'null': None, 'object': {}, 'array': []}[kind]
            self.tree.setCurrentItem(self.populate(parent, key, value))
            parent.setExpanded(True)

    def remove_entry(self):
        item = self.tree.currentItem()
        if item is not None and item.parent() is not None:
            item.parent().removeChild(item)

    def add_preset(self):
        presets = self.load_presets()
        if not presets:
            QMessageBox.information(self, 'Option presets', 'No readable option presets were found.')
            return
        name, ok = QInputDialog.getItem(self, 'Option preset', 'Section to replace:', sorted(presets), 0, False)
        if not ok:
            return
        source, ok = QInputDialog.getItem(self, 'Option preset', 'Source example:', sorted(presets[name]), 0, False)
        if not ok:
            return
        value = presets[name][source]
        root = self.tree.topLevelItem(0)
        for i in reversed(range(root.childCount())):
            if root.child(i).text(0) == name:
                root.removeChild(root.child(i))
        self.populate(root, name, copy.deepcopy(value))
        root.setExpanded(True)

    def apply(self):
        try:
            document = self.read(self.tree.topLevelItem(0))
            load_snapshot(json.dumps(document, allow_nan=False).encode('utf-8'))
        except (ValueError, TypeError, PhysicsConfigurationError) as exc:
            QMessageBox.warning(self, 'Invalid Value', str(exc))
            return
        self.result_document = document
        self.accept()
