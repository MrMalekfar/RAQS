"""Task-oriented Qt layout and shared visual language; no simulation imports."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QStackedWidget, QListWidget, QSplitter, QTableWidget, QHeaderView,
    QScrollArea, QProgressBar, QTabWidget, QCheckBox, QSizePolicy,
)
from PySide6.QtGui import QShortcut, QKeySequence
from .editor import ProfileEditor

STYLE = '''
QMainWindow, QWidget { background: #f3f5f8; color: #17293d; }
QLabel#pageTitle { font-size: 12pt; font-weight: 600; }
QLabel#muted { color: #4e6074; }
QLabel#notice { background: #e8eef5; padding: 9px; border-radius: 4px; }
QLabel#notice[severity="error"] { background: #fae8e5; color: #902a23; }
QLabel#eligibility { color: #304d67; padding: 4px 0; }
QPushButton { background: #ffffff; border: 1px solid #bdc9d5; border-radius: 4px; padding: 7px 12px; }
QPushButton:hover { background: #e8eff6; border-color: #738ba3; }
QPushButton:pressed { background: #dce7f3; }
QPushButton:focus, QLineEdit:focus, QComboBox:focus { border: 2px solid #2969a0; }
QPushButton#primary { background: #235f91; color: white; border-color: #235f91; font-weight: 600; }
QPushButton#primary:hover { background: #194c78; }
QPushButton:disabled, QPushButton#primary:disabled { background: #e6ebf0; color: #667586; border-color: #cbd4de; }
QLineEdit, QComboBox, QPlainTextEdit, QTableWidget, QTreeWidget { background: white; selection-background-color: #235f91; selection-color: white; border: 1px solid #bdc9d5; border-radius: 3px; padding: 5px; }
QLineEdit[invalid="true"] { border: 2px solid #ad3434; background: #fff4f3; }
QLabel#fieldError { color: #a12828; }
QGroupBox { border: 1px solid #d2dbe5; border-radius: 4px; margin-top: 12px; padding: 15px 10px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QListWidget { border: none; background: #e6ecf3; padding: 8px; }
QListWidget::item { padding: 13px 9px; border-radius: 4px; }
QListWidget::item:selected { background: #235f91; color: white; }
QListWidget::item:hover:!selected { background: #d9e3ee; }
QListWidget#configurationSections { background: #edf1f6; padding: 3px; }
QListWidget#configurationSections::item { padding: 6px 6px; }
QLabel#figureFeedback[severity="error"] { color: #902a23; }
QTabWidget::pane { border: 1px solid #cbd5df; }
QTabBar::tab { padding: 8px 14px; }
QProgressBar { border: 1px solid #c4ced9; border-radius: 3px; background: #e6ebf0; max-height: 8px; }
QProgressBar::chunk { background: #235f91; }
QSplitter::handle { background: #dce3eb; }
'''


def build_workbench(w, root, text_view):
    w.setStyleSheet(STYLE)
    central = QWidget()
    w.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(10, 8, 10, 6)
    layout.setSpacing(6)
    row = QHBoxLayout()
    row.addStretch()
    w.diagnostic_button = QPushButton('Run &Diagnostics')
    w.diagnostic_button.clicked.connect(lambda: w.start_run(True))
    w.sensing_button = QPushButton('Start Calculation'); w.sensing_button.setObjectName('primary')
    w.sensing_button.setShortcut('Ctrl+Return')
    w.sensing_button.clicked.connect(lambda: w.start_run(False))
    w.stop_button = QPushButton('Stop Calculation'); w.stop_button.setShortcut('Ctrl+Shift+X')
    w.stop_button.clicked.connect(w.stop_run)
    row.addWidget(w.diagnostic_button); row.addWidget(w.sensing_button); row.addWidget(w.stop_button)
    layout.addLayout(row)
    w.eligibility = QLabel(); w.eligibility.setWordWrap(True); w.eligibility.setObjectName('eligibility')
    w.notice = QLabel('Create or import a profile to begin.'); w.notice.setObjectName('notice')
    w.notice.setWordWrap(True); w.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    split = QSplitter()
    w.navigation = QListWidget(); w.navigation.setAccessibleName('Workspace navigation')
    w.navigation.addItems(['Configure', 'Inspect physics', 'Run monitor', 'Results'])
    w.navigation.setMinimumWidth(125); w.navigation.setMaximumWidth(160)
    w.tabs = QStackedWidget()
    w.navigation.currentRowChanged.connect(w.tabs.setCurrentIndex)
    w.tabs.currentChanged.connect(w.navigation.setCurrentRow)
    split.addWidget(w.navigation); split.addWidget(w.tabs); split.setStretchFactor(1, 1)
    split.setSizes([140, 1150])
    layout.addWidget(split, 1)
    w.bottom_status = QWidget(); bottom = QVBoxLayout(w.bottom_status)
    bottom.setContentsMargins(0, 0, 0, 0); bottom.setSpacing(2)
    bottom.addWidget(w.eligibility); bottom.addWidget(w.notice)
    layout.addWidget(w.bottom_status)
    w.page_headings = []

    def page(title, description):
        widget = QWidget(); column = QVBoxLayout(widget)
        column.setContentsMargins(6, 2, 0, 0); column.setSpacing(6)
        heading = QLabel(title); heading.setObjectName('pageTitle'); column.addWidget(heading)
        heading.setToolTip(description); w.page_headings.append(heading)
        w.tabs.addWidget(widget)
        return column

    column = page('Configure the experiment', 'Choose a starting point, set the physical inputs, then inspect or run the current draft.')
    row = QHBoxLayout()
    w.template_choice = QComboBox(); w.template_choice.setAccessibleName('New profile template')
    for label, file in [('AoA · near-resonant reference', 'lab006/experimental_near_resonant.json'),
                        ('Atomic diagnostics · prescribed', 'lab006/prescribed_reference.json'),
                        ('Thermal spectrum · prescribed', 'lab007/prescribed_full.json'),
                        ('Thermal spectrum · experimental', 'lab007/experimental_full.json'),
                        ('IF-only · Cs133 reference', 'atom001/cs133_prescribed.json')]:
        w.template_choice.addItem(label, file)
    w.new_button = QPushButton('&New Profile'); w.new_button.clicked.connect(w.new_profile)
    w.open_button = QPushButton('&Import JSON…'); w.open_button.setShortcut('Ctrl+O'); w.open_button.clicked.connect(w.open_profile)
    w.copy_button = QPushButton('Save &As…'); w.copy_button.setShortcut('Ctrl+Shift+S'); w.copy_button.clicked.connect(w.save_copy)
    w.reload_button = QPushButton('Reload'); w.reload_button.clicked.connect(w.reload_profile)
    row.addWidget(w.template_choice, 1)
    for control in (w.new_button, w.open_button, w.copy_button, w.reload_button): row.addWidget(control)
    column.addLayout(row)
    w.profile_label = QLabel(); w.profile_label.setTextFormat(Qt.TextFormat.PlainText); w.profile_label.setWordWrap(True); w.profile_label.setObjectName('muted'); column.addWidget(w.profile_label)
    row = QHBoxLayout()
    w.add_option_button = QPushButton('Add / replace option…'); w.add_option_button.clicked.connect(w.add_option)
    w.remove_option_button = QPushButton('Remove option…'); w.remove_option_button.clicked.connect(w.remove_option)
    w.structure_button = QPushButton('Advanced structure…'); w.structure_button.clicked.connect(w.edit_structure)
    w.related_button = QPushButton('Choose referenced profile…'); w.related_button.clicked.connect(w.open_related_profile)
    for control in (w.add_option_button, w.remove_option_button, w.structure_button, w.related_button): row.addWidget(control)
    w.preview_toggle = QCheckBox('JSON preview'); row.addWidget(w.preview_toggle)
    row.addStretch(); column.addLayout(row)
    config = QSplitter(); w.editor = ProfileEditor(); w.editor.changed.connect(w.form_changed)
    config.addWidget(w.editor)
    preview = QWidget(); pv = QVBoxLayout(preview); pv.setContentsMargins(8, 0, 0, 0)
    pv.addWidget(QLabel('LIVE JSON PREVIEW · current draft'))
    w.profile_text = text_view()
    w.profile_text.setAccessibleName('Live JSON preview')
    w.profile_text.setLineWrapMode(w.profile_text.LineWrapMode.NoWrap)
    pv.addWidget(w.profile_text)
    config.addWidget(preview); config.setSizes([680, 350]); config.setStretchFactor(0, 2); config.setStretchFactor(1, 1)
    preview.hide()
    w.preview_toggle.toggled.connect(preview.setVisible)
    column.addWidget(config, 1)
    column = page('Inspect the declared physics', 'Diagnostics resolves the atomic inputs. It is separate from a sensing calculation and does not establish model validity.')
    w.diagnostics_summary = QLabel('No diagnostics for the current draft. Run Diagnostics to resolve its declared physics.'); w.diagnostics_summary.setWordWrap(True)
    column.addWidget(w.diagnostics_summary)
    w.diagnostics_text = text_view(); w.diagnostics_text.setAccessibleName('Diagnostics report'); column.addWidget(w.diagnostics_text, 1)

    column = page('Run monitor', 'Each job uses a fresh Python process and a fixed input snapshot. Closing is blocked while the solver is active.')
    row = QHBoxLayout(); row.addWidget(QLabel('Output parent'))
    w.output = QLineEdit(str(root / 'outputs' / 'gui')); w.output.setAccessibleName('Output parent folder')
    w.browse_button = QPushButton('Browse…'); w.browse_button.clicked.connect(w.choose_output)
    row.addWidget(w.output, 1); row.addWidget(w.browse_button); column.addLayout(row)
    w.stage = QLabel('Idle — no simulation is running'); w.stage.setWordWrap(True); column.addWidget(w.stage)
    w.elapsed_label = QLabel('Elapsed: —'); column.addWidget(w.elapsed_label)
    w.progress = QProgressBar(); w.progress.setRange(0, 1); w.progress.setTextVisible(False); column.addWidget(w.progress)
    w.log = text_view(); w.log.setAccessibleName('Worker log'); w.log.setPlaceholderText('The worker log will appear here. The complete log is preserved in the run folder.'); column.addWidget(w.log, 1)
    w.follow_log = QCheckBox('Follow latest output'); w.follow_log.setChecked(True); column.addWidget(w.follow_log)
    log_scroll = w.log.verticalScrollBar()
    def follow_latest():
        if w.follow_log.isChecked():
            log_scroll.setValue(log_scroll.maximum())

    w.follow_log.toggled.connect(follow_latest)
    log_scroll.rangeChanged.connect(follow_latest)

    column = page('Review the measurement', 'Inspect the scientific figures, then the measurement summary and full technical report.')
    w.result_actions = QWidget(); row = QHBoxLayout(w.result_actions); row.setContentsMargins(0, 0, 0, 0)
    w.open_result_button = QPushButton('Open Saved Run…'); w.open_result_button.clicked.connect(w.open_result)
    w.folder_button = QPushButton('Open Result Folder'); w.folder_button.clicked.connect(w.open_folder)
    row.addWidget(w.open_result_button); row.addWidget(w.folder_button); row.addStretch(); column.addWidget(w.result_actions)
    w.result_label = QLabel('No completed measurement loaded. Run a calculation or open a verified saved run.'); w.result_label.setTextFormat(Qt.TextFormat.PlainText); w.result_label.setWordWrap(True); column.addWidget(w.result_label)
    w.result_tabs = QTabWidget()
    w.summary_table = QTableWidget(0, 2); w.summary_table.setHorizontalHeaderLabels(['Measurement / provenance', 'Value'])
    w.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    w.summary_table.horizontalHeader().setStretchLastSection(True)
    w.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); w.summary_table.verticalHeader().hide()
    w.result_tabs.addTab(w.summary_table, 'Summary')
    figures = QWidget(); fc = QVBoxLayout(figures); row = QHBoxLayout()
    w.figure_choice = QComboBox(); w.figure_choice.setAccessibleName('Result figure'); w.figure_choice.currentIndexChanged.connect(w.show_image)
    row.addWidget(w.figure_choice, 1)
    w.figure_zoom = QComboBox(); w.figure_zoom.setAccessibleName('Figure zoom'); w.figure_zoom.addItems(['Fit page', 'Fit width', '100%', '150%', '200%']); w.figure_zoom.currentIndexChanged.connect(lambda _: w.show_image(w.figure_choice.currentIndex()))
    row.addWidget(w.figure_zoom)
    w.export_button = QPushButton('Save Figure…'); w.export_button.clicked.connect(w.save_figure); row.addWidget(w.export_button)
    w.figure_focus = QPushButton('Focus figure'); w.figure_focus.setCheckable(True); w.figure_focus.toggled.connect(w.set_figure_focus); row.addWidget(w.figure_focus)
    w.escape_figure = QShortcut(QKeySequence('Esc'), w)
    w.escape_figure.activated.connect(lambda: w.figure_focus.setChecked(False))
    fc.addLayout(row)
    w.figure_feedback = QLabel()
    w.figure_feedback.setObjectName('figureFeedback')
    w.figure_feedback.setTextFormat(Qt.TextFormat.PlainText)
    w.figure_feedback.setWordWrap(True)
    w.figure_feedback.hide()
    fc.addWidget(w.figure_feedback)
    w.figure_scroll = QScrollArea(); w.figure_scroll.setWidgetResizable(True)
    w.image = QLabel('No figure available for this result.'); w.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    w.figure_scroll.setWidget(w.image); fc.addWidget(w.figure_scroll, 1)
    w.figures_page = figures
    w.result_tabs.insertTab(0, figures, 'Figures')
    w.report_text = text_view(); w.report_text.setAccessibleName('Technical run report'); w.result_tabs.addTab(w.report_text, 'Technical report')
    w.result_tabs.currentChanged.connect(lambda _: QTimer.singleShot(0, lambda: w.show_image(w.figure_choice.currentIndex())))
    column.addWidget(w.result_tabs, 1)
    w.navigation.setCurrentRow(0)
