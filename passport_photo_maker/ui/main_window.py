"""Modern PySide6 desktop interface for AI Passport Photo Maker."""

import logging
import os
import sys
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QProgressBar, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFrame, QMessageBox,
    QTextEdit, QScrollArea, QSizePolicy, QToolButton,
)

from ..core.input_handler import InputHandler, InputScanResult
from ..core.processing_engine import (
    ProcessingEngine, ProcessingStatus, BatchProgress, BatchReport, ImageResult
)
from ..utils.config import PassportConfig
from .. import __app_name__, __version__

logger = logging.getLogger(__name__)


# ─── Modern Stylesheet (Slate / Indigo / Emerald Palette) ───────────────────

STYLESHEET = """
QMainWindow, QWidget#centralWidget, QScrollArea, QWidget#scrollContent {
    background-color: #F1F5F9;
    color: #0F172A;
}
QMessageBox, QDialog {
    background-color: #FFFFFF;
    color: #0F172A;
}
QMessageBox QLabel, QDialog QLabel {
    color: #0F172A;
    background-color: transparent;
}
QMessageBox QPushButton, QDialog QPushButton {
    min-width: 80px;
}
QLabel {
    background-color: transparent;
    color: #0F172A;
}
QFrame#headerBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
}
QLabel#appTitle {
    font-size: 19px;
    font-weight: 700;
    color: #0F172A;
}
QLabel#appSubtitle {
    font-size: 12px;
    color: #64748B;
}
QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}
QLabel#sectionTitle {
    font-size: 12px;
    font-weight: 700;
    color: #475569;
    letter-spacing: 0.5px;
}
QLabel#mutedLabel {
    font-size: 12px;
    color: #64748B;
}
QLabel#pathBox {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 8px 10px;
    color: #334155;
    font-size: 12px;
}
QLabel#badgeInfo {
    background-color: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}
QLabel#badgeSuccess {
    background-color: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 7px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #1D4ED8;
}
QPushButton:pressed {
    background-color: #1E40AF;
}
QPushButton:disabled {
    background-color: #E2E8F0;
    color: #94A3B8;
}
QPushButton#btnSecondary {
    background-color: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
}
QPushButton#btnSecondary:hover {
    background-color: #E2E8F0;
    color: #0F172A;
}
QPushButton#btnSecondary:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
}
QPushButton#btnHeader {
    background-color: #F8FAFC;
    color: #334155;
    border: 1px solid #E2E8F0;
    padding: 6px 12px;
    font-size: 12px;
    min-height: 16px;
}
QPushButton#btnHeader:hover {
    background-color: #E2E8F0;
}
QPushButton#btnStart {
    background-color: #10B981;
    color: #FFFFFF;
    font-size: 14px;
    padding: 10px 22px;
    min-height: 22px;
}
QPushButton#btnStart:hover {
    background-color: #059669;
}
QPushButton#btnStart:disabled {
    background-color: #E2E8F0;
    color: #94A3B8;
}
QPushButton#btnPause {
    background-color: #F59E0B;
    color: #FFFFFF;
    padding: 10px 18px;
}
QPushButton#btnPause:hover {
    background-color: #D97706;
}
QPushButton#btnPause:disabled {
    background-color: #E2E8F0;
    color: #94A3B8;
}
QPushButton#btnCancel {
    background-color: #EF4444;
    color: #FFFFFF;
    padding: 10px 18px;
}
QPushButton#btnCancel:hover {
    background-color: #DC2626;
}
QPushButton#btnCancel:disabled {
    background-color: #E2E8F0;
    color: #94A3B8;
}
QToolButton {
    background-color: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}
QToolButton:hover {
    background-color: #E2E8F0;
}
QLineEdit {
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 10px;
    background-color: #F8FAFC;
    color: #0F172A;
    font-size: 12px;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
    background-color: #FFFFFF;
    color: #0F172A;
    font-size: 12px;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #F1F5F9;
    color: #94A3B8;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #0F172A;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
}
QProgressBar {
    border: none;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    font-size: 12px;
    min-height: 22px;
    background-color: #E2E8F0;
    color: #0F172A;
}
QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 8px;
}
QFrame#statCard {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
QTextEdit#logArea {
    font-family: 'Menlo', 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    background-color: #0F172A;
    color: #E2E8F0;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 8px;
}
QStatusBar {
    background-color: #FFFFFF;
    color: #475569;
    border-top: 1px solid #E2E8F0;
    font-size: 12px;
}
"""


# ─── Worker Thread ───────────────────────────────────────────────────────────

class ProcessingWorker(QThread):
    """Background worker thread for batch processing."""

    progress_updated = Signal(object)      # BatchProgress
    image_completed = Signal(object)       # ImageResult
    batch_completed = Signal(object)       # BatchReport
    engine_initialized = Signal()
    init_error = Signal(str)

    def __init__(
        self,
        engine: ProcessingEngine,
        scan_result: InputScanResult,
        output_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.scan_result = scan_result
        self.output_path = output_path
        self._init_only = False

    def set_init_only(self, value: bool):
        self._init_only = value

    def run(self):
        try:
            if self.engine._face_detector is None:
                self.engine.initialize()
            self.engine_initialized.emit()

            if self._init_only:
                return

            self.engine.set_callbacks(
                on_progress=lambda p: self.progress_updated.emit(p),
                on_image_complete=lambda r: self.image_completed.emit(r),
                on_batch_complete=lambda r: self.batch_completed.emit(r),
            )

            self.engine.process_batch(self.scan_result, self.output_path)

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            self.init_error.emit(str(e))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Create a modern card frame with a section header."""
    card = QFrame()
    card.setObjectName("card")
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)

    lbl = QLabel(title)
    lbl.setObjectName("sectionTitle")
    layout.addWidget(lbl)

    return card, layout


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.config = PassportConfig()
        self.engine = ProcessingEngine(self.config)
        self.input_handler = InputHandler()
        self.scan_result: Optional[InputScanResult] = None
        self.worker: Optional[ProcessingWorker] = None
        self._was_cancelled = False

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._update_dimension_preview()

        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(880, 740)
        self.resize(960, 840)

    def _setup_ui(self):
        """Build the modern card-based UI layout."""
        root = QWidget()
        root.setObjectName("centralWidget")
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Top Header Bar ──
        header_bar = QFrame()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 12, 20, 12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_lbl = QLabel(f"🛂  {__app_name__}")
        title_lbl.setObjectName("appTitle")
        sub_lbl = QLabel("Local AI Face Detection · Landmark Cropping · Pure White Background")
        sub_lbl.setObjectName("appSubtitle")
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        header_layout.addLayout(title_col)
        header_layout.addStretch()

        self.btn_reset_header = QPushButton("Reset Settings")
        self.btn_reset_header.setObjectName("btnHeader")
        self.btn_help_header = QPushButton("Help")
        self.btn_help_header.setObjectName("btnHeader")
        header_layout.addWidget(self.btn_reset_header)
        header_layout.addWidget(self.btn_help_header)

        root_layout.addWidget(header_bar)

        # ── Scrollable Body ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget()
        body.setObjectName("scrollContent")
        main_layout = QVBoxLayout(body)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # ── 1. Input Card ──
        input_card, input_layout = _make_card("1. INPUT PHOTOGRAPHS")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_folder = QPushButton("📂  Select Folder")
        self.btn_zip = QPushButton("🗜️  Select ZIP Archive")
        self.btn_zip.setObjectName("btnSecondary")
        btn_row.addWidget(self.btn_folder)
        btn_row.addWidget(self.btn_zip)
        btn_row.addStretch()

        self.lbl_total_photos = QLabel("0 photos detected")
        self.lbl_total_photos.setObjectName("badgeInfo")
        btn_row.addWidget(self.lbl_total_photos)

        self.lbl_input_path = QLabel("No folder or ZIP archive selected yet.")
        self.lbl_input_path.setObjectName("pathBox")
        self.lbl_input_path.setWordWrap(True)

        input_layout.addLayout(btn_row)
        input_layout.addWidget(self.lbl_input_path)
        main_layout.addWidget(input_card)

        # ── 2. Processing Settings Card ──
        settings_card, settings_layout = _make_card("2. PASSPORT SETTINGS")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        # Row 0: Passport Size & Custom dimensions & DPI
        grid.addWidget(QLabel("Passport Size:"), 0, 0)
        self.combo_size = QComboBox()
        self.combo_size.addItems([
            "45 × 55 mm (Standard)",
            "35 × 45 mm (EU / India)",
            "51 × 51 mm (US 2×2 in)",
            "Custom (mm)",
        ])
        grid.addWidget(self.combo_size, 0, 1)

        grid.addWidget(QLabel("Resolution:"), 0, 2)
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(150, 600)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSuffix(" DPI")
        grid.addWidget(self.spin_dpi, 0, 3)

        # Row 1: Custom width/height + pixel preview
        grid.addWidget(QLabel("Dimensions:"), 1, 0)
        dim_row = QHBoxLayout()
        dim_row.setSpacing(6)
        self.spin_width_mm = QDoubleSpinBox()
        self.spin_width_mm.setRange(20.0, 150.0)
        self.spin_width_mm.setValue(45.0)
        self.spin_width_mm.setDecimals(1)
        self.spin_width_mm.setSuffix(" mm")
        self.spin_width_mm.setEnabled(False)

        x_lbl = QLabel("×")
        x_lbl.setAlignment(Qt.AlignCenter)

        self.spin_height_mm = QDoubleSpinBox()
        self.spin_height_mm.setRange(20.0, 150.0)
        self.spin_height_mm.setValue(55.0)
        self.spin_height_mm.setDecimals(1)
        self.spin_height_mm.setSuffix(" mm")
        self.spin_height_mm.setEnabled(False)

        self.lbl_pixel_size = QLabel("531 × 650 px")
        self.lbl_pixel_size.setObjectName("mutedLabel")

        dim_row.addWidget(self.spin_width_mm)
        dim_row.addWidget(x_lbl)
        dim_row.addWidget(self.spin_height_mm)
        dim_row.addWidget(self.lbl_pixel_size)
        grid.addLayout(dim_row, 1, 1)

        grid.addWidget(QLabel("Background:"), 1, 2)
        bg_label = QLabel("Pure White #FFFFFF (Locked)")
        bg_label.setObjectName("badgeSuccess")
        grid.addWidget(bg_label, 1, 3)

        # Row 2: Output location
        grid.addWidget(QLabel("Output Folder:"), 2, 0)
        out_layout = QHBoxLayout()
        out_layout.setSpacing(6)
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Defaults to input folder")
        self.txt_output.setReadOnly(True)
        self.btn_browse_out = QToolButton()
        self.btn_browse_out.setText("Browse...")
        out_layout.addWidget(self.txt_output)
        out_layout.addWidget(self.btn_browse_out)
        grid.addLayout(out_layout, 2, 1, 1, 3)

        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(3, 1)
        settings_layout.addLayout(grid)
        main_layout.addWidget(settings_card)

        # ── 3. Processing & Live Progress Card ──
        proc_card, proc_layout = _make_card("3. BATCH PROCESSING")

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.btn_start = QPushButton("▶  Start Processing")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setEnabled(False)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btnPause")
        self.btn_pause.setEnabled(False)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setEnabled(False)

        controls_row.addWidget(self.btn_start, 2)
        controls_row.addWidget(self.btn_pause, 1)
        controls_row.addWidget(self.btn_cancel, 1)
        proc_layout.addLayout(controls_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%  —  Idle")
        proc_layout.addWidget(self.progress_bar)

        self.lbl_current_file = QLabel("Ready to process.")
        self.lbl_current_file.setObjectName("mutedLabel")
        proc_layout.addWidget(self.lbl_current_file)

        # Statistics Row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self.stat_widgets = {}
        for key, title, color in [
            ("total", "TOTAL", "#1D4ED8"),
            ("processed", "PROCESSED", "#7C3AED"),
            ("cropped", "CROPPED", "#059669"),
            ("rejected", "REJECTED", "#DC2626"),
            ("remaining", "REMAINING", "#D97706"),
        ]:
            frame = QFrame()
            frame.setObjectName("statCard")
            frame.setMinimumHeight(70)
            vl = QVBoxLayout(frame)
            vl.setContentsMargins(10, 8, 10, 8)
            vl.setSpacing(2)
            vl.setAlignment(Qt.AlignCenter)

            num_label = QLabel("0")
            num_label.setMinimumHeight(28)
            num_label.setStyleSheet(
                f"font-size: 22px; font-weight: 700; color: {color}; background: transparent;"
            )
            num_label.setAlignment(Qt.AlignCenter)

            title_label = QLabel(title)
            title_label.setStyleSheet(
                "font-size: 10px; font-weight: 700; color: #64748B; "
                "letter-spacing: 0.6px; background: transparent;"
            )
            title_label.setAlignment(Qt.AlignCenter)

            vl.addWidget(num_label)
            vl.addWidget(title_label)
            stats_row.addWidget(frame)
            self.stat_widgets[key] = num_label

        proc_layout.addLayout(stats_row)
        main_layout.addWidget(proc_card)

        # ── 4. Output & Activity Log Card ──
        out_card, out_card_layout = _make_card("4. OUTPUT & ACTIVITY LOG")

        output_btn_row = QHBoxLayout()
        output_btn_row.setSpacing(10)

        self.btn_open_cropped = QPushButton("Open Cropped")
        self.btn_open_cropped.setEnabled(False)
        self.btn_open_rejected = QPushButton("Open Rejected")
        self.btn_open_rejected.setObjectName("btnSecondary")
        self.btn_open_rejected.setEnabled(False)
        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setObjectName("btnSecondary")
        self.btn_open_output.setEnabled(False)
        self.btn_csv_report = QPushButton("Export CSV Report")
        self.btn_csv_report.setObjectName("btnSecondary")
        self.btn_csv_report.setEnabled(False)

        output_btn_row.addWidget(self.btn_open_cropped)
        output_btn_row.addWidget(self.btn_open_rejected)
        output_btn_row.addWidget(self.btn_open_output)
        output_btn_row.addWidget(self.btn_csv_report)
        out_card_layout.addLayout(output_btn_row)

        self.log_area = QTextEdit()
        self.log_area.setObjectName("logArea")
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(130)
        self.log_area.setPlaceholderText("Processing log will appear here...")
        out_card_layout.addWidget(self.log_area)

        main_layout.addWidget(out_card)

        scroll.setWidget(body)
        root_layout.addWidget(scroll)

        self.statusBar().showMessage("Ready")

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Select Folder...", self._select_folder)
        file_menu.addAction("Select ZIP...", self._select_zip)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction("Reset to Defaults", self._reset_settings)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("How to Use", self._show_help)
        help_menu.addAction("About", self._show_about)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_folder.clicked.connect(self._select_folder)
        self.btn_zip.clicked.connect(self._select_zip)
        self.btn_browse_out.clicked.connect(self._select_output_folder)
        self.btn_reset_header.clicked.connect(self._reset_settings)
        self.btn_help_header.clicked.connect(self._show_help)

        self.btn_start.clicked.connect(self._start_processing)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel.clicked.connect(self._cancel_processing)

        self.btn_open_cropped.clicked.connect(lambda: self._open_folder("cropped"))
        self.btn_open_rejected.clicked.connect(lambda: self._open_folder("rejected"))
        self.btn_open_output.clicked.connect(lambda: self._open_folder("output"))
        self.btn_csv_report.clicked.connect(self._export_csv)

        self.combo_size.currentIndexChanged.connect(self._on_size_changed)
        self.spin_dpi.valueChanged.connect(self._on_dpi_changed)
        self.spin_width_mm.valueChanged.connect(self._on_custom_dim_changed)
        self.spin_height_mm.valueChanged.connect(self._on_custom_dim_changed)

    # ─── Input Selection ─────────────────────────────────────────────────

    @Slot()
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Photographs"
        )
        if folder:
            self._scan_input("folder", folder)

    @Slot()
    def _select_zip(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ZIP Archive", "", "ZIP Archives (*.zip)"
        )
        if path:
            self._scan_input("zip", path)

    def _scan_input(self, source_type: str, path: str):
        """Scan selected input for photographs."""
        self.log("Scanning input...")
        self.statusBar().showMessage("Scanning...")

        if source_type == "folder":
            self.scan_result = self.input_handler.scan_folder(path)
        else:
            self.scan_result = self.input_handler.scan_zip(path)

        total = self.scan_result.total_found
        self.lbl_input_path.setText(path)
        self.lbl_total_photos.setText(f"{total} photos detected")
        self.stat_widgets["total"].setText(str(total))
        self.stat_widgets["remaining"].setText(str(total))
        self.stat_widgets["processed"].setText("0")
        self.stat_widgets["cropped"].setText("0")
        self.stat_widgets["rejected"].setText("0")

        if self.scan_result.errors:
            self.log(f"⚠️ Errors: {'; '.join(self.scan_result.errors)}")
        if self.scan_result.duplicates:
            self.log(f"ℹ️ {len(self.scan_result.duplicates)} duplicate filenames detected")
        if self.scan_result.unsupported:
            self.log(f"ℹ️ {len(self.scan_result.unsupported)} unsupported files skipped")

        self.log(f"✅ Found {total} photographs ready for processing")
        self.btn_start.setEnabled(total > 0)

        if source_type == "folder":
            self.txt_output.setText(path)
        else:
            self.txt_output.setText(os.path.dirname(path))

        self.statusBar().showMessage(f"Found {total} photographs")

    # ─── Settings ────────────────────────────────────────────────────────

    def _select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Location")
        if folder:
            self.txt_output.setText(folder)

    @Slot(int)
    def _on_size_changed(self, index: int):
        presets = [
            (45.0, 55.0),
            (35.0, 45.0),
            (51.0, 51.0),
            None,
        ]
        is_custom = index == 3
        self.spin_width_mm.setEnabled(is_custom)
        self.spin_height_mm.setEnabled(is_custom)

        if index < len(presets) and presets[index]:
            w, h = presets[index]
            self.spin_width_mm.blockSignals(True)
            self.spin_height_mm.blockSignals(True)
            self.spin_width_mm.setValue(w)
            self.spin_height_mm.setValue(h)
            self.spin_width_mm.blockSignals(False)
            self.spin_height_mm.blockSignals(False)
            self.config.width_mm = w
            self.config.height_mm = h
            self._update_dimension_preview()

    @Slot()
    def _on_custom_dim_changed(self):
        self.config.width_mm = self.spin_width_mm.value()
        self.config.height_mm = self.spin_height_mm.value()
        self._update_dimension_preview()

    @Slot(int)
    def _on_dpi_changed(self, value: int):
        self.config.dpi = value
        self._update_dimension_preview()

    def _update_dimension_preview(self):
        self.lbl_pixel_size.setText(
            f"{self.config.width_px} × {self.config.height_px} px"
        )

    def _reset_settings(self):
        self.config = PassportConfig()
        self.engine = ProcessingEngine(self.config)
        self.combo_size.setCurrentIndex(0)
        self.spin_dpi.setValue(300)
        self._update_dimension_preview()
        self.log("Settings reset to defaults (45 × 55 mm, 300 DPI)")

    # ─── Processing ──────────────────────────────────────────────────────

    @Slot()
    def _start_processing(self):
        if not self.scan_result or self.scan_result.total_found == 0:
            QMessageBox.warning(self, "No Input", "Please select photographs first.")
            return

        output_path = self.txt_output.text() or os.path.dirname(
            self.scan_result.source_path
        )

        total = self.scan_result.total_found
        self._was_cancelled = False
        self.stat_widgets["total"].setText(str(total))
        self.stat_widgets["processed"].setText("0")
        self.stat_widgets["cropped"].setText("0")
        self.stat_widgets["rejected"].setText("0")
        self.stat_widgets["remaining"].setText(str(total))

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%  —  Initializing...")
        self.log_area.clear()

        self.engine.config = self.config

        self.btn_start.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_zip.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.combo_size.setEnabled(False)
        self.spin_dpi.setEnabled(False)

        self.log("🚀 Initializing AI models... (this may take a moment on first run)")
        self.statusBar().showMessage("Initializing AI models...")

        self.worker = ProcessingWorker(
            self.engine, self.scan_result, output_path, self
        )
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.image_completed.connect(self._on_image_complete)
        self.worker.batch_completed.connect(self._on_batch_complete)
        self.worker.engine_initialized.connect(self._on_engine_initialized)
        self.worker.init_error.connect(self._on_init_error)
        self.worker.start()

    @Slot()
    def _on_engine_initialized(self):
        self.log("✅ AI models loaded successfully")
        self.statusBar().showMessage("Processing...")

    @Slot(str)
    def _on_init_error(self, error: str):
        self.log(f"❌ Initialization error: {error}")
        QMessageBox.critical(
            self,
            "Initialization Error",
            f"Failed to initialize processing engine:\n\n{error}\n\n"
            "Please ensure all required packages are installed.",
        )
        self._reset_ui_after_processing()

    @Slot(object)
    def _on_progress(self, progress: BatchProgress):
        self.progress_bar.setValue(int(progress.percentage))
        self.progress_bar.setFormat(
            f"{progress.percentage:.1f}%  —  {progress.processed}/{progress.total}"
        )
        self.lbl_current_file.setText(f"Processing: {progress.current_file}")

        self.stat_widgets["processed"].setText(str(progress.processed))
        self.stat_widgets["cropped"].setText(str(progress.successful))
        self.stat_widgets["rejected"].setText(str(progress.rejected + progress.errors))
        self.stat_widgets["remaining"].setText(str(progress.remaining))

        self.statusBar().showMessage(
            f"Processing {progress.processed}/{progress.total} "
            f"({progress.elapsed_time:.0f}s elapsed)"
        )

    @Slot(object)
    def _on_image_complete(self, result: ImageResult):
        if result.success:
            self.log(f"  ✅ {result.filename} ({result.processing_time:.1f}s)")
        elif result.error:
            self.log(f"  ❌ {result.filename}: ERROR - {result.error}")
        else:
            self.log(f"  ⛔ {result.filename}: {result.rejection_reason}")

    @Slot(object)
    def _on_batch_complete(self, report: BatchReport):
        self.log("")
        self.log("=" * 52)
        self.log("📋  PROCESSING REPORT")
        self.log("=" * 52)
        self.log(f"  Total photographs:     {report.total_found}")
        self.log(f"  Successfully cropped:  {report.successfully_cropped}")
        self.log(f"  Rejected:              {report.rejected}")
        self.log(f"  Processing errors:     {report.processing_errors}")
        self.log(f"  Total time:            {report.total_time:.1f} seconds")
        self.log("=" * 52)

        self._last_report = report
        self._reset_ui_after_processing()

        self.btn_open_cropped.setEnabled(True)
        self.btn_open_rejected.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        self.btn_csv_report.setEnabled(True)

        if self._was_cancelled:
            self.progress_bar.setFormat("Cancelled")
            self.lbl_current_file.setText("Processing cancelled by user.")
            self.statusBar().showMessage(
                f"Cancelled ({report.successfully_cropped} cropped)"
            )
        else:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100%  —  Complete!")
            self.lbl_current_file.setText("Batch processing complete.")
            self.statusBar().showMessage(
                f"Done! {report.successfully_cropped}/{report.total_found} cropped "
                f"in {report.total_time:.1f}s"
            )

        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processing {'cancelled' if self._was_cancelled else 'complete'}!\n\n"
            f"✅ Cropped: {report.successfully_cropped}\n"
            f"⛔ Rejected: {report.rejected}\n"
            f"❌ Errors: {report.processing_errors}\n\n"
            f"Output: {report.output_folder}",
        )

    @Slot()
    def _toggle_pause(self):
        if self.engine.status == ProcessingStatus.RUNNING:
            self.engine.pause()
            self.btn_pause.setText("▶  Resume")
            self.statusBar().showMessage("Paused")
            self.log("⏸️ Processing paused")
        elif self.engine.status == ProcessingStatus.PAUSED:
            self.engine.resume()
            self.btn_pause.setText("⏸  Pause")
            self.statusBar().showMessage("Processing...")
            self.log("▶️ Processing resumed")

    @Slot()
    def _cancel_processing(self):
        reply = QMessageBox.question(
            self,
            "Cancel Processing",
            "Are you sure you want to cancel?\n\n"
            "Already processed photos will be kept.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._was_cancelled = True
            self.engine.cancel()
            self.log("🛑 Cancelling processing...")
            self.statusBar().showMessage("Cancelling...")

    def _reset_ui_after_processing(self):
        """Reset UI controls after processing ends."""
        self.btn_start.setEnabled(True)
        self.btn_folder.setEnabled(True)
        self.btn_zip.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_cancel.setEnabled(False)
        self.combo_size.setEnabled(True)
        self.spin_dpi.setEnabled(True)

    # ─── Output ──────────────────────────────────────────────────────────

    def _open_folder(self, folder_type: str):
        """Open output folder in file manager."""
        if not hasattr(self, "_last_report"):
            return

        report = self._last_report
        if folder_type == "cropped":
            path = report.cropped_folder
        elif folder_type == "rejected":
            path = report.rejected_folder
        else:
            path = report.output_folder

        if os.path.isdir(path):
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Folder does not exist:\n{path}")

    @Slot()
    def _export_csv(self):
        if not hasattr(self, "_last_report"):
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Report", "processing_report.csv",
            "CSV Files (*.csv)"
        )
        if path:
            self.engine.generate_csv_report(self._last_report, path)
            self.log(f"📄 CSV report saved: {path}")

    # ─── Help ────────────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(
            self,
            "About",
            f"<h3>{__app_name__}</h3>"
            f"<p>Version {__version__}</p>"
            f"<p>Professional AI-powered passport photo processor.</p>"
            f"<p>Automatically detects faces, removes backgrounds, "
            f"and crops photos to passport dimensions.</p>",
        )

    def _show_help(self):
        QMessageBox.information(
            self,
            "How to Use",
            "<h3>Quick Start</h3>"
            "<ol>"
            "<li><b>Select Input:</b> Choose a folder or ZIP archive containing photographs.</li>"
            "<li><b>Configure:</b> Adjust passport size (or Custom mm) and DPI if needed.</li>"
            "<li><b>Process:</b> Click 'Start Processing' and monitor live progress.</li>"
            "<li><b>Results:</b> Open the Cropped or Rejected folders to inspect results.</li>"
            "</ol>"
            "<h3>Tips</h3>"
            "<ul>"
            "<li>Original photos are never modified or deleted.</li>"
            "<li>Rejected photos are saved with their rejection reason.</li>"
            "<li>Export a CSV report for full batch details.</li>"
            "</ul>",
        )

    # ─── Logging ─────────────────────────────────────────────────────────

    def log(self, message: str):
        """Append a message to the log area."""
        self.log_area.append(message)
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ─── Cleanup ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Handle window close."""
        if self.engine.status == ProcessingStatus.RUNNING:
            reply = QMessageBox.question(
                self,
                "Exit",
                "Processing is still running. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.engine.cancel()

        if self.worker and self.worker.isRunning():
            self.engine.cancel()
            self.worker.wait(5000)

        self.engine.cleanup()
        self.input_handler.cleanup()
        event.accept()
