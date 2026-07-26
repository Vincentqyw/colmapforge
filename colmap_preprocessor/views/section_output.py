"""
Output & Build section widget.

Browses for output directory, shows progress, and triggers the pipeline.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .icons import _icon_check, _icon_folder
from .widgets import _label, _section_card, _section_header


class OutputSection(QWidget):
    """Output directory picker + Build button + progress bar + result."""

    browse_requested = pyqtSignal()
    run_clicked = pyqtSignal()
    open_output_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(_section_header("Output"))

        card, grid = _section_card(); row = 0

        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self.browse_requested.emit)
        grid.addWidget(_label("Folder"), row, 0)
        grid.addWidget(btn_browse, row, 1, 1, 2); row += 1

        self._path_label = QLabel("(not set)"); self._path_label.setObjectName("hintLabel")
        self._path_label.setWordWrap(True)
        grid.addWidget(self._path_label, row, 1, 1, 2); row += 1

        self.progress = QProgressBar(); self.progress.setVisible(False)
        grid.addWidget(self.progress, row, 0, 1, 3); row += 1

        self.btn_run = QPushButton("Build COLMAP Database"); self.btn_run.setObjectName("btnRun")
        self.btn_run.setIconSize(QSize(16, 16))
        self.btn_run.clicked.connect(self.run_clicked.emit)
        grid.addWidget(self.btn_run, row, 0, 1, 3); row += 1

        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setObjectName("btnSecondary")
        self.btn_open_output.setIcon(_icon_folder())
        self.btn_open_output.setIconSize(QSize(14, 14))
        self.btn_open_output.setEnabled(False)
        self.btn_open_output.clicked.connect(self.open_output_requested.emit)
        grid.addWidget(self.btn_open_output, row, 0, 1, 3); row += 1

        self.lbl_result = QLabel(""); self.lbl_result.setWordWrap(True)
        self.lbl_result.setObjectName("resultLabel"); self.lbl_result.setVisible(False)
        grid.addWidget(self.lbl_result, row, 0, 1, 3)

        ly.addWidget(card)

    # ── public API ──

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def set_output_dir(self, path: str) -> None:
        self._output_dir = path
        self._path_label.setText(path)
        import os
        self.btn_open_output.setEnabled(os.path.isdir(path))

    def set_progress(self, pct: int, msg: str = "") -> None:
        self.progress.setValue(pct)

    def show_result(self, db_path: str, images_dir: str) -> None:
        self.progress.setValue(100); self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_result.setText(f"Database ready\n{db_path}\nOpen this DB + {images_dir} in COLMAP")
        self.lbl_result.setVisible(True)
        self.btn_run.setIcon(_icon_check())

    def set_busy(self, busy: bool) -> None:
        self.btn_run.setEnabled(not busy)
        if busy:
            self.progress.setVisible(True); self.progress.setValue(0)
            self.btn_run.setIcon(_icon_check())  # clear any previous result icon
            self.lbl_result.setVisible(False)

    def set_open_enabled(self, enabled: bool) -> None:
        self.btn_open_output.setEnabled(enabled)

    def reset(self) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_result.setVisible(False)
        self.btn_run.setIcon(_icon_check())  # clear checkmark
