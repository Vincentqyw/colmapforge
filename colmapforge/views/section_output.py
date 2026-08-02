"""
Output & Build section widget.

Browses for output directory, shows progress, and triggers the pipeline.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .icons import _icon_check, _icon_colmap, _icon_folder, _icon_stop
from .widgets import _label, _section_card, _section_header


class OutputSection(QWidget):
    """Output directory picker + Build/Stop button + progress bar + result."""

    browse_requested = pyqtSignal()
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    open_output_requested = pyqtSignal()
    launch_colmap_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._is_running = False
        self._db_path = ""; self._images_dir = ""
        self.chk_launch_colmap = _section_header("Launch COLMAP GUI after build", checkable=True)
        self.chk_launch_colmap.setChecked(False)

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(self.chk_launch_colmap)
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

        # ── Build / Stop (full width, primary action) ──
        self.btn_run = QPushButton("Build COLMAP Database"); self.btn_run.setObjectName("btnRun")
        self.btn_run.setIconSize(QSize(16, 16))
        self.btn_run.clicked.connect(self._on_btn_run_clicked)
        grid.addWidget(self.btn_run, row, 0, 1, 3); row += 1

        # ── Secondary actions side by side (Launch COLMAP | Open Output) ──
        actions = QWidget(); al = QHBoxLayout(actions)
        al.setContentsMargins(0, 0, 0, 0); al.setSpacing(6)

        self.btn_launch_colmap = QPushButton("Launch COLMAP GUI")
        self.btn_launch_colmap.setObjectName("btnSecondary")
        self.btn_launch_colmap.setIcon(_icon_colmap())
        self.btn_launch_colmap.setIconSize(QSize(14, 14))
        self.btn_launch_colmap.setEnabled(False)
        self.btn_launch_colmap.setToolTip("Open the built database + images in COLMAP")
        self.btn_launch_colmap.clicked.connect(self.launch_colmap_clicked.emit)
        al.addWidget(self.btn_launch_colmap, 1)

        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setObjectName("btnSecondary")
        self.btn_open_output.setIcon(_icon_folder())
        self.btn_open_output.setIconSize(QSize(14, 14))
        self.btn_open_output.setEnabled(False)
        self.btn_open_output.clicked.connect(self.open_output_requested.emit)
        al.addWidget(self.btn_open_output, 1)

        grid.addWidget(actions, row, 0, 1, 3); row += 1

        self.lbl_result = QLabel(""); self.lbl_result.setWordWrap(True)
        self.lbl_result.setObjectName("resultLabel"); self.lbl_result.setVisible(False)
        grid.addWidget(self.lbl_result, row, 0, 1, 3)

        ly.addWidget(card)

    # ── internals ──

    def _on_btn_run_clicked(self) -> None:
        """Toggle between Build and Stop."""
        if self._is_running:
            self.stop_clicked.emit()
        else:
            self.run_clicked.emit()

    def _set_run_button(self, running: bool, icon: QIcon | None = None) -> None:
        """Style the Build/Stop toggle button; re-polish so QSS follows."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Stop" if running else "Build COLMAP Database")
        self.btn_run.setObjectName("btnStop" if running else "btnRun")
        self.btn_run.setIcon(icon if icon is not None else QIcon())
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

    # ── public API ──

    def set_output_dir(self, path: str) -> None:
        self._path_label.setText(path)
        import os
        self.btn_open_output.setEnabled(os.path.isdir(path))

    def set_progress(self, pct: int, msg: str = "") -> None:
        self.progress.setValue(pct)
        if msg:
            # Show the status message directly on the progress bar so users
            # always know what the app is doing — especially during long
            # model downloads where a bare percentage looks like a freeze.
            self.progress.setFormat(f"%p%  —  {msg}")
        else:
            self.progress.setFormat("%p%")

    def show_result(self, db_path: str, images_dir: str) -> None:
        self._is_running = False
        self._db_path = db_path; self._images_dir = images_dir
        self.progress.setValue(100); self.progress.setVisible(False)
        self.progress.setFormat("%p%")  # reset format
        self._set_run_button(running=False, icon=_icon_check())
        self.btn_launch_colmap.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        self.lbl_result.setText(f"Database ready\n{db_path}")
        self.lbl_result.setVisible(True)

    def set_busy(self, busy: bool) -> None:
        self._is_running = busy
        if busy:
            self.progress.setVisible(True); self.progress.setValue(0)
            # keep the button enabled so the user can click Stop
            self._set_run_button(running=True, icon=_icon_stop(14))
            self.btn_launch_colmap.setEnabled(False)
            self.lbl_result.setVisible(False)
        else:
            self._set_run_button(running=False)

    @property
    def launch_colmap(self) -> bool:
        return self.chk_launch_colmap.isChecked()

    def reset(self) -> None:
        self._is_running = False
        self.progress.setVisible(False)
        self.progress.setFormat("%p%")  # reset format
        self._set_run_button(running=False)
        self.btn_launch_colmap.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.lbl_result.setVisible(False)
