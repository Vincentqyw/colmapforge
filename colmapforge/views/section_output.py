"""
Output & Build section widget.

Browses for output directory, shows progress, and triggers the pipeline.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPixmap, QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .icons import _icon_check, _icon_folder
from .widgets import _label, _section_card, _section_header


def _icon_stop(size: int = 16) -> QIcon:
    """Filled square icon for the stop button."""
    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    if dpr < 1.0: dpr = 1.0
    phys = max(1, int(size * dpr))
    pm = QPixmap(phys, phys); pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#ff3b30")))
    m = size * 0.22
    p.drawRoundedRect(int(m), int(m), int(size - 2 * m), int(size - 2 * m), 2, 2)
    p.end()
    return QIcon(pm)


from PyQt6.QtGui import QBrush


class OutputSection(QWidget):
    """Output directory picker + Build/Stop button + progress bar + result."""

    browse_requested = pyqtSignal()
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    open_output_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._is_running = False
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

        # ── Build / Stop button ──
        self.btn_run = QPushButton("Build COLMAP Database"); self.btn_run.setObjectName("btnRun")
        self.btn_run.setIconSize(QSize(16, 16))
        self.btn_run.clicked.connect(self._on_btn_run_clicked)
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

    # ── internals ──

    def _on_btn_run_clicked(self) -> None:
        """Toggle between Build and Stop."""
        if self._is_running:
            self.stop_clicked.emit()
        else:
            self.run_clicked.emit()

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
        if msg:
            # Show the status message directly on the progress bar so users
            # always know what the app is doing — especially during long
            # model downloads where a bare percentage looks like a freeze.
            self.progress.setFormat(f"%p%  —  {msg}")
        else:
            self.progress.setFormat("%p%")

    def show_result(self, db_path: str, images_dir: str) -> None:
        self._is_running = False
        self.progress.setValue(100); self.progress.setVisible(False)
        self.progress.setFormat("%p%")  # reset format
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Build COLMAP Database")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.setIcon(_icon_check())
        # Re-polish so QSS picks up the new objectName
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)
        self.lbl_result.setText(f"Database ready\n{db_path}\nOpen this DB + {images_dir} in COLMAP")
        self.lbl_result.setVisible(True)

    def set_busy(self, busy: bool) -> None:
        self._is_running = busy
        if busy:
            self.progress.setVisible(True); self.progress.setValue(0)
            self.btn_run.setEnabled(True)          # keep enabled so user can click Stop
            self.btn_run.setText("Stop")
            self.btn_run.setObjectName("btnStop")
            self.btn_run.setIcon(_icon_stop(14))
            self.btn_run.style().unpolish(self.btn_run)
            self.btn_run.style().polish(self.btn_run)
            self.lbl_result.setVisible(False)
        else:
            self._is_running = False
            self.btn_run.setEnabled(True)
            self.btn_run.setText("Build COLMAP Database")
            self.btn_run.setObjectName("btnRun")
            self.btn_run.setIcon(QIcon())          # clear icon
            self.btn_run.style().unpolish(self.btn_run)
            self.btn_run.style().polish(self.btn_run)

    @property
    def launch_colmap(self) -> bool:
        return self.chk_launch_colmap.isChecked()

    def set_open_enabled(self, enabled: bool) -> None:
        self.btn_open_output.setEnabled(enabled)

    def reset(self) -> None:
        self._is_running = False
        self.progress.setVisible(False)
        self.progress.setFormat("%p%")  # reset format
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Build COLMAP Database")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.setIcon(QIcon())
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)
        self.lbl_result.setVisible(False)
