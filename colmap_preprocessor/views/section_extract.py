"""
Frame Extraction section widget.

Controls video frame sampling: method (interval/fps), max frames, and output format.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from .widgets import _grid_row, _label, _section_card, _section_header, _combo, _spin, _dspin


class ExtractSection(QWidget):
    """Frame extraction settings for video input."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_extract = _section_header("Frame Extraction", checkable=True)
        self.chk_extract.toggled.connect(lambda _: self.config_changed.emit())
        ly.addWidget(self.chk_extract)

        card, grid = _section_card(); row = 0

        self.cmb_extract_method = _combo("Every N frames", "Target FPS",
                                          data=["interval", "target_fps"])
        _grid_row(grid, row, "Sampling", self.cmb_extract_method); row += 1

        self.lbl_extract_param = _label("Interval")
        grid.addWidget(self.lbl_extract_param, row, 0)

        self._extract_stack = QStackedWidget()
        self.spin_interval = _spin(1, 10000, 60)
        self._extract_stack.addWidget(self.spin_interval)
        self.spin_target_fps = _dspin(0.1, 120, 2.0, 0.5, 1)
        self._extract_stack.addWidget(self.spin_target_fps)
        grid.addWidget(self._extract_stack, row, 1, 1, 2); row += 1

        self.spin_max_frames = _spin(0, 1000000, 0); self.spin_max_frames.setSpecialValueText("∞")
        _grid_row(grid, row, "Max frames", self.spin_max_frames); row += 1

        self.cmb_format = _combo("JPEG 95%", "JPEG 85%", "PNG", data=[".jpg", ".jpg", ".png"])
        _grid_row(grid, row, "Format", self.cmb_format); row += 1

        self.lbl_extract_info = QLabel(""); self.lbl_extract_info.setObjectName("hintLabel")
        grid.addWidget(self.lbl_extract_info, row, 0, 1, 3)

        self.cmb_extract_method.currentIndexChanged.connect(self._on_extract_method_changed)

        ly.addWidget(card)

    # ── properties ──

    @property
    def is_enabled(self) -> bool:
        return self.chk_extract.isChecked()

    @property
    def method(self) -> str:
        return self.cmb_extract_method.currentData()

    @property
    def interval(self) -> int:
        return self.spin_interval.value()

    @property
    def target_fps(self) -> float:
        return self.spin_target_fps.value()

    @property
    def max_frames(self) -> int | None:
        v = self.spin_max_frames.value()
        return v if v > 0 else None

    @property
    def output_format(self) -> str:
        return self.cmb_format.currentData()

    @property
    def jpg_quality(self) -> int:
        return 95 if "95" in self.cmb_format.currentText() else 85

    # ── public API ──

    def set_preview_text(self, text: str) -> None:
        self.lbl_extract_info.setText(text)

    # ── internals ──

    def _on_extract_method_changed(self) -> None:
        idx = self.cmb_extract_method.currentIndex()
        self._extract_stack.setCurrentIndex(idx)
        self.lbl_extract_param.setText("Interval" if idx == 0 else "FPS")
        self.config_changed.emit()
