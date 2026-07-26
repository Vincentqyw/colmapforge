"""
Input section widget.

Selects videos/images, displays the input list, and supports right-click removal.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from .widgets import _section_card, _section_header


class InputSection(QWidget):
    """Input picker: Videos / Images buttons, item list, info hint."""

    videos_requested = pyqtSignal()
    images_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    context_menu_remove = pyqtSignal(int)  # row index

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(_section_header("Input"))

        card, grid = _section_card(); row = 0

        btn_row = QWidget(); br = QHBoxLayout(btn_row); br.setContentsMargins(0, 0, 0, 0); br.setSpacing(4)
        btn_vid = QPushButton("Videos"); btn_vid.clicked.connect(self.videos_requested.emit)
        btn_img = QPushButton("Images"); btn_img.clicked.connect(self.images_requested.emit)
        self.btn_clear_input = QPushButton("Clear"); self.btn_clear_input.clicked.connect(self.clear_requested.emit)
        self.btn_clear_input.setEnabled(False)
        br.addWidget(btn_vid); br.addWidget(btn_img); br.addWidget(self.btn_clear_input); br.addStretch()
        grid.addWidget(btn_row, row, 0, 1, 3); row += 1

        self.input_list = QListWidget(); self.input_list.setAlternatingRowColors(True)
        self.input_list.setMaximumHeight(36)
        self.input_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.input_list.customContextMenuRequested.connect(self._on_context_menu)
        grid.addWidget(self.input_list, row, 0, 1, 3); row += 1

        self.lbl_input_info = QLabel("No input selected."); self.lbl_input_info.setObjectName("hintLabel")
        grid.addWidget(self.lbl_input_info, row, 0, 1, 3)

        ly.addWidget(card)

    # ── properties ──

    @property
    def has_input(self) -> bool:
        return self.input_list.count() > 0

    # ── public API ──

    def set_info(self, text: str) -> None:
        self.lbl_input_info.setText(text)

    def set_clear_enabled(self, enabled: bool) -> None:
        self.btn_clear_input.setEnabled(enabled)

    def refresh_list(self, items: list[tuple[QIcon, str]]) -> None:
        """Rebuild the input list from (icon, label) pairs."""
        self.input_list.clear()
        for icon, label in items:
            self.input_list.addItem(QListWidgetItem(icon, label))

    # ── internals ──

    def _on_context_menu(self, pos) -> None:
        item = self.input_list.itemAt(pos)
        if not item: return
        idx = self.input_list.row(item)
        self.context_menu_remove.emit(idx)
