"""
Resize section widget.

Controls output image dimensions via max-dimension or downscale-factor modes.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .widgets import _grid_row, _label, _section_card, _section_header


class ResizeSection(QWidget):
    """Resize config: mode (max dim / downscale), controls, result preview."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image_w = 1920; self._image_h = 1080
        self._has_dims = False
        self._ds_factor = 4

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_resize = _section_header("Resize", checkable=True)
        ly.addWidget(self.chk_resize)

        card, grid = _section_card(); row = 0

        self.cmb_resize_mode = QComboBox()
        self.cmb_resize_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_resize_mode.setMinimumContentsLength(20)
        self.cmb_resize_mode.addItem("Max dimension", "max_dim")
        self.cmb_resize_mode.addItem("Downscale factor", "downscale")
        _grid_row(grid, row, "Mode", self.cmb_resize_mode); row += 1

        self.lbl_resize_param = _label("Max")
        grid.addWidget(self.lbl_resize_param, row, 0)

        self._resize_stack = QStackedWidget()
        self.spin_max_dim = QSpinBox(); self.spin_max_dim.setRange(64, 16384)
        self.spin_max_dim.setValue(2000); self.spin_max_dim.setSingleStep(100)
        self.spin_max_dim.setSuffix(" px")
        self.spin_max_dim.valueChanged.connect(self._update_resize_result)
        self._resize_stack.addWidget(self.spin_max_dim)

        p1 = QWidget(); p1l = QHBoxLayout(p1); p1l.setContentsMargins(0, 0, 0, 0); p1l.setSpacing(4)
        self._ds_btns = []
        for factor in [1, 2, 4, 8]:
            btn = QPushButton(f"{factor}x"); btn.setCheckable(True); btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, f=factor: self._on_ds_clicked(f))
            p1l.addWidget(btn); self._ds_btns.append(btn)
        self._ds_btns[2].setChecked(True); p1l.addStretch()
        self._resize_stack.addWidget(p1)

        # Stack + result hint in same VBox (0 spacing) → no grid row gap
        self.lbl_resize_result = QLabel(""); self.lbl_resize_result.setObjectName("hintLabel")
        self.lbl_resize_result.hide()
        resize_col = QWidget()
        rl = QVBoxLayout(resize_col); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(self._resize_stack)
        rl.addWidget(self.lbl_resize_result)
        grid.addWidget(resize_col, row, 1, 1, 2); row += 1

        self.cmb_resize_mode.currentIndexChanged.connect(self._on_resize_mode_changed)
        self.cmb_resize_mode.setCurrentIndex(1)

        ly.addWidget(card)

    # ── properties ──

    @property
    def is_enabled(self) -> bool:
        return self.chk_resize.isChecked()

    @property
    def mode(self) -> str:
        return self.cmb_resize_mode.currentData()

    @property
    def max_dim(self) -> int:
        return self.spin_max_dim.value()

    @property
    def ds_factor(self) -> int:
        return self._ds_factor

    @property
    def config(self):
        """Return current resize config as a hashable tuple, or None if disabled."""
        if not self.chk_resize.isChecked():
            return None
        mode = self.cmb_resize_mode.currentData()
        if mode == "max_dim":
            return ("max_dim", self.spin_max_dim.value())
        elif mode == "downscale":
            return ("downscale", self._ds_factor)
        return (mode,)

    # ── public API ──

    def set_image_dims(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0: return
        self._image_w, self._image_h = w, h
        self._has_dims = True
        self.spin_max_dim.setValue(max(w, h))
        self._update_resize_result()

    # ── internals ──

    def _on_resize_mode_changed(self, i: int) -> None:
        self._resize_stack.setCurrentIndex(i)
        self.lbl_resize_param.setText("Max" if i == 0 else "Scale")
        self._update_resize_result()
        self.config_changed.emit()

    def _on_ds_clicked(self, factor: int) -> None:
        self._ds_factor = factor
        for b in self._ds_btns: b.setChecked(b.text() == f"{factor}x")
        self._update_resize_result()
        self.config_changed.emit()

    def _update_resize_result(self, _=None) -> None:
        if not self._has_dims:
            self.lbl_resize_result.hide()
            return
        self.lbl_resize_result.show()
        w, h = self._image_w, self._image_h
        mode = self.cmb_resize_mode.currentData()
        if mode == "max_dim":
            scale = self.spin_max_dim.value() / max(w, h)
            self.lbl_resize_result.setText(f"{w} x {h} → {int(w * scale)} x {int(h * scale)}")
        else:
            rw, rh = w // self._ds_factor, h // self._ds_factor
            self.lbl_resize_result.setText(f"{w} x {h} → {rw} x {rh}  (÷{self._ds_factor})")
