"""
Camera Model section widget.

Selects a COLMAP camera model and exposes editable intrinsic params.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QPushButton, QVBoxLayout, QWidget,
)

from ..camera_models import FISHEYE_MODELS, PINHOLE_MODELS, get_camera_model
from .constants import _W
from .widgets import _label, _section_card, _section_header, ScrollBlocker


class CameraSection(QWidget):
    """Camera model picker with collapsible intrinsic parameter grid."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scroll_blocker = ScrollBlocker(self)
        self._param_spinboxes: list[QDoubleSpinBox] = []

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(_section_header("Camera Model"))

        card, grid = _section_card(); row = 0

        self.cmb_camera = QComboBox()
        self.cmb_camera.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_camera.setMinimumContentsLength(20)
        self.cmb_camera.setMinimumWidth(_W.COMBO)
        for model in PINHOLE_MODELS:
            self.cmb_camera.addItem(f"{model.name} ({model.num_params}p)", model.model_id)
        self.cmb_camera.insertSeparator(len(PINHOLE_MODELS))
        for model in FISHEYE_MODELS:
            self.cmb_camera.addItem(f"{model.name} ({model.num_params}p)", model.model_id)
        idx = self.cmb_camera.findData(2)
        if idx >= 0: self.cmb_camera.setCurrentIndex(idx)
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_changed)
        grid.addWidget(_label("Model"), row, 0)
        grid.addWidget(self.cmb_camera, row, 1, 1, 2); row += 1

        self.btn_toggle_cam_params = QPushButton("+ Show")
        self.btn_toggle_cam_params.clicked.connect(self._toggle_cam_params)
        grid.addWidget(_label("Params"), row, 0)
        grid.addWidget(self.btn_toggle_cam_params, row, 1, 1, 2); row += 1

        self._params_w = QWidget()
        self._params_ly = QGridLayout(self._params_w)
        self._params_ly.setContentsMargins(0, 0, 0, 0)
        self._params_ly.setHorizontalSpacing(8); self._params_ly.setVerticalSpacing(2)
        self._params_ly.setColumnStretch(1, 1)
        grid.addWidget(self._params_w, row, 0, 1, 3)
        self._params_w.hide()

        self._image_w = 1920; self._image_h = 1080
        self._on_camera_changed()

        ly.addWidget(card)

    # ── properties ──

    @property
    def camera_model_id(self) -> int | None:
        return self.cmb_camera.currentData()

    @property
    def camera_params(self) -> list[float]:
        return [sb.value() for sb in self._param_spinboxes]

    # ── public API ──

    def set_image_dims(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0: return
        self._image_w, self._image_h = w, h
        self._rebuild_params()

    # ── internals ──

    def _toggle_cam_params(self) -> None:
        if not hasattr(self, "_cam_params_expanded"):
            self._cam_params_expanded = False
        self._cam_params_expanded = not self._cam_params_expanded
        self._params_w.setVisible(self._cam_params_expanded)
        self.btn_toggle_cam_params.setText("- Hide" if self._cam_params_expanded else "+ Show")

    def _on_camera_changed(self) -> None:
        self._rebuild_params()
        self.config_changed.emit()

    def _rebuild_params(self) -> None:
        while self._params_ly.count():
            it = self._params_ly.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._param_spinboxes.clear()

        mid = self.cmb_camera.currentData()
        if mid is None: return
        model = get_camera_model(mid)
        defaults = model.default_params(self._image_w, self._image_h)
        for r, (pname, pval) in enumerate(zip(model.params, defaults)):
            sb = QDoubleSpinBox(); sb.setRange(-1e6, 1e6); sb.setDecimals(4); sb.setValue(pval)
            sb.setMinimumWidth(_W.SPIN_MIN)
            sb.setToolTip(pname); sb.installEventFilter(self._scroll_blocker)
            sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._params_ly.addWidget(_label(f"{pname}"), r, 0)
            self._params_ly.addWidget(sb, r, 1, 1, 2)
            self._param_spinboxes.append(sb)
