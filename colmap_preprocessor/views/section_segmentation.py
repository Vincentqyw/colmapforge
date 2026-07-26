"""
Dynamic Object Masking section widget.

SAM/SkyWater model selection, confidence threshold, class presets,
custom class input, and collapsible class checkboxes.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from .constants import PRESET_CLASSES, QUICK_PRESETS, _W
from .widgets import _combo, _dspin, _grid_row, _label, _section_card, _section_header


class SegmentationSection(QWidget):
    """Segmentation config: model, confidence, presets, class selection."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._custom_classes: list[str] = []

        ly = QVBoxLayout(self); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_seg = _section_header("Dynamic Object Masking", checkable=True)
        self.chk_seg.toggled.connect(lambda _: self.config_changed.emit())
        ly.addWidget(self.chk_seg)

        card, grid = _section_card(); row = 0

        self.cmb_sam_model = _combo(min_w=_W.COMBO)
        _grid_row(grid, row, "Model", self.cmb_sam_model); row += 1

        self.spin_conf = _dspin(0.1, 1.0, 0.3, 0.05, 2)
        _grid_row(grid, row, "Confidence", self.spin_conf); row += 1

        self.cmb_presets = _combo(*QUICK_PRESETS.keys())
        _grid_row(grid, row, "Preset", self.cmb_presets); row += 1

        self.btn_toggle_classes = QPushButton("+ Show")
        self.btn_toggle_classes.clicked.connect(self._toggle_classes)
        grid.addWidget(_label("Classes"), row, 0)
        grid.addWidget(self.btn_toggle_classes, row, 1, 1, 2); row += 1

        self._class_checks: dict[str, QCheckBox] = {}
        self.w_classes = QWidget(); cg = QHBoxLayout(self.w_classes)
        cg.setContentsMargins(0, 0, 0, 0); cg.setSpacing(4)
        col1, col2, col3 = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        col1.setSpacing(0); col2.setSpacing(0); col3.setSpacing(0)
        for i, cls_name in enumerate(PRESET_CLASSES):
            cb = QCheckBox(cls_name); cb.setChecked(False)
            self._class_checks[cls_name] = cb
            [col1, col2, col3][i % 3].addWidget(cb)
        cg.addLayout(col1); cg.addLayout(col2); cg.addLayout(col3)
        self.w_classes.hide(); grid.addWidget(self.w_classes, row, 0, 1, 3); row += 1

        # Connect preset signal AFTER _class_checks is populated (setCurrentIndex
        # triggers _apply_preset which iterates _class_checks).
        self.cmb_presets.currentIndexChanged.connect(self._apply_preset)
        self.cmb_presets.setCurrentIndex(1)

        cust_row = QWidget(); cr = QHBoxLayout(cust_row); cr.setContentsMargins(0, 0, 0, 0); cr.setSpacing(4)
        self.edit_custom_cls = QLineEdit(); self.edit_custom_cls.setPlaceholderText("Custom class…")
        self.edit_custom_cls.returnPressed.connect(self._add_custom_class)
        btn_custom = QPushButton("+"); btn_custom.setFixedWidth(28)
        btn_custom.clicked.connect(self._add_custom_class)
        cr.addWidget(self.edit_custom_cls, 1); cr.addWidget(btn_custom)
        grid.addWidget(_label("Custom"), row, 0)
        grid.addWidget(cust_row, row, 1, 1, 2); row += 1

        self.lbl_custom_cls = QLabel(""); self.lbl_custom_cls.setObjectName("customClassLabel")
        grid.addWidget(self.lbl_custom_cls, row, 1, 1, 2)

        ly.addWidget(card)

    # ── properties ──

    @property
    def is_enabled(self) -> bool:
        return self.chk_seg.isChecked()

    @property
    def model_config(self) -> dict | None:
        return self.cmb_sam_model.currentData()

    @property
    def confidence(self) -> float:
        return self.spin_conf.value()

    @property
    def target_classes(self) -> list[str]:
        checked = [k for k, cb in self._class_checks.items() if cb.isChecked()]
        return checked + self._custom_classes

    # ── public API ──

    def populate_models(self, configs: list[dict]) -> None:
        """Fill the model combo with discovered configs."""
        self.cmb_sam_model.clear()
        for cfg in configs:
            label = cfg.get("display_name", cfg.get("name", ""))
            has_text = " [TEXT]" if cfg.get("language_encoder_path") else ""
            self.cmb_sam_model.addItem(f"{label}{has_text}", cfg)

    def add_model_item(self, label: str, cfg: dict) -> None:
        self.cmb_sam_model.addItem(label, cfg)

    def apply_preset(self) -> None:
        self._apply_preset()

    # ── internals ──

    def _apply_preset(self) -> None:
        try: name = self.cmb_presets.currentText()
        except RuntimeError: return
        if name not in QUICK_PRESETS: return
        targets = set(QUICK_PRESETS[name])
        for cls_name, cb in self._class_checks.items():
            cb.setChecked(cls_name in targets)
        self.config_changed.emit()

    def _toggle_classes(self) -> None:
        if not hasattr(self, "_classes_expanded"):
            self._classes_expanded = False
        self._classes_expanded = not self._classes_expanded
        self.w_classes.setVisible(self._classes_expanded)
        self.btn_toggle_classes.setText("- Hide" if self._classes_expanded else "+ Show")

    def _add_custom_class(self) -> None:
        text = self.edit_custom_cls.text().strip().lower()
        if not text or text in self._custom_classes: return
        if text in self._class_checks: return
        self._custom_classes.append(text); self.edit_custom_cls.clear()
        self.lbl_custom_cls.setText("Custom: " + ", ".join(self._custom_classes))
        self.config_changed.emit()
