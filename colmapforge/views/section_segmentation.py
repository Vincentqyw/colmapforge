"""
Dynamic Object Masking section widget.

SAM/SkyWater model selection, confidence threshold, class presets,
custom class input, and collapsible class checkboxes.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from .constants import PRESET_CLASSES, QUICK_PRESETS, _W
from .widgets import _combo, _dspin, _grid_row, _label, _section_card, _section_header


def _icon_clear(size: int = 14) -> QIcon:
    """× icon for the clear-custom-classes button."""
    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    if dpr < 1.0: dpr = 1.0
    phys = max(1, int(size * dpr))
    pm = QPixmap(phys, phys); pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor("#ff3b30"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    m = size * 0.25
    p.drawLine(int(m), int(m), int(size - m), int(size - m))
    p.drawLine(int(size - m), int(m), int(m), int(size - m))
    p.end()
    return QIcon(pm)


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
        self.cmb_sam_model.currentIndexChanged.connect(self._on_model_changed)
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

        # ── Custom class row: input + add button + clear all button ──
        cust_row = QWidget(); cr = QHBoxLayout(cust_row)
        cr.setContentsMargins(0, 0, 0, 0); cr.setSpacing(4)
        self.edit_custom_cls = QLineEdit()
        self.edit_custom_cls.setPlaceholderText("Custom prompts…")
        self.edit_custom_cls.returnPressed.connect(self._add_custom_class)
        btn_add = QPushButton("+"); btn_add.setFixedWidth(28)
        btn_add.clicked.connect(self._add_custom_class)
        self.btn_clear_custom = QPushButton(); self.btn_clear_custom.setFixedWidth(28)
        self.btn_clear_custom.setIcon(_icon_clear(12))
        self.btn_clear_custom.setToolTip("Clear all custom prompts…")
        self.btn_clear_custom.clicked.connect(self._clear_custom_classes)
        self.btn_clear_custom.setVisible(False)
        cr.addWidget(self.edit_custom_cls, 1); cr.addWidget(btn_add)
        cr.addWidget(self.btn_clear_custom)
        grid.addWidget(_label("Custom"), row, 0)
        grid.addWidget(cust_row, row, 1, 1, 2); row += 1

        # ── Custom class display: clickable tags (click to remove) ──
        self.lbl_custom_cls = QLabel("")
        self.lbl_custom_cls.setObjectName("customClassLabel")
        self.lbl_custom_cls.setToolTip("Click a class name to remove it")
        self.lbl_custom_cls.linkActivated.connect(self._remove_custom_class)
        grid.addWidget(self.lbl_custom_cls, row, 1, 1, 2)

        # ── Hint: custom class model compatibility ──
        self.lbl_custom_hint = QLabel("")
        self.lbl_custom_hint.setObjectName("hintLabel")
        grid.addWidget(self.lbl_custom_hint, row + 1, 1, 1, 2)

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

    def _is_skywater_model(self) -> bool:
        """Check whether the currently selected model is SkyWater."""
        cfg = self.cmb_sam_model.currentData()
        if not cfg:
            return False
        model_path = cfg.get("model_path", "")
        return (
            "skywater" in cfg.get("type", "").lower() or
            "skywater" in cfg.get("name", "").lower() or
            "skywater" in model_path.lower()
        )

    def _on_model_changed(self) -> None:
        """When the model combo changes, update custom class hint."""
        is_sw = self._is_skywater_model()
        if is_sw:
            self.lbl_custom_hint.setText(
                "SkyWater: custom classes not supported (fixed: sky, water, person)")
            self.edit_custom_cls.setEnabled(False)
            self.edit_custom_cls.setPlaceholderText("Not available for SkyWater")
        else:
            self.lbl_custom_hint.setText(
                "SAM3: custom classes used as text prompts")
            self.edit_custom_cls.setEnabled(True)
            self.edit_custom_cls.setPlaceholderText("Custom prompts…")

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
        """Add a custom class from the input field."""
        # Block custom classes for SkyWater model
        if self._is_skywater_model():
            return
        text = self.edit_custom_cls.text().strip().lower()
        if not text or text in self._custom_classes:
            return
        if text in self._class_checks:
            return
        self._custom_classes.append(text)
        self.edit_custom_cls.clear()
        self._refresh_custom_display()
        self.config_changed.emit()

    def _clear_custom_classes(self) -> None:
        """Remove all custom classes."""
        if not self._custom_classes:
            return
        self._custom_classes.clear()
        self._refresh_custom_display()
        self.config_changed.emit()

    def _remove_custom_class(self, cls_name: str) -> None:
        """Remove a single custom class by name."""
        if cls_name in self._custom_classes:
            self._custom_classes.remove(cls_name)
            self._refresh_custom_display()
            self.config_changed.emit()

    def _refresh_custom_display(self) -> None:
        """Update the custom class label and clear button visibility."""
        if self._custom_classes:
            # Build clickable tags: each class name is a link that removes it
            tags = "  ".join(
                f'<a href="{c}" style="text-decoration:none;">{c} ×</a>'
                for c in self._custom_classes
            )
            self.lbl_custom_cls.setText(f"Custom: {tags}")
        else:
            self.lbl_custom_cls.setText("")
        self.btn_clear_custom.setVisible(len(self._custom_classes) > 0)
