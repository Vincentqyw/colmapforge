"""
Shared UI building blocks: layout helpers and the ScrollBlocker event filter.

These are pure factory functions and a tiny QObject subclass — no MainWindow
dependency. Used by section widgets and MainWindow itself.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QEvent, QObject
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QSpinBox, QWidget,
)

from .constants import _S, _W


# ── Scroll blocker ────────────────────────────────────────────────────


class ScrollBlocker(QObject):
    """Blocks mouse wheel events on spinboxes/comboboxes in a scroll area."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(obj, event)


# ── Layout Helpers ────────────────────────────────────────────────────


def _label(text: str) -> QLabel:
    """Right-aligned form label with fixed width."""
    lbl = QLabel(text)
    lbl.setObjectName("formLabel")
    lbl.setFixedWidth(_S.LABEL_W)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


def _combo(*items, data: list | None = None, min_w: int = _W.COMBO) -> QComboBox:
    """ComboBox with anti-jump sizing policy."""
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(20)
    c.setMinimumWidth(min_w)
    for i, item in enumerate(items):
        if data and i < len(data):
            c.addItem(item, data[i])
        else:
            c.addItem(item)
    return c


def _spin(min_v=0, max_v=999999, val=0, step=1) -> QSpinBox:
    s = QSpinBox()
    s.setRange(min_v, max_v); s.setValue(val); s.setSingleStep(step)
    s.setMinimumWidth(_W.SPIN_MIN)
    return s


def _dspin(min_v=0.0, max_v=1e6, val=0.0, step=0.1, dec=2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(min_v, max_v); s.setValue(val)
    s.setSingleStep(step); s.setDecimals(dec)
    s.setMinimumWidth(_W.SPIN_MIN)
    return s


def _grid_row(grid: QGridLayout, row: int, label_text: str, field: QWidget, aux: QWidget | None = None):
    """Add a label+field[+aux] row to a QGridLayout at the given row."""
    grid.addWidget(_label(label_text), row, 0)
    if aux is not None:
        grid.addWidget(field, row, 1)
        grid.addWidget(aux, row, 2)
    else:
        grid.addWidget(field, row, 1, 1, 2)


def _section_card() -> tuple[QWidget, QGridLayout]:
    """Create a section card (rounded bg) with a 3-col QGridLayout inside."""
    card = QWidget(); card.setObjectName("sectionCard")
    grid = QGridLayout(card)
    grid.setContentsMargins(*_S.CARD_PAD)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(_S.ROW_GAP)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(2, 0)
    return card, grid


def _section_header(title: str, checkable: bool = False) -> QCheckBox | QLabel:
    """Create section header. If checkable, returns QCheckBox#sectionCheck; else QLabel#sectionHeader."""
    if checkable:
        cb = QCheckBox(title); cb.setObjectName("sectionCheck"); cb.setChecked(True)
        return cb
    lbl = QLabel(title); lbl.setObjectName("sectionHeader"); return lbl
