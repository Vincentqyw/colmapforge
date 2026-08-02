"""
Programmatic icon generation for COLMAP Forge.

Real pixel icons painted at runtime. devicePixelRatio-aware for crisp HiDPI.
Zero font dependency → cross-platform identical rendering.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import (
    QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygon,
)
from PyQt6.QtWidgets import QApplication


def _make_icon(size: int, draw_fn, color: str = "#8e8e93") -> QIcon:
    """Create QIcon by painting onto a transparent QPixmap (HiDPI-aware)."""
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 1.0
    if dpr < 1.0 or not dpr:
        dpr = 1.0
    phys = max(1, int(size * dpr))
    pm = QPixmap(phys, phys)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    draw_fn(p, size, QColor(color))
    p.end()
    return QIcon(pm)


def _icon_moon() -> QIcon:
    """Crescent moon — dark mode toggle."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        r = s * 0.36
        cx, cy = s * 0.5, s * 0.5
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        p.setBrush(QBrush(QColor(0, 0, 0, 255)))
        r2 = r * 0.85
        cx2, cy2 = s * 0.62, s * 0.42
        p.drawEllipse(QRectF(cx2 - r2, cy2 - r2, r2 * 2, r2 * 2))
    return _make_icon(20, draw, "#8e8e93")


def _icon_sun() -> QIcon:
    """Sun with rays — light mode toggle."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy, r = s * 0.5, s * 0.5, s * 0.22
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        for a in range(0, 360, 45):
            rad = math.radians(a)
            x1 = cx + r * 1.5 * math.cos(rad); y1 = cy + r * 1.5 * math.sin(rad)
            x2 = cx + r * 2.1 * math.cos(rad); y2 = cy + r * 2.1 * math.sin(rad)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
    return _make_icon(20, draw, "#8e8e93")


def _icon_left(color: str = "#ffffff") -> QIcon:
    """Left-pointing chevron."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        m = s * 0.30
        p.drawLine(QPointF(s - m, m), QPointF(m, s * 0.5))
        p.drawLine(QPointF(m, s * 0.5), QPointF(s - m, s - m))
    return _make_icon(16, draw, color)


def _icon_right(color: str = "#ffffff") -> QIcon:
    """Right-pointing chevron."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        m = s * 0.30
        p.drawLine(QPointF(m, m), QPointF(s - m, s * 0.5))
        p.drawLine(QPointF(s - m, s * 0.5), QPointF(m, s - m))
    return _make_icon(16, draw, color)


def _icon_video(color: str = "#8e8e93") -> QIcon:
    """Play triangle for video items."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setBrush(QBrush(c)); p.setPen(Qt.PenStyle.NoPen)
        m = s * 0.20
        pts = [QPoint(int(m), int(m)), QPoint(int(m), int(s - m)), QPoint(int(s - m), int(s * 0.5))]
        p.drawPolygon(QPolygon(pts))
    return _make_icon(14, draw, color)


def _icon_folder(color: str = "#8e8e93") -> QIcon:
    """Folder icon for image items."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setBrush(QBrush(c)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(1, s * 0.32, s - 2, s * 0.58), 1.5, 1.5)
        p.drawRect(QRectF(1, s * 0.22, s * 0.45, s * 0.12))
    return _make_icon(14, draw, color)


def _icon_check(color: str = "#30d158") -> QIcon:
    """Checkmark for success."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        m = s * 0.20
        p.drawLine(QPointF(m, s * 0.55), QPointF(s * 0.40, s - m))
        p.drawLine(QPointF(s * 0.40, s - m), QPointF(s - m, m))
    return _make_icon(16, draw, color)


def _icon_info(color: str = "#8e8e93") -> QIcon:
    """Info glyph — 'i' inside a circle (About button)."""
    def draw(p: QPainter, s: int, c: QColor):
        cx = cy = s * 0.5
        r = s * 0.36
        p.setPen(QPen(c, 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawEllipse(QPointF(cx, cy - r * 0.48), s * 0.055, s * 0.055)
        pen = QPen(c, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy - r * 0.18), QPointF(cx, cy + r * 0.50))
    return _make_icon(20, draw, color)


def _icon_stop(size: int = 14) -> QIcon:
    """Filled rounded square — stop button."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setBrush(QBrush(c))
        m = s * 0.22
        p.drawRoundedRect(int(m), int(m), int(s - 2 * m), int(s - 2 * m), 2, 2)
    return _make_icon(size, draw, "#ff3b30")


def _icon_colmap(size: int = 14) -> QIcon:
    """2×2 grid glyph for the COLMAP launch button."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setBrush(QBrush(c))
        cell = s / 4.0
        for i in range(2):
            for j in range(2):
                p.drawRoundedRect(int(cell * (1 + 2 * i)), int(cell * (1 + 2 * j)),
                                  int(cell), int(cell), 1, 1)
    return _make_icon(size, draw, "#5ac8fa")


def _icon_clear(size: int = 14) -> QIcon:
    """× glyph for clear buttons."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        m = s * 0.25
        p.drawLine(int(m), int(m), int(s - m), int(s - m))
        p.drawLine(int(s - m), int(m), int(m), int(s - m))
    return _make_icon(size, draw, "#ff3b30")


def _icon_checkbox_unchecked(color: str = "#8e8e93") -> QIcon:
    """Empty rounded-rect box — unchecked state."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(QPen(c, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(1.5, 1.5, s - 3.0, s - 3.0), 4.0, 4.0)
    return _make_icon(16, draw, color)


def _icon_checkbox_checked(color: str = "#30d158") -> QIcon:
    """Filled rounded-rect box + white checkmark — checked state."""
    def draw(p: QPainter, s: int, c: QColor):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawRoundedRect(1, 1, s - 2, s - 2, 4, 4)
        # white checkmark
        p.setPen(QPen(QColor("#ffffff"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        m = s * 0.22
        p.drawLine(QPointF(m, s * 0.55), QPointF(s * 0.40, s - m - 1))
        p.drawLine(QPointF(s * 0.40, s - m - 1), QPointF(s - m, m + 1))
    return _make_icon(16, draw, color)
