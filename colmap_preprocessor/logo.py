"""
COLMAP Preprocessor — application logo.

Concept: isometric cube — hexagon silhouette + 3 internal edges + vertex dots.
Represents 3D structure / reconstruction — the core of COLMAP.

QPainter-drawn for cross-platform pixel-identical rendering.
  · HiDPI-aware (devicePixelRatio)
  · Theme-aware (dark/light color sets)
  · Detail scales with size: full dots ≥40px, center dot ≥20px, wireframe only <20px
"""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QPointF, QPoint
from PyQt6.QtGui import (
    QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygon,
)
from PyQt6.QtWidgets import QApplication


# ── Theme palettes ────────────────────────────────────────────────────
def _theme_colors(is_dark: bool) -> dict:
    if is_dark:
        return {
            "line":   "#8e8e93",   # hexagon outline + internal edges
            "accent": "#0a84ff",   # vertex dots
        }
    return {
        "line":   "#8e8e93",
        "accent": "#007aff",
    }


# ── Geometry: isometric cube ──────────────────────────────────────────
# A regular hexagon is the silhouette of an isometric cube. 3 internal
# edges from the center to alternating vertices (top, lower-left, lower-
# right in screen space) divide the hexagon into 3 rhombi = 3 visible
# faces. The center point is the cube's front vertex (nearest to viewer).
_HEX_ANGLES_DEG = (30, 90, 150, 210, 270, 330)
# Indices into _HEX_ANGLES_DEG of the 3 vertices the internal edges target.
# Angles 90° (screen-bottom), 210° (upper-left), 330° (upper-right) → the
# 3 edges form a "Y" dividing the silhouette into top/left/right faces.
_INTERNAL_IDX = (1, 3, 5)


def _draw_logo(p: QPainter, s: int, is_dark: bool) -> None:
    """Draw the isometric-cube logo into a square of side s."""
    c = _theme_colors(is_dark)
    cx = cy = s * 0.5
    R = s * 0.42
    line_w = max(1.0, s * 0.014)

    # Hexagon vertices (screen coords, y-down)
    hex_v = [
        (cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a)))
        for a in _HEX_ANGLES_DEG
    ]

    # ── 1. Hexagon outline (cube silhouette) ──
    pen = QPen(QColor(c["line"]), line_w)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygon([QPoint(int(x), int(y)) for x, y in hex_v]))

    # ── 2. 3 internal edges (center → alternating vertices) ──
    for idx in _INTERNAL_IDX:
        p.drawLine(QPoint(int(cx), int(cy)),
                   QPoint(int(hex_v[idx][0]), int(hex_v[idx][1])))

    # ── 3. Vertex dots (size-adaptive) ──
    # Full dots at ≥40px; center dot only at ≥20px; wireframe only below.
    if s >= 20:
        dr = max(1.4, s * 0.038)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(c["accent"])))
        if s >= 40:
            # 6 hexagon vertex dots
            for x, y in hex_v:
                p.drawEllipse(QPointF(x, y), dr, dr)
        # Center dot (front vertex) — always larger, the focal point
        p.drawEllipse(QPointF(cx, cy), dr * 1.5, dr * 1.5)


# ── Public API ────────────────────────────────────────────────────────
def make_logo_pixmap(size: int, is_dark: bool | None = None) -> QPixmap:
    """Render the logo at the given logical size (HiDPI-aware).

    is_dark: None → auto-detect from Theme singleton.
    """
    if is_dark is None:
        try:
            from .theme import Theme
            is_dark = Theme.get().is_dark
        except Exception:
            is_dark = True

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
    _draw_logo(p, size, is_dark)
    p.end()
    return pm


def make_logo_icon(size: int, is_dark: bool | None = None) -> QIcon:
    """Render the logo as a QIcon at the given logical size."""
    return QIcon(make_logo_pixmap(size, is_dark))


def make_app_icon() -> QIcon:
    """Multi-resolution window icon covering 16/32/64/128 px.

    16px → wireframe only; 32px → +center dot; 64/128 → full vertex dots.
    """
    icon = QIcon()
    for sz in (16, 32, 64, 128):
        icon.addPixmap(make_logo_pixmap(sz), QIcon.Mode.Normal, QIcon.State.Off)
    return icon
