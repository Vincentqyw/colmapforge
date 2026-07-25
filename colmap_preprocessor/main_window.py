"""
Single-page COLMAP preprocessing tool — MVC view layer.
Pixel-aligned layout via QGridLayout + card-based sections.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QEvent, QObject, QThreadPool, QTimer, QSize, QPoint, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QKeySequence, QIcon, QPainter, QPen, QColor, QBrush, QPixmap, QPolygon, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider,
    QSpinBox, QSplitter, QStackedWidget, QStatusBar, QStyle, QVBoxLayout, QWidget,
)

from .camera_models import (
    FISHEYE_MODELS, PINHOLE_MODELS, get_camera_model,
)
from .about_dialog import AboutDialog
from .logo import make_app_icon, make_logo_pixmap
from .theme import Theme, apply_theme
from .utils import (
    VIDEO_EXTENSIONS, analyze_videos, collect_image_files,
)
from .workers import (
    DatabaseBuildWorker, FrameExtractionWorker, SegmentationWorker,
)

logger = logging.getLogger(__name__)

# ── Design Constants ──────────────────────────────────────────────────
class _S:
    """Spacing constants."""
    SECTION_GAP = 6
    ROW_GAP = 1
    CARD_PAD = (10, 3, 10, 3)   # left top right bottom — tight
    LABEL_W = 90                  # form label fixed width (fits "Confidence", "Max frames")

class _W:
    """Widget width constants."""
    COMBO = 180
    SPIN_MIN = 80
    # No SPIN_MAX — let spinbox stretch to fill grid col 1-2, same as combo box

# ── Data Constants ────────────────────────────────────────────────────
PRESET_CLASSES = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "bird", "cat", "dog", "boat", "sky", "water",
    "reflection", "shadow", "tree branch", "cloud",
]
QUICK_PRESETS = {
    "People + Vehicles": ["person", "car", "bus", "truck", "motorcycle", "bicycle"],
    "Sky + Water": ["sky", "water", "cloud", "reflection"],
    "All Dynamics": list(PRESET_CLASSES),
    "Animals": ["bird", "cat", "dog"],
}

# ── Scroll blocker ────────────────────────────────────────────────────
class ScrollBlocker(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(obj, event)


# ── Programmatic Icon Generation ──────────────────────────────────────
# Real pixel icons painted at runtime. devicePixelRatio-aware for crisp HiDPI.
# Zero font dependency → cross-platform identical rendering.

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
        # Draw a full disk in moon color, then carve out a smaller disk to make crescent.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        r = s * 0.36
        cx, cy = s * 0.5, s * 0.5
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        # Carve out crescent with transparent color (DestinationOut keeps existing alpha)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        p.setBrush(QBrush(QColor(0, 0, 0, 255)))
        r2 = r * 0.85
        cx2, cy2 = s * 0.62, s * 0.42
        p.drawEllipse(QRectF(cx2 - r2, cy2 - r2, r2 * 2, r2 * 2))
    return _make_icon(20, draw, "#8e8e93")


def _icon_sun() -> QIcon:
    """Sun with rays — light mode toggle."""
    import math
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
        # Chevron: top-right → middle-left → bottom-right
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
        # Body
        p.drawRoundedRect(QRectF(1, s * 0.32, s - 2, s * 0.58), 1.5, 1.5)
        # Tab
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
        # Circle outline
        p.setPen(QPen(c, 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        # 'i' dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawEllipse(QPointF(cx, cy - r * 0.48), s * 0.055, s * 0.055)
        # 'i' stem
        pen = QPen(c, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy - r * 0.18), QPointF(cx, cy + r * 0.50))
    return _make_icon(20, draw, color)


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


# ======================================================================
# MainWindow
# ======================================================================

class MainWindow(QMainWindow):
    """COLMAP Preprocessor — pixel-aligned, card-based, Apple-style."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("COLMAP Preprocessor")
        self.setMinimumSize(960, 880); self.resize(1200, 1060)
        self.setWindowIcon(make_app_icon())

        # ── state ──
        self._theme = Theme.get()
        self._detect_system_theme()
        self._video_paths: list[str] = []
        self._image_paths: list[str] = []
        self._video_info: list[dict] = []
        self._model_configs: list[dict] = []
        self._output_dir = ""
        self._current_image: str = ""
        self._mask_cache: dict[str, np.ndarray] = {}
        self._all_output_images: list[str] = []
        self._custom_classes: list[str] = []
        self._param_spinboxes: list[QDoubleSpinBox] = []
        self._image_w: int = 1920; self._image_h: int = 1080
        self._preview_key = None; self._preview_full: QPixmap | None = None
        self._thread_pool = QThreadPool(); self._active_worker = None
        self._phase = ""
        self._ds_factor = 4; self._ds_btns: list[QPushButton] = []
        self._class_checks: dict[str, QCheckBox] = {}
        self._last_resize_cfg = None  # track last build's resize config for re-build detection

        self._scroll_blocker = ScrollBlocker(self)
        self._setup_ui()
        self._setup_shortcuts()
        self._load_models()
        self._apply_theme()
        self._install_scroll_blocker()
        self._refresh_gpu_status()
        QTimer.singleShot(200, self._refresh_preview)

    # ── Theme ──
    def _theme_color_text(self) -> str:
        """Foreground color suited to current theme (for icon painting)."""
        return "#ffffff" if self._theme.is_dark else "#1c1c1e"

    def _theme_color_secondary(self) -> str:
        """Secondary label color (for list item icons)."""
        return "#8e8e93"

    def _detect_system_theme(self) -> None:
        try:
            import darkdetect
            self._theme.set_dark(darkdetect.isDark())
        except ImportError:
            pass

    # ── ONNX Runtime status indicator ──────────────────────────────────
    def _refresh_gpu_status(self) -> None:
        """Update the toolbar GPU/CPU indicator from onnx_utils.diagnose().

        Color codes (objectName → QSS):
          gpuStatusOk    (green)  — CUDA or DirectML active
          gpuStatusWarn  (orange) — installed but CPU-only
          gpuStatusErr   (red)    — not installed, or wheel-overwrite detected
          gpuStatusUnknown (gray) — initial state, before first refresh
        """
        from .onnx_utils import diagnose
        diag = diagnose()
        if not diag["installed"]:
            self._lbl_gpu_status.setText("  ONNX: missing")
            self._lbl_gpu_status.setObjectName("gpuStatusErr")
        elif diag["issues"]:
            # Silent-overwrite problem detected
            self._lbl_gpu_status.setText(f"  GPU: broken ({diag['version']})")
            self._lbl_gpu_status.setObjectName("gpuStatusErr")
        elif diag["gpu_active"]:
            self._lbl_gpu_status.setText(f"  GPU: {diag['active_provider']}")
            self._lbl_gpu_status.setObjectName("gpuStatusOk")
        else:
            self._lbl_gpu_status.setText("  GPU: CPU only")
            self._lbl_gpu_status.setObjectName("gpuStatusWarn")
        # Re-apply stylesheet so the new objectName takes effect immediately.
        self._lbl_gpu_status.style().unpolish(self._lbl_gpu_status)
        self._lbl_gpu_status.style().polish(self._lbl_gpu_status)
        self._onnx_diag = diag

    def _show_onnx_diagnostics(self) -> None:
        """Show a message box with detailed ONNX Runtime diagnostics + fix hints."""
        from .onnx_utils import diagnose
        diag = diagnose()

        if not diag["installed"]:
            title = "ONNX Runtime — Not Installed"
            body = (
                "ONNX Runtime is not installed. Segmentation features will not work.\n\n"
                + (diag.get("hint") or "")
            )
            QMessageBox.warning(self, title, body)
            return

        lines = [
            f"ONNX Runtime version: {diag['version']}",
            f"Installed wheels: {', '.join(diag['installed_wheels']) or '(none)'}",
            f"Available providers: {', '.join(diag['providers'])}",
            f"GPU active: {diag['gpu_active']}"
            + (f" ({diag['active_provider']})" if diag['active_provider'] else ""),
        ]
        if diag["issues"]:
            lines.append("")
            lines.append("Issues detected:")
            for i, issue in enumerate(diag["issues"], 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("")
            lines.append("No issues detected. Setup is healthy.")

        title = "ONNX Runtime Diagnostics"
        if diag["issues"]:
            QMessageBox.warning(self, title, "\n".join(lines))
        else:
            QMessageBox.information(self, title, "\n".join(lines))

    def _apply_theme(self) -> None:
        apply_theme(QApplication.instance(), self._theme)
        # Force-refresh the left panel scroll area + its viewport + content
        # widget. Qt's stylesheet engine doesn't always repaint QScrollArea's
        # viewport on setStyleSheet(), leaving the old theme's bg visible.
        if hasattr(self, "_left_scroll"):
            sa = self._left_scroll
            for w in (sa, sa.viewport(), sa.widget()):
                if w is None:
                    continue
                w.style().unpolish(w)
                w.style().polish(w)
                w.update()
        # Refresh app logo + window icon (theme-aware colors)
        self.setWindowIcon(make_app_icon())
        if hasattr(self, "_lbl_logo"):
            self._lbl_logo.setPixmap(make_logo_pixmap(24))
        # Theme toggle icon
        self.btn_theme.setIcon(_icon_moon() if self._theme.is_dark else _icon_sun())
        self.btn_theme.setText("")
        # Preview nav icons — re-paint with theme-appropriate color
        if hasattr(self, "btn_prev"):
            self.btn_prev.setIcon(_icon_left(self._theme_color_text()))
            self.btn_next.setIcon(_icon_right(self._theme_color_text()))
        # Re-paint list item icons (theme-aware secondary color)
        if hasattr(self, "input_list"):
            self._refresh_input_list()
        self.status_bar.showMessage("Ready — Ctrl+O add images · Ctrl+B build")

    def _toggle_theme(self) -> None:
        self._theme.toggle(); self._apply_theme(); self._refresh_preview()

    def _show_about(self) -> None:
        """Open the About dialog (theme-aware)."""
        dlg = AboutDialog(self, is_dark=self._theme.is_dark)
        dlg.exec()

    # ── Shortcuts ──
    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_add_images)
        QShortcut(QKeySequence("Ctrl+B"), self, self._run_pipeline)
        QShortcut(QKeySequence("Ctrl+T"), self, self._toggle_theme)
        QShortcut(QKeySequence("Right"), self, self._next_image)
        QShortcut(QKeySequence("Left"), self, self._prev_image)

    # ==================================================================
    # Top-level layout
    # ==================================================================
    def _setup_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._make_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([420, 780])
        splitter.setStretchFactor(0, 0)  # left panel: don't grow
        splitter.setStretchFactor(1, 1)  # right panel: grow
        root.addWidget(splitter, 1)

        self.status_bar = QStatusBar(); root.addWidget(self.status_bar)

    # ── Toolbar ──
    def _make_toolbar(self) -> QWidget:
        bar = QWidget(); bar.setFixedHeight(36); bar.setObjectName("toolbar")
        ly = QHBoxLayout(bar); ly.setContentsMargins(14, 4, 14, 4)

        # App logo (theme-aware, refreshed in _apply_theme)
        self._lbl_logo = QLabel()
        self._lbl_logo.setFixedSize(24, 24)
        self._lbl_logo.setPixmap(make_logo_pixmap(24))
        ly.addWidget(self._lbl_logo)
        ly.addSpacing(8)

        title = QLabel("COLMAP Preprocessor"); title.setObjectName("appTitle")
        ly.addWidget(title); ly.addStretch()

        self._lbl_img_count = QLabel("Images: 0"); self._lbl_img_count.setObjectName("statusInfo")
        ly.addWidget(self._lbl_img_count)
        self._lbl_model_count = QLabel("  Models: —"); self._lbl_model_count.setObjectName("statusInfo")
        ly.addWidget(self._lbl_model_count)

        # GPU status indicator — click to view install/fix instructions.
        # Surfaces the silent onnxruntime wheel-overwrite problem at a glance.
        self._lbl_gpu_status = QLabel("  GPU: …")
        self._lbl_gpu_status.setObjectName("gpuStatusUnknown")
        self._lbl_gpu_status.setToolTip("Click to view ONNX Runtime diagnostics")
        self._lbl_gpu_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lbl_gpu_status.mousePressEvent = lambda _e: self._show_onnx_diagnostics()
        ly.addWidget(self._lbl_gpu_status); ly.addSpacing(10)

        self.btn_theme = QPushButton(); self.btn_theme.setFixedSize(32, 28)
        self.btn_theme.setObjectName("themeBtn")
        self.btn_theme.setIconSize(QSize(20, 20))
        # Set initial icon (will be updated in _apply_theme)
        self.btn_theme.setIcon(_icon_sun())
        self.btn_theme.clicked.connect(self._toggle_theme); ly.addWidget(self.btn_theme)

        # About button — opens project info dialog
        self.btn_about = QPushButton(); self.btn_about.setFixedSize(32, 28)
        self.btn_about.setObjectName("themeBtn")
        self.btn_about.setIconSize(QSize(20, 20))
        self.btn_about.setIcon(_icon_info())
        self.btn_about.setToolTip("About COLMAP Preprocessor")
        self.btn_about.clicked.connect(self._show_about); ly.addWidget(self.btn_about)
        return bar

    # ── Left panel ──
    def _make_left_panel(self) -> QWidget:
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360); scroll.setMaximumWidth(520)
        scroll.setObjectName("leftPanel")
        # Ensure the scroll area itself doesn't draw a default background —
        # we want the QSS rule on #leftPanelContent to show through.
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._left_scroll = scroll  # kept for theme-switch viewport refresh

        w = QWidget(); w.setObjectName("leftPanelContent")
        ly = QVBoxLayout(w)
        ly.setSpacing(_S.SECTION_GAP); ly.setContentsMargins(10, 8, 10, 8)

        ly.addWidget(self._build_section_input())
        ly.addWidget(self._build_section_extract())
        ly.addWidget(self._build_section_resize())
        ly.addWidget(self._build_section_segmentation())
        ly.addWidget(self._build_section_camera())
        ly.addWidget(self._build_section_output())
        ly.addStretch()

        scroll.setWidget(w); return scroll

    # ── Right panel ──
    def _make_right_panel(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(8, 6, 8, 6); ly.setSpacing(3)

        self.preview_label = QLabel("Preview appears after Build (Ctrl+B)")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 200)
        self.preview_label.setScaledContents(False)
        ly.addWidget(self.preview_label, 1)

        # Controls bar
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(0, 2, 0, 2); cl.setSpacing(8)

        self.btn_prev = QPushButton(); self.btn_prev.setFixedWidth(32); self.btn_prev.setToolTip("Previous (Left)")
        self.btn_prev.setIconSize(QSize(16, 16))
        self.btn_prev.setIcon(_icon_left(self._theme_color_text())); self.btn_prev.clicked.connect(self._prev_image); cl.addWidget(self.btn_prev)

        self.lbl_preview_info = QLabel("No image"); self.lbl_preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_info.setObjectName("previewInfo"); cl.addWidget(self.lbl_preview_info, 1)

        self.btn_next = QPushButton(); self.btn_next.setFixedWidth(32); self.btn_next.setToolTip("Next (Right)")
        self.btn_next.setIconSize(QSize(16, 16))
        self.btn_next.setIcon(_icon_right(self._theme_color_text())); self.btn_next.clicked.connect(self._next_image); cl.addWidget(self.btn_next)

        cl.addWidget(QLabel("Mask:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100); self.slider_opacity.setValue(60)
        self.slider_opacity.setFixedWidth(80)
        self.slider_opacity.valueChanged.connect(self._refresh_preview); cl.addWidget(self.slider_opacity)

        self.chk_show_mask = QCheckBox("Show"); self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self._refresh_preview); cl.addWidget(self.chk_show_mask)
        ly.addWidget(ctrl)

        # Thumbnail strip
        self.thumb_list = QListWidget(); self.thumb_list.setMaximumHeight(48)
        self.thumb_list.setAlternatingRowColors(True)
        self.thumb_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.thumb_list.currentRowChanged.connect(self._on_thumb_selected)
        ly.addWidget(self.thumb_list)
        return w

    # ==================================================================
    # Section builders — each returns a QWidget with header + card
    # ==================================================================

    def _section_header(self, title: str, checkable: bool = False) -> QCheckBox | QLabel:
        """Create section header. If checkable, returns QCheckBox#sectionCheck; else QLabel#sectionHeader."""
        if checkable:
            cb = QCheckBox(title); cb.setObjectName("sectionCheck"); cb.setChecked(True)
            return cb
        lbl = QLabel(title); lbl.setObjectName("sectionHeader"); return lbl

    # ── 1. Input ──
    def _build_section_input(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(self._section_header("Input"))

        card, grid = _section_card(); row = 0
        # Button row
        btn_row = QWidget(); br = QHBoxLayout(btn_row); br.setContentsMargins(0, 0, 0, 0); br.setSpacing(4)
        btn_vid = QPushButton("Videos"); btn_vid.clicked.connect(self._on_add_videos)
        btn_img = QPushButton("Images"); btn_img.clicked.connect(self._on_add_images)
        self.btn_clear_input = QPushButton("Clear"); self.btn_clear_input.clicked.connect(self._on_clear_input)
        self.btn_clear_input.setEnabled(False)
        br.addWidget(btn_vid); br.addWidget(btn_img); br.addWidget(self.btn_clear_input); br.addStretch()
        grid.addWidget(btn_row, row, 0, 1, 3); row += 1

        # Input list
        self.input_list = QListWidget(); self.input_list.setAlternatingRowColors(True)
        self.input_list.setMaximumHeight(36)
        self.input_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.input_list.customContextMenuRequested.connect(self._on_input_context_menu)
        grid.addWidget(self.input_list, row, 0, 1, 3); row += 1

        self.lbl_input_info = QLabel("No input selected."); self.lbl_input_info.setObjectName("hintLabel")
        grid.addWidget(self.lbl_input_info, row, 0, 1, 3)

        ly.addWidget(card); return w

    # ── 2. Frame Extraction ──
    def _build_section_extract(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_extract = self._section_header("Frame Extraction", checkable=True)
        self.chk_extract.toggled.connect(lambda _: self._update_extract_preview())
        ly.addWidget(self.chk_extract)

        card, grid = _section_card(); row = 0

        self.cmb_extract_method = _combo("Every N frames", "Target FPS",
                                           data=["interval", "target_fps"])
        _grid_row(grid, row, "Sampling", self.cmb_extract_method); row += 1

        # Dynamic label in grid col 0 (text updates with mode) + stacked spinbox in col 1-2.
        # This keeps the field aligned with all other rows in the grid.
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
        ly.addWidget(card); return w

    def _on_extract_method_changed(self) -> None:
        idx = self.cmb_extract_method.currentIndex()
        self._extract_stack.setCurrentIndex(idx)
        self.lbl_extract_param.setText("Interval" if idx == 0 else "FPS")
        self._update_extract_preview()

    def _update_extract_preview(self) -> None:
        if not self._video_info or not self.chk_extract.isChecked():
            self.lbl_extract_info.setText(""); return
        from .utils import compute_frame_indices
        method = self.cmb_extract_method.currentData()
        lines = []
        for info in self._video_info:
            try:
                if method == "interval":
                    frames = compute_frame_indices(
                        info["total_frames"], info["fps"], method="interval",
                        interval=self.spin_interval.value(),
                        max_frames=self.spin_max_frames.value() or None)
                elif method == "target_fps":
                    frames = compute_frame_indices(
                        info["total_frames"], info["fps"], method="target_fps",
                        target_fps=self.spin_target_fps.value(),
                        max_frames=self.spin_max_frames.value() or None)
                else:
                    continue
                lines.append(f"{Path(info['path']).name}: {len(frames)} frames")
            except Exception:
                pass
        self.lbl_extract_info.setText("  ·  ".join(lines) if lines else "")

    # ── 3. Resize ──
    def _build_section_resize(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_resize = self._section_header("Resize", checkable=True)
        ly.addWidget(self.chk_resize)

        card, grid = _section_card(); row = 0

        self.cmb_resize_mode = _combo("Max dimension", "Downscale factor", data=["max_dim", "downscale"])
        _grid_row(grid, row, "Mode", self.cmb_resize_mode); row += 1

        # Dynamic label in grid col 0 + stacked content in col 1-2 (aligned with other rows)
        self.lbl_resize_param = _label("Max")
        grid.addWidget(self.lbl_resize_param, row, 0)

        self._resize_stack = QStackedWidget()
        # Page 0: spin_max_dim only
        self.spin_max_dim = _spin(64, 16384, 2000, 100); self.spin_max_dim.setSuffix(" px")
        self.spin_max_dim.valueChanged.connect(self._update_resize_result)
        self._resize_stack.addWidget(self.spin_max_dim)
        # Page 1: downscale buttons row
        p1 = QWidget(); p1l = QHBoxLayout(p1); p1l.setContentsMargins(0,0,0,0); p1l.setSpacing(4)
        self._ds_btns = []
        for factor in [1, 2, 4, 8]:
            btn = QPushButton(f"{factor}×"); btn.setCheckable(True); btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, f=factor: self._on_ds_clicked(f))
            p1l.addWidget(btn); self._ds_btns.append(btn)
        self._ds_btns[2].setChecked(True); p1l.addStretch()
        self._resize_stack.addWidget(p1)
        grid.addWidget(self._resize_stack, row, 1, 1, 2); row += 1

        self.lbl_resize_result = QLabel(""); self.lbl_resize_result.setObjectName("hintLabel")
        grid.addWidget(self.lbl_resize_result, row, 0, 1, 3)

        def _on_resize_mode_changed(i: int):
            self._resize_stack.setCurrentIndex(i)
            self.lbl_resize_param.setText("Max" if i == 0 else "Scale")
            self._update_resize_result()
        self.cmb_resize_mode.currentIndexChanged.connect(_on_resize_mode_changed)
        self.cmb_resize_mode.setCurrentIndex(1)
        ly.addWidget(card); return w

    def _on_ds_clicked(self, factor: int) -> None:
        self._ds_factor = factor
        for b in self._ds_btns: b.setChecked(b.text() == f"{factor}×")
        self._update_resize_result()

    def _update_resize_result(self, _=None) -> None:
        w, h = self._image_w, self._image_h
        mode = self.cmb_resize_mode.currentData()
        if mode == "max_dim":
            scale = self.spin_max_dim.value() / max(w, h)
            self.lbl_resize_result.setText(f"→ {int(w*scale)} × {int(h*scale)}")
        else:
            rw, rh = w // self._ds_factor, h // self._ds_factor
            self.lbl_resize_result.setText(f"→ {rw} × {rh}  (÷{self._ds_factor})")

    def _update_image_dims(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0: return
        self._image_w, self._image_h = w, h
        self.spin_max_dim.setValue(max(w, h))
        self._update_resize_result(); self._on_camera_changed()

    # ── 4. Segmentation ──
    def _build_section_segmentation(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        self.chk_seg = self._section_header("Dynamic Object Masking", checkable=True)
        ly.addWidget(self.chk_seg)

        card, grid = _section_card(); row = 0

        self.cmb_sam_model = _combo(min_w=_W.COMBO)
        _grid_row(grid, row, "Model", self.cmb_sam_model); row += 1

        self.spin_conf = _dspin(0.1, 1.0, 0.3, 0.05, 2)
        _grid_row(grid, row, "Confidence", self.spin_conf); row += 1

        self.cmb_presets = _combo(*QUICK_PRESETS.keys())
        _grid_row(grid, row, "Preset", self.cmb_presets); row += 1

        # "Classes" label in col 0 (aligned with Model/Confidence/Preset) + toggle button in col 1-2
        self.btn_toggle_classes = QPushButton("+ Show")
        self.btn_toggle_classes.clicked.connect(self._toggle_classes)
        grid.addWidget(_label("Classes"), row, 0)
        grid.addWidget(self.btn_toggle_classes, row, 1, 1, 2); row += 1

        # Class checkboxes (collapsible) — appears below the button when expanded
        self._class_checks = {}
        self.w_classes = QWidget(); cg = QHBoxLayout(self.w_classes)
        cg.setContentsMargins(0,0,0,0); cg.setSpacing(4)
        col1, col2, col3 = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        col1.setSpacing(0); col2.setSpacing(0); col3.setSpacing(0)
        for i, cls_name in enumerate(PRESET_CLASSES):
            cb = QCheckBox(cls_name); cb.setChecked(False)
            self._class_checks[cls_name] = cb
            [col1, col2, col3][i % 3].addWidget(cb)
        cg.addLayout(col1); cg.addLayout(col2); cg.addLayout(col3)
        self.w_classes.hide(); grid.addWidget(self.w_classes, row, 0, 1, 3); row += 1

        # Custom class row — label in col 0, lineedit+button in col 1-2 (aligned with other fields)
        cust_row = QWidget(); cr = QHBoxLayout(cust_row); cr.setContentsMargins(0,0,0,0); cr.setSpacing(4)
        self.edit_custom_cls = QLineEdit(); self.edit_custom_cls.setPlaceholderText("Custom class…")
        self.edit_custom_cls.returnPressed.connect(self._add_custom_class)
        btn_custom = QPushButton("+"); btn_custom.setFixedWidth(28)
        btn_custom.clicked.connect(self._add_custom_class)
        cr.addWidget(self.edit_custom_cls, 1); cr.addWidget(btn_custom)
        grid.addWidget(_label("Custom"), row, 0)
        grid.addWidget(cust_row, row, 1, 1, 2); row += 1

        self.lbl_custom_cls = QLabel(""); self.lbl_custom_cls.setObjectName("customClassLabel")
        grid.addWidget(self.lbl_custom_cls, row, 1, 1, 2)

        self.cmb_presets.currentIndexChanged.connect(self._apply_preset)
        self.cmb_presets.setCurrentIndex(1)
        ly.addWidget(card); return w

    def _apply_preset(self) -> None:
        try: name = self.cmb_presets.currentText()
        except RuntimeError: return
        if name not in QUICK_PRESETS: return
        targets = set(QUICK_PRESETS[name])
        for cls_name, cb in self._class_checks.items():
            cb.setChecked(cls_name in targets)

    def _toggle_classes(self) -> None:
        # Track state explicitly (isVisible() is unreliable before window is shown)
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

    # ── 5. Camera ──
    def _build_section_camera(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(self._section_header("Camera Model"))

        card, grid = _section_card(); row = 0

        self.cmb_camera = _combo(min_w=_W.COMBO)
        for model in PINHOLE_MODELS:
            self.cmb_camera.addItem(f"{model.name} ({model.num_params}p)", model.model_id)
        self.cmb_camera.insertSeparator(len(PINHOLE_MODELS))
        for model in FISHEYE_MODELS:
            self.cmb_camera.addItem(f"{model.name} ({model.num_params}p)", model.model_id)
        idx = self.cmb_camera.findData(2)
        if idx >= 0: self.cmb_camera.setCurrentIndex(idx)
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_changed)
        _grid_row(grid, row, "Model", self.cmb_camera); row += 1

        # "Params" label in col 0 (aligned with Model) + toggle button in col 1-2
        self.btn_toggle_cam_params = QPushButton("+ Show")
        self.btn_toggle_cam_params.clicked.connect(self._toggle_cam_params)
        grid.addWidget(_label("Params"), row, 0)
        grid.addWidget(self.btn_toggle_cam_params, row, 1, 1, 2); row += 1

        # Dynamic params area
        self.camera_params_w = QWidget()
        self.camera_params_ly = QGridLayout(self.camera_params_w)
        self.camera_params_ly.setContentsMargins(0, 0, 0, 0)
        self.camera_params_ly.setHorizontalSpacing(8); self.camera_params_ly.setVerticalSpacing(2)
        self.camera_params_ly.setColumnStretch(1, 1)
        grid.addWidget(self.camera_params_w, row, 0, 1, 3)
        self.camera_params_w.hide()
        self._on_camera_changed()

        ly.addWidget(card); return w

    def _toggle_cam_params(self) -> None:
        if not hasattr(self, "_cam_params_expanded"):
            self._cam_params_expanded = False
        self._cam_params_expanded = not self._cam_params_expanded
        self.camera_params_w.setVisible(self._cam_params_expanded)
        self.btn_toggle_cam_params.setText("- Hide" if self._cam_params_expanded else "+ Show")

    def _on_camera_changed(self) -> None:
        while self.camera_params_ly.count():
            it = self.camera_params_ly.takeAt(0)
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
            self.camera_params_ly.addWidget(_label(f"{pname}"), r, 0)
            self.camera_params_ly.addWidget(sb, r, 1, 1, 2)
            self._param_spinboxes.append(sb)

    # ── 6. Output & Build ──
    def _build_section_output(self) -> QWidget:
        w = QWidget(); ly = QVBoxLayout(w); ly.setContentsMargins(0, 0, 0, 0); ly.setSpacing(2)
        ly.addWidget(self._section_header("Output"))

        card, grid = _section_card(); row = 0

        # Folder label in col 0, Browse button in col 1-2 (aligned with all other fields)
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_output)
        grid.addWidget(_label("Folder"), row, 0)
        grid.addWidget(btn_browse, row, 1, 1, 2); row += 1

        # Path display below the button
        self.edit_output = QLabel("(not set)"); self.edit_output.setObjectName("hintLabel")
        self.edit_output.setWordWrap(True)
        grid.addWidget(self.edit_output, row, 1, 1, 2); row += 1

        self.progress = QProgressBar(); self.progress.setVisible(False)
        grid.addWidget(self.progress, row, 0, 1, 3); row += 1

        self.btn_run = QPushButton("Build COLMAP Database"); self.btn_run.setObjectName("btnRun")
        self.btn_run.setIconSize(QSize(16, 16))
        self.btn_run.clicked.connect(self._run_pipeline)
        grid.addWidget(self.btn_run, row, 0, 1, 3); row += 1

        # Secondary action: reveal the output folder in the OS file manager.
        # Disabled until an output directory is chosen.
        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setObjectName("btnSecondary")
        self.btn_open_output.setIcon(_icon_folder())
        self.btn_open_output.setIconSize(QSize(14, 14))
        self.btn_open_output.setEnabled(False)
        self.btn_open_output.clicked.connect(self._open_output)
        grid.addWidget(self.btn_open_output, row, 0, 1, 3); row += 1

        self.lbl_result = QLabel(""); self.lbl_result.setWordWrap(True)
        self.lbl_result.setObjectName("resultLabel"); self.lbl_result.setVisible(False)
        grid.addWidget(self.lbl_result, row, 0, 1, 3)

        ly.addWidget(card); return w

    # ==================================================================
    # Input handlers
    # ==================================================================
    def _on_clear_input(self) -> None:
        self._video_paths.clear(); self._image_paths.clear(); self._video_info.clear()
        self._all_output_images.clear(); self._current_image = ""; self._preview_full = None
        self.preview_label.setText("Preview appears after Build (Ctrl+B)")
        self._update_input_info()

    def _on_input_context_menu(self, pos) -> None:
        item = self.input_list.itemAt(pos)
        if not item: return
        idx = self.input_list.row(item)
        all_vid = len(self._video_paths)
        if idx < all_vid:
            del self._video_paths[idx]
            self._video_info = analyze_videos(self._video_paths)
            self._update_extract_preview()
        else:
            img_idx = idx - all_vid
            if img_idx < len(self._image_paths):
                del self._image_paths[img_idx]
        self._update_input_info()

    def _update_input_info(self) -> None:
        parts = []
        if self._video_paths: parts.append(f"{len(self._video_paths)} video(s)")
        if self._image_paths:
            total = sum(len(collect_image_files([f])) for f in self._image_paths)
            parts.append(f"{total} images")
        self.lbl_input_info.setText("  ·  ".join(parts) if parts else "No input selected.")
        self.btn_clear_input.setEnabled(bool(self._video_paths or self._image_paths))
        self._refresh_input_list(); self._update_image_list()

    def _refresh_input_list(self) -> None:
        self.input_list.clear()
        sec = self._theme_color_secondary()
        for p in self._video_paths:
            self.input_list.addItem(QListWidgetItem(_icon_video(sec), Path(p).name))
        for p in self._image_paths:
            n = len(collect_image_files([p]))
            self.input_list.addItem(QListWidgetItem(_icon_folder(sec), f"{Path(p).name}  ({n})"))

    def _on_add_videos(self) -> None:
        flt = "Videos (%s);;All Files (*)" % " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", flt)
        for f in files:
            if f not in self._video_paths: self._video_paths.append(f)
        if files:
            self._video_info = analyze_videos(self._video_paths)
            self._update_extract_preview()
        self._update_input_info()

    def _on_add_images(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder and folder not in self._image_paths:
            self._image_paths.append(folder)
            self._update_input_info()

    def _update_image_list(self) -> None:
        self._all_output_images = []
        for folder in self._image_paths:
            self._all_output_images.extend(collect_image_files([folder]))
        self._lbl_img_count.setText(f"Images: {len(self._all_output_images)}")
        if self._all_output_images:
            img = cv2.imread(self._all_output_images[0])
            if img is not None:
                h, w = img.shape[:2]; self._update_image_dims(w, h)
        self.thumb_list.clear()
        for p in self._all_output_images:
            self.thumb_list.addItem(Path(p).name)
        self._preview_key = None; self._refresh_preview()

    # ==================================================================
    # Preview
    # ==================================================================
    def _refresh_preview(self, _=None) -> None:
        if not self._current_image and self._all_output_images:
            self._current_image = self._all_output_images[0]
        if not self._current_image:
            self.preview_label.setText("Preview appears after Build (Ctrl+B)")
            self._preview_full = None; return

        key = (self._current_image, self.slider_opacity.value(), self.chk_show_mask.isChecked())
        if key == self._preview_key: return
        self._preview_key = key

        img = cv2.imread(self._current_image)
        if img is None:
            self.preview_label.setText("Cannot load image"); return

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self._mask_cache.get(self._current_image)
        if mask is not None and self.chk_show_mask.isChecked():
            alpha = self.slider_opacity.value() / 100.0
            red = np.zeros_like(img_rgb); red[:, :, 0] = 255
            mask_bin = (mask > 127).astype(np.float32)
            if mask_bin.ndim == 2: mask_bin = mask_bin[:, :, np.newaxis]
            img_rgb = (img_rgb * (1 - alpha * mask_bin) + red * (alpha * mask_bin)).astype(np.uint8)

        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._preview_full = QPixmap.fromImage(qimg)
        self._fit_preview()

        name = Path(self._current_image).name
        idx = self._all_output_images.index(self._current_image)
        self.lbl_preview_info.setText(f"{name}  {w}×{h}  [{idx + 1}/{len(self._all_output_images)}]")

    def _fit_preview(self) -> None:
        if self._preview_full is None: return
        lw, lh = self.preview_label.width(), self.preview_label.height()
        if lw <= 0 or lh <= 0: return
        scaled = self._preview_full.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event); self._fit_preview()

    def _prev_image(self) -> None:
        if not self._all_output_images: return
        idx = self._all_output_images.index(self._current_image) if self._current_image in self._all_output_images else -1
        self._current_image = self._all_output_images[(idx - 1) % len(self._all_output_images)]
        self._preview_key = None; self._refresh_preview()

    def _next_image(self) -> None:
        if not self._all_output_images: return
        idx = self._all_output_images.index(self._current_image) if self._current_image in self._all_output_images else -1
        self._current_image = self._all_output_images[(idx + 1) % len(self._all_output_images)]
        self._preview_key = None; self._refresh_preview()

    def _on_thumb_selected(self, row: int) -> None:
        if 0 <= row < len(self._all_output_images):
            self._current_image = self._all_output_images[row]
            self._preview_key = None; self._refresh_preview()

    def _switch_preview_to_output(self, images_dir: str) -> None:
        images = collect_image_files([images_dir])
        if not images: return
        self._all_output_images = images; self._current_image = images[0]
        self._lbl_img_count.setText(f"Images: {len(images)}")
        img = cv2.imread(images[0])
        if img is not None:
            h, w = img.shape[:2]; self._update_image_dims(w, h)
        self.thumb_list.clear()
        for p in images: self.thumb_list.addItem(Path(p).name)
        masks_dir = os.path.join(os.path.dirname(images_dir), "masks")
        if os.path.isdir(masks_dir):
            stem_to_path = {Path(ip).stem: ip for ip in images}
            for mp in sorted(os.listdir(masks_dir)):
                if mp.endswith("_mask.png"):
                    stem = mp[:-len("_mask.png")]
                    ip = stem_to_path.get(stem)
                    if ip:
                        mask = cv2.imread(os.path.join(masks_dir, mp), cv2.IMREAD_GRAYSCALE)
                        if mask is not None: self._mask_cache[ip] = mask
        self._preview_key = None; self._refresh_preview()

    # ==================================================================
    # Model loading
    # ==================================================================
    def _load_models(self) -> None:
        import yaml
        models_root = os.path.join(os.path.expanduser("~"), "anylabeling_data", "models")
        self._model_configs = []
        if not os.path.isdir(models_root):
            self._lbl_model_count.setText("  Models: 0"); return
        for model_name in sorted(os.listdir(models_root)):
            model_dir = os.path.join(models_root, model_name)
            cfg_path = os.path.join(model_dir, "config.yaml")
            if not os.path.isfile(cfg_path): continue
            try:
                with open(cfg_path) as f: cfg = yaml.safe_load(f)
            except Exception: continue
            if cfg.get("type") not in ("segment_anything", "segment_anything_model"): continue
            onnx_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".onnx"))
            encoder = cfg.get("encoder_model_path", "")
            decoder = cfg.get("decoder_model_path", "")
            lang = cfg.get("language_encoder_path", "")
            ep = os.path.join(model_dir, encoder) if encoder else ""
            dp = os.path.join(model_dir, decoder) if decoder else ""
            lp = os.path.join(model_dir, lang) if lang else ""
            if not dp or not os.path.isfile(dp):
                decs = [f for f in onnx_files if "decoder" in f.lower()]
                if decs: dp = os.path.join(model_dir, decs[0])
            if not ep or not os.path.isfile(ep):
                encs = [f for f in onnx_files if "encoder" in f.lower() and "language" not in f.lower()]
                if encs: ep = os.path.join(model_dir, encs[0])
            if not lp or not os.path.isfile(lp):
                langs = [f for f in onnx_files if "language" in f.lower() or "text" in f.lower()]
                if langs: lp = os.path.join(model_dir, langs[0])
            if not os.path.isfile(dp): continue
            merged = dict(cfg); merged["encoder_model_path"] = ep; merged["decoder_model_path"] = dp
            if lp and os.path.isfile(lp): merged["language_encoder_path"] = lp
            self._model_configs.append(merged)
        self.cmb_sam_model.clear()
        # ── SkyWater: served from HuggingFace Hub (Realcat/skywater_seg) ──
        # No hardcoded local paths — the model auto-downloads on first use
        # and is cached in the standard HF cache (~/.cache/huggingface/hub).
        # If the file is already cached we show "[READY]", otherwise we show
        # "[Download ~48 MB]" to set user expectations before they click Build.
        from .model_downloader import resolve_cached_path, is_huggingface_hub_available
        sw_logical_name = "skywater_segformer_b2_fp16"
        sw_cached_path = resolve_cached_path(sw_logical_name)
        if sw_cached_path:
            sw_cfg = {
                "type": "skywater",
                "name": "SkyWater SegFormer-B2",
                "display_name": "SkyWater (Sky/Water/Person) [FAST]",
                "model_path": sw_cached_path,
                "encoder_model_path": sw_cached_path,
                "decoder_model_path": sw_cached_path,
                "logical_name": sw_logical_name,
            }
        else:
            # Not yet downloaded — SegmentationWorker will fetch on first run.
            # model_path=None is the signal for "needs download".
            download_tag = "[Download ~48 MB]" if is_huggingface_hub_available() else "[HF Hub missing]"
            sw_cfg = {
                "type": "skywater",
                "name": "SkyWater SegFormer-B2",
                "display_name": f"SkyWater (Sky/Water/Person) {download_tag}",
                "model_path": None,
                "encoder_model_path": None,
                "decoder_model_path": None,
                "logical_name": sw_logical_name,
            }
        self.cmb_sam_model.addItem(sw_cfg["display_name"], sw_cfg)
        for cfg in self._model_configs:
            label = cfg.get("display_name", cfg.get("name", ""))
            has_text = " [TEXT]" if cfg.get("language_encoder_path") else ""
            self.cmb_sam_model.addItem(f"{label}{has_text}", cfg)
        self._lbl_model_count.setText(f"  Models: {self.cmb_sam_model.count()}")

        # Re-apply preset now that models are loaded (ensures SkyWater + preset work together)
        self._apply_preset()

    # ==================================================================
    # Pipeline
    # ==================================================================
    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._output_dir = folder; self.edit_output.setText(folder)
            # Enable "Open Output Folder" once a valid directory is chosen.
            self.btn_open_output.setEnabled(os.path.isdir(self._output_dir))

    def _open_output(self) -> None:
        """Reveal the output directory in the OS file manager (cross-platform)."""
        if not self._output_dir:
            return
        # Fall back to opening the parent if the folder was deleted externally.
        target = self._output_dir if os.path.isdir(self._output_dir) else os.path.dirname(self._output_dir)
        if not target or not os.path.isdir(target):
            QMessageBox.warning(self, "Open Output", f"Folder not found:\n{self._output_dir}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as e:
            logger.warning("Failed to open output folder: %s", e)
            QMessageBox.warning(self, "Open Output", f"Could not open folder:\n{e}")

    def _get_seg_targets(self) -> list[str]:
        checked = [k for k, cb in self._class_checks.items() if cb.isChecked()]
        return checked + self._custom_classes

    def _resize_cfg(self):
        """Return current resize config as a hashable tuple, or None if disabled."""
        if not self.chk_resize.isChecked():
            return None
        mode = self.cmb_resize_mode.currentData()
        if mode == "max_dim":
            return ("max_dim", self.spin_max_dim.value())
        elif mode == "downscale":
            return ("downscale", self._ds_factor)
        return (mode,)

    def _run_pipeline(self) -> None:
        if not self._output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output directory first."); return
        # Clear previous result icon
        self.btn_run.setIcon(QIcon())
        self.lbl_result.setVisible(False)
        images_dir = os.path.join(self._output_dir, "images")
        masks_dir = os.path.join(self._output_dir, "masks")
        db_path = os.path.join(self._output_dir, "database.db")
        images_exist = os.path.isdir(images_dir) and collect_image_files([images_dir])

        if not images_exist:
            if not self._video_paths and not self._image_paths:
                QMessageBox.warning(self, "Missing Input", "Add videos or images first."); return
            os.makedirs(images_dir, exist_ok=True)
            for folder in self._image_paths:
                for src in collect_image_files([folder]):
                    dst = os.path.join(images_dir, os.path.basename(src))
                    if src != dst: shutil.copy2(src, dst)
            if self._video_paths:
                self._run_extraction(images_dir, masks_dir, db_path)
            else:
                self._apply_resize(images_dir)
                self._switch_preview_to_output(images_dir)
                self._run_seg_or_db(images_dir, masks_dir, db_path)
        else:
            # --- Re-build path: images already exist ---
            # Detect resize config change → re-process images from source
            cur_resize = self._resize_cfg()
            if cur_resize != self._last_resize_cfg:
                # Resize params changed (or first build): re-extract / re-copy + resize from source
                if os.path.isdir(images_dir):
                    import shutil as _s; _s.rmtree(images_dir)
                os.makedirs(images_dir, exist_ok=True)
                if os.path.isdir(masks_dir):
                    import shutil as _s; _s.rmtree(masks_dir)
                os.makedirs(masks_dir, exist_ok=True)
                if os.path.isfile(db_path): os.remove(db_path)
                self._mask_cache.clear(); self._preview_key = None
                self.btn_run.setEnabled(False); self.progress.setVisible(True); self.progress.setValue(0)
                if self._video_paths:
                    # Re-extract frames (resize applied during extraction)
                    self._run_extraction(images_dir, masks_dir, db_path)
                    return
                else:
                    # Re-copy from source folders + apply resize
                    for folder in self._image_paths:
                        for src in collect_image_files([folder]):
                            dst = os.path.join(images_dir, os.path.basename(src))
                            if src != dst: shutil.copy2(src, dst)
                    self._apply_resize(images_dir)
                    self._switch_preview_to_output(images_dir)
                    self._run_seg_or_db(images_dir, masks_dir, db_path)
                    return
            # Resize unchanged: just re-run segmentation / database
            if self.chk_seg.isChecked() and not self._get_seg_targets():
                QMessageBox.warning(self, "No Target Classes",
                    "Segmentation enabled but no target classes selected."); return
            if os.path.isdir(masks_dir):
                import shutil as _s; _s.rmtree(masks_dir)
            os.makedirs(masks_dir, exist_ok=True)
            if os.path.isfile(db_path): os.remove(db_path)
            self._mask_cache.clear(); self._preview_key = None
            self.btn_run.setEnabled(False); self.progress.setVisible(True); self.progress.setValue(0)
            self._phase = "re-segmentation"
            self._switch_preview_to_output(images_dir)
            self._run_seg_or_db(images_dir, masks_dir, db_path)

    def _apply_resize(self, images_dir: str) -> None:
        if not self.chk_resize.isChecked(): return
        mode = self.cmb_resize_mode.currentData()
        from .utils import resize_image
        for p in collect_image_files([images_dir]):
            img = cv2.imread(p)
            if img is None: continue
            h, w = img.shape[:2]
            if mode == "downscale":
                factor = self._ds_factor
                if factor <= 1: continue
                img = resize_image(img, width=w // factor, height=h // factor, keep_aspect=False)
            elif mode == "max_dim":
                img = resize_image(img, max_dim=self.spin_max_dim.value())
            cv2.imwrite(p, img)

    def _run_extraction(self, images_dir, masks_dir, db_path) -> None:
        method = self.cmb_extract_method.currentData()
        self._phase = "extraction"
        r_mode = ""; r_max = 0; r_factor = 1
        if self.chk_resize.isChecked():
            r_mode = self.cmb_resize_mode.currentData()
            if r_mode == "max_dim": r_max = self.spin_max_dim.value()
            elif r_mode == "downscale": r_factor = self._ds_factor
        worker = FrameExtractionWorker(
            video_paths=self._video_paths, output_dir=images_dir,
            method=method, interval=self.spin_interval.value(),
            target_fps=self.spin_target_fps.value(),
            max_frames=self.spin_max_frames.value() or None,
            resize_mode=r_mode, resize_max_dim=r_max, resize_factor=r_factor,
            output_format=self.cmb_format.currentData(),
            jpg_quality=95 if "95" in self.cmb_format.currentText() else 85)
        self._active_worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda p: self._on_extract_done(p, images_dir, masks_dir, db_path))
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_extract_done(self, paths, images_dir, masks_dir, db_path) -> None:
        self._switch_preview_to_output(images_dir)
        self._run_seg_or_db(images_dir, masks_dir, db_path)

    def _run_seg_or_db(self, images_dir, masks_dir, db_path) -> None:
        if self.chk_seg.isChecked() and self._get_seg_targets():
            cfg = self.cmb_sam_model.currentData()
            if not cfg: self._on_error("No segmentation model selected"); return
            is_sw = "skywater" in cfg.get("type", "").lower()
            resolved = dict(cfg)
            if not is_sw:
                dp = cfg.get("decoder_model_path", "")
                if not dp or not os.path.isfile(dp):
                    self._on_error(f"Model file not found: {dp}"); return
            else:
                # SkyWater: if not yet cached, ensure huggingface_hub is
                # installed before handing off to the worker — otherwise
                # the user gets a confusing ImportError from a background thread.
                sw_path = cfg.get("model_path")
                if not sw_path or not os.path.isfile(sw_path):
                    from .model_downloader import is_huggingface_hub_available
                    if not is_huggingface_hub_available():
                        self._on_error(
                            "SkyWater model needs to be downloaded from HuggingFace, "
                            "but huggingface_hub is not installed.\n"
                            "Install with:  uv pip install huggingface_hub"
                        ); return
                    # Confirm with the user before triggering a ~48 MB download
                    info = cfg.get("logical_name", "skywater_segformer_b2_fp16")
                    from .model_downloader import get_model_info
                    meta = get_model_info(info) or {}
                    size_hint = meta.get("size_hint_mb", "?")
                    reply = QMessageBox.question(
                        self, "Download SkyWater Model",
                        f"The SkyWater SegFormer-B2 model (~{size_hint} MB) will be "
                        "downloaded from HuggingFace (Realcat/skywater_seg) and cached "
                        "for future use.\n\nProceed?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        self._on_error("SkyWater download cancelled by user"); return
            self._phase = "segmentation"
            worker = SegmentationWorker(
                image_paths=collect_image_files([images_dir]), mask_output_dir=masks_dir,
                model_config=resolved, target_classes=self._get_seg_targets(),
                confidence_threshold=self.spin_conf.value() if not is_sw else 0.0,
                max_inference_dim=512)
            self._active_worker = worker
            worker.signals.progress.connect(self._on_progress)
            worker.signals.image_done.connect(self._on_mask_ready)
            worker.signals.finished.connect(lambda p: self._on_seg_done(p, images_dir, db_path))
            worker.signals.error.connect(self._on_error)
            self._thread_pool.start(worker)
        else:
            self._run_db(images_dir, db_path)

    def _on_mask_ready(self, img_path: str, mask_path: str) -> None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            self._mask_cache[img_path] = mask
            if img_path == self._current_image:
                self._preview_key = None; self._refresh_preview()
            if not self._current_image or self._current_image not in self._all_output_images:
                self._current_image = img_path
                self._preview_key = None; self._refresh_preview()

    def _on_seg_done(self, paths, images_dir, db_path) -> None:
        self._run_db(images_dir, db_path)

    def _run_db(self, images_dir, db_path) -> None:
        mid = self.cmb_camera.currentData()
        params = [sb.value() for sb in self._param_spinboxes]
        self._phase = "database"
        worker = DatabaseBuildWorker(
            image_dir=images_dir, db_path=db_path, camera_model_id=mid, camera_params=params)
        self._active_worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_db_done)
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_db_done(self, db_path: str) -> None:
        self.progress.setValue(100); self.progress.setVisible(False)
        self.btn_run.setEnabled(True); self._active_worker = None
        # Record resize config so re-build can detect changes
        self._last_resize_cfg = self._resize_cfg()
        images_dir = os.path.join(self._output_dir, "images")
        self.lbl_result.setText(f"Database ready\n{db_path}\nOpen this DB + {images_dir} in COLMAP")
        self.lbl_result.setVisible(True)
        self.btn_run.setIcon(_icon_check())
        self.status_bar.showMessage(f"Database ready: {db_path}")

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(pct); self.status_bar.showMessage(msg)
        if self._phase == "extraction" and pct % 10 == 0:
            self._switch_preview_to_output(os.path.join(self._output_dir, "images"))

    def _on_error(self, msg: str) -> None:
        self.progress.setVisible(False); self.btn_run.setEnabled(True); self._active_worker = None
        QMessageBox.critical(self, "Error", f"Pipeline failed ({self._phase}):\n{msg}")

    # ── Scroll blocker ──
    def _install_scroll_blocker(self) -> None:
        for widget in self.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
            widget.installEventFilter(self._scroll_blocker)
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
