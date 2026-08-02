"""
Top toolbar widget.

Displays app logo, title, image/model counts, GPU status indicator,
theme toggle, and About button.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .icons import _icon_info, _icon_moon, _icon_sun


class AppToolbar(QWidget):
    """Fixed-height top bar with status labels and action buttons."""

    theme_toggled = pyqtSignal()
    about_clicked = pyqtSignal()
    gpu_status_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36); self.setObjectName("toolbar")
        ly = QHBoxLayout(self); ly.setContentsMargins(14, 4, 14, 4)

        self._lbl_logo = QLabel()
        self._lbl_logo.setFixedSize(24, 24)
        ly.addWidget(self._lbl_logo)
        ly.addSpacing(8)

        title = QLabel("COLMAP Forge"); title.setObjectName("appTitle")
        ly.addWidget(title); ly.addStretch()

        self._lbl_img_count = QLabel("Images: 0"); self._lbl_img_count.setObjectName("statusInfo")
        ly.addWidget(self._lbl_img_count)
        self._lbl_model_count = QLabel("  Models: —"); self._lbl_model_count.setObjectName("statusInfo")
        ly.addWidget(self._lbl_model_count)

        self._lbl_gpu_status = QLabel("  GPU: …")
        self._lbl_gpu_status.setObjectName("gpuStatusUnknown")
        self._lbl_gpu_status.setToolTip("Click to view ONNX Runtime diagnostics")
        self._lbl_gpu_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lbl_gpu_status.mousePressEvent = lambda _e: self.gpu_status_clicked.emit()
        ly.addWidget(self._lbl_gpu_status); ly.addSpacing(10)

        self.btn_theme = QPushButton(); self.btn_theme.setFixedSize(32, 28)
        self.btn_theme.setObjectName("themeBtn")
        self.btn_theme.setIconSize(QSize(20, 20))
        self.btn_theme.setIcon(_icon_sun())
        self.btn_theme.clicked.connect(self.theme_toggled.emit); ly.addWidget(self.btn_theme)

        self.btn_about = QPushButton(); self.btn_about.setFixedSize(32, 28)
        self.btn_about.setObjectName("themeBtn")
        self.btn_about.setIconSize(QSize(20, 20))
        self.btn_about.setIcon(_icon_info())
        self.btn_about.setToolTip("About COLMAP Forge")
        self.btn_about.clicked.connect(self.about_clicked.emit); ly.addWidget(self.btn_about)

    # ── public API ──

    def refresh_theme_icons(self, is_dark: bool) -> None:
        self.btn_theme.setIcon(_icon_moon() if is_dark else _icon_sun())
        self.btn_theme.setText("")

    def set_gpu_status(self, text: str, object_name: str) -> None:
        self._lbl_gpu_status.setText(text)
        self._lbl_gpu_status.setObjectName(object_name)
        self._lbl_gpu_status.style().unpolish(self._lbl_gpu_status)
        self._lbl_gpu_status.style().polish(self._lbl_gpu_status)

    def set_model_count(self, n: int) -> None:
        self._lbl_model_count.setText(f"  Models: {n}" if n > 0 else "  Models: —")

    def set_image_count(self, n: int) -> None:
        self._lbl_img_count.setText(f"Images: {n}")

    def set_logo(self, pixmap: QPixmap) -> None:
        self._lbl_logo.setPixmap(pixmap)
