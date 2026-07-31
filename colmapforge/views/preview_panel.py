"""
Image preview panel with mask overlay, navigation, and thumbnail strip.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from .icons import _icon_left, _icon_right


class PreviewPanel(QWidget):
    """Right-side preview: image display, mask overlay, navigation, thumbnails."""

    def __init__(self, theme_color_text: str, parent=None) -> None:
        super().__init__(parent)
        self._all_output_images: list[str] = []
        self._current_image: str = ""
        self._mask_cache: dict[str, np.ndarray] = {}
        self._preview_key = None
        self._preview_full: QPixmap | None = None
        self._current_size: tuple[int, int] | None = None

        ly = QVBoxLayout(self); ly.setContentsMargins(8, 6, 8, 6); ly.setSpacing(3)

        self.preview_label = QLabel("Preview appears after Build (Ctrl+B)")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 200)
        self.preview_label.setScaledContents(False)
        ly.addWidget(self.preview_label, 1)

        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(0, 2, 0, 2); cl.setSpacing(8)

        self.btn_prev = QPushButton(); self.btn_prev.setFixedWidth(32); self.btn_prev.setToolTip("Previous (Left)")
        self.btn_prev.setIconSize(QSize(16, 16))
        self.btn_prev.setIcon(_icon_left(theme_color_text))
        self.btn_prev.clicked.connect(self.prev_image); cl.addWidget(self.btn_prev)

        self.lbl_preview_info = QLabel("No image"); self.lbl_preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_info.setObjectName("previewInfo"); cl.addWidget(self.lbl_preview_info, 1)

        self.btn_next = QPushButton(); self.btn_next.setFixedWidth(32); self.btn_next.setToolTip("Next (Right)")
        self.btn_next.setIconSize(QSize(16, 16))
        self.btn_next.setIcon(_icon_right(theme_color_text))
        self.btn_next.clicked.connect(self.next_image); cl.addWidget(self.btn_next)

        cl.addWidget(QLabel("Mask:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100); self.slider_opacity.setValue(60)
        self.slider_opacity.setFixedWidth(80)
        self.slider_opacity.valueChanged.connect(self.refresh_preview); cl.addWidget(self.slider_opacity)

        self.chk_show_mask = QCheckBox("Show"); self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self.refresh_preview); cl.addWidget(self.chk_show_mask)
        ly.addWidget(ctrl)

        self.thumb_list = QListWidget(); self.thumb_list.setMaximumHeight(48)
        self.thumb_list.setAlternatingRowColors(True)
        self.thumb_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.thumb_list.currentRowChanged.connect(self._on_thumb_selected)
        ly.addWidget(self.thumb_list)

    # ── public API ──

    def set_images(self, images: list[str]) -> None:
        self._all_output_images = images
        self.thumb_list.clear()
        for p in images: self.thumb_list.addItem(Path(p).name)

    def set_mask_cache(self, cache: dict[str, np.ndarray]) -> None:
        self._mask_cache = cache

    def set_current_image(self, path: str) -> None:
        self._current_image = path

    def set_theme_colors(self, text_color: str) -> None:
        self.btn_prev.setIcon(_icon_left(text_color))
        self.btn_next.setIcon(_icon_right(text_color))

    def refresh_preview(self, _=None) -> None:
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
        self._current_size = (w, h)
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

        self._set_info(w, h)

    def fit_in_view(self) -> None:
        self._fit_preview()

    def prev_image(self) -> None:
        if not self._all_output_images: return
        idx = self._all_output_images.index(self._current_image) if self._current_image in self._all_output_images else -1
        self._current_image = self._all_output_images[(idx - 1) % len(self._all_output_images)]
        self._preview_key = None; self.refresh_preview()

    def next_image(self) -> None:
        if not self._all_output_images: return
        idx = self._all_output_images.index(self._current_image) if self._current_image in self._all_output_images else -1
        self._current_image = self._all_output_images[(idx + 1) % len(self._all_output_images)]
        self._preview_key = None; self.refresh_preview()

    def update_image_count(self, total: int) -> None:
        """Lightweight count-only update — no image re-render.

        Called during frame extraction so the user sees the frame count
        growing in real-time without the cost of reloading all images.
        Keeps the same info format (with dimensions when known) as a full
        refresh, so the label never jumps between layouts.
        """
        if not self._current_image:
            if self._all_output_images:
                self._current_image = self._all_output_images[0]
            else:
                self._set_info(None, None, total=total)
                return
        self._set_info(None, None, total=total)

    def clear(self) -> None:
        self._all_output_images.clear()
        self._current_image = ""
        self._preview_full = None
        self._preview_key = None
        self._current_size = None
        self.lbl_preview_info.setText("")
        self.preview_label.setText("Preview appears after Build (Ctrl+B)")
        self.thumb_list.clear()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event); self._fit_preview()

    # ── internals ──

    def _set_info(self, w: int | None, h: int | None, total: int | None = None) -> None:
        """Update the info label under the preview in one consistent format.

        ``name [idx/N]`` is always shown; ``WxH`` is appended whenever the
        current image's dimensions are known, so the label never changes
        shape between a lightweight count update and a full re-render.
        """
        if not self._current_image or not self._all_output_images:
            self.lbl_preview_info.setText("")
            return

        name = Path(self._current_image).name
        idx = 0
        try:
            idx = self._all_output_images.index(self._current_image)
        except ValueError:
            pass

        size = self._current_size if (w is None or h is None) else (w, h)
        n = total or len(self._all_output_images)
        if size:
            self.lbl_preview_info.setText(f"{name}  {size[0]}×{size[1]}  [{idx + 1}/{n}]")
        else:
            self.lbl_preview_info.setText(f"{name}  [{idx + 1}/{n}]")

    def _on_thumb_selected(self, row: int) -> None:
        if 0 <= row < len(self._all_output_images):
            self._current_image = self._all_output_images[row]
            self._preview_key = None; self.refresh_preview()

    def _fit_preview(self) -> None:
        if self._preview_full is None: return
        lw, lh = self.preview_label.width(), self.preview_label.height()
        if lw <= 0 or lh <= 0: return
        scaled = self._preview_full.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
