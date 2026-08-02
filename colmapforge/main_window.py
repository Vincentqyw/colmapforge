"""
Single-page COLMAP preprocessing tool — MVC view layer (coordinator).

Owns shared state, instantiates section widgets, toolbar, preview panel,
and pipeline orchestrator. Wires signals between them.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox,
    QScrollArea, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from .about_dialog import AboutDialog
from .camera_models import DEFAULT_CAMERA_MODEL_ID
from .logo import make_app_icon, make_logo_pixmap
from .pipeline import PipelineConfig, PipelineOrchestrator
from .pipeline_core import MASK_SUFFIX, output_layout
from .theme import Theme, apply_theme
from .utils import (
    analyze_videos, collect_image_files, colmap_gui_command, compute_frame_indices,
)
from .views.constants import _S
from .views.icons import (
    _icon_folder, _icon_video,
)
from .views.preview_panel import PreviewPanel
from .views.section_camera import CameraSection
from .views.section_extract import ExtractSection
from .views.section_input import InputSection
from .views.section_output import OutputSection
from .views.section_resize import ResizeSection
from .views.section_segmentation import SegmentationSection
from .views.toolbar import AppToolbar
from .views.widgets import ScrollBlocker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """COLMAP Forge — pixel-aligned, card-based, Apple-style."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("COLMAP Forge")
        self.setMinimumSize(960, 900)
        self.setWindowIcon(make_app_icon())

        # ── shared state ──
        self._theme = Theme.get()
        self._detect_system_theme()
        self._video_paths: list[str] = []
        self._image_paths: list[str] = []
        self._video_info: list[dict] = []
        self._model_configs: list[dict] = []
        self._output_dir = ""
        self._mask_cache: dict[str, str] = {}  # image path → mask path
        self._all_output_images: list[str] = []
        self._last_count_scan = 0.0

        # ── create widgets ──
        self._pipeline = PipelineOrchestrator(self)
        self._toolbar = AppToolbar()
        self._input_section = InputSection()
        self._extract_section = ExtractSection()
        self._resize_section = ResizeSection()
        self._seg_section = SegmentationSection()
        self._camera_section = CameraSection()
        self._output_section = OutputSection()
        self._preview = PreviewPanel(self._theme_color_text())

        self._scroll_blocker = ScrollBlocker(self)

        self._setup_ui()
        self._setup_shortcuts()
        self._wire_signals()
        self._load_models()
        self._apply_theme()
        self._install_scroll_blocker()
        self._refresh_gpu_status()
        QTimer.singleShot(200, self._preview.refresh_preview)

    def showEvent(self, event) -> None:
        """Defer resize until window frame geometry is resolved (avoids height
        jump when dragging the title bar on Windows after first show)."""
        super().showEvent(event)
        self.resize(1200, 1000)

    # ═══════════════════════════════════════════════════════════════════
    # Theme
    # ═══════════════════════════════════════════════════════════════════

    def _theme_color_text(self) -> str:
        return "#ffffff" if self._theme.is_dark else "#1c1c1e"

    def _theme_color_secondary(self) -> str:
        return "#8e8e93"

    def _detect_system_theme(self) -> None:
        try:
            import darkdetect
            self._theme.set_dark(darkdetect.isDark())
        except ImportError:
            pass

    def _refresh_gpu_status(self) -> None:
        from .onnx_utils import diagnose
        diag = diagnose()
        if not diag["installed"]:
            self._toolbar.set_gpu_status("  ONNX: missing", "gpuStatusErr")
        elif diag["issues"]:
            self._toolbar.set_gpu_status(f"  GPU: broken ({diag['version']})", "gpuStatusErr")
        elif diag["gpu_active"]:
            self._toolbar.set_gpu_status(f"  GPU: {diag['active_provider']}", "gpuStatusOk")
        else:
            self._toolbar.set_gpu_status("  GPU: CPU only", "gpuStatusWarn")

    def _show_onnx_diagnostics(self) -> None:
        from .onnx_utils import diagnose
        diag = diagnose()
        if not diag["installed"]:
            QMessageBox.warning(self, "ONNX Runtime — Not Installed",
                "ONNX Runtime is not installed. Segmentation features will not work.\n\n"
                + (diag.get("hint") or "")); return
        lines = [
            f"ONNX Runtime version: {diag['version']}",
            f"Installed wheels: {', '.join(diag['installed_wheels']) or '(none)'}",
            f"Available providers: {', '.join(diag['providers'])}",
            f"GPU active: {diag['gpu_active']}"
            + (f" ({diag['active_provider']})" if diag['active_provider'] else ""),
        ]
        if diag["issues"]:
            lines.append(""); lines.append("Issues detected:")
            for i, issue in enumerate(diag["issues"], 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append(""); lines.append("No issues detected. Setup is healthy.")
        if diag["issues"]:
            QMessageBox.warning(self, "ONNX Runtime Diagnostics", "\n".join(lines))
        else:
            QMessageBox.information(self, "ONNX Runtime Diagnostics", "\n".join(lines))

    def _apply_theme(self) -> None:
        apply_theme(QApplication.instance(), self._theme)
        if hasattr(self, "_left_scroll"):
            sa = self._left_scroll
            for w in (sa, sa.viewport(), sa.widget()):
                if w is None: continue
                w.style().unpolish(w); w.style().polish(w); w.update()
        self.setWindowIcon(make_app_icon())
        self._toolbar.set_logo(make_logo_pixmap(24))
        self._toolbar.refresh_theme_icons(self._theme.is_dark)
        self._preview.set_theme_colors(self._theme_color_text())
        self._refresh_input_list()
        self.status_bar.showMessage("Ready — Ctrl+O add images · Ctrl+B build")

    def _toggle_theme(self) -> None:
        self._theme.toggle(); self._apply_theme(); self._preview.refresh_preview()

    def _show_about(self) -> None:
        AboutDialog(self, is_dark=self._theme.is_dark).exec()

    # ═══════════════════════════════════════════════════════════════════
    # Shortcuts
    # ═══════════════════════════════════════════════════════════════════

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_add_images)
        QShortcut(QKeySequence("Ctrl+B"), self, self._run_pipeline)
        QShortcut(QKeySequence("Ctrl+T"), self, self._toggle_theme)
        QShortcut(QKeySequence("Right"), self, self._preview.next_image)
        QShortcut(QKeySequence("Left"), self, self._preview.prev_image)

    # ═══════════════════════════════════════════════════════════════════
    # UI layout
    # ═══════════════════════════════════════════════════════════════════

    def _setup_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._preview)
        splitter.setCollapsible(0, False)  # prevent left panel from disappearing when dragged
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 780])
        root.addWidget(splitter, 1)

        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)

    def _make_left_panel(self) -> QWidget:
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340); scroll.setMaximumWidth(480)
        scroll.setObjectName("leftPanel")
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._left_scroll = scroll

        w = QWidget(); w.setObjectName("leftPanelContent")
        ly = QVBoxLayout(w)
        ly.setSpacing(_S.SECTION_GAP); ly.setContentsMargins(10, 8, 10, 8)

        ly.addWidget(self._input_section)
        ly.addWidget(self._extract_section)
        ly.addWidget(self._resize_section)
        ly.addWidget(self._seg_section)
        ly.addWidget(self._camera_section)
        ly.addWidget(self._output_section)
        ly.addStretch()

        scroll.setWidget(w); return scroll

    # ═══════════════════════════════════════════════════════════════════
    # Signal wiring
    # ═══════════════════════════════════════════════════════════════════

    def _wire_signals(self) -> None:
        # Toolbar
        self._toolbar.theme_toggled.connect(self._toggle_theme)
        self._toolbar.about_clicked.connect(self._show_about)
        self._toolbar.gpu_status_clicked.connect(self._show_onnx_diagnostics)

        # Input section
        self._input_section.videos_requested.connect(self._on_add_videos)
        self._input_section.images_requested.connect(self._on_add_images)
        self._input_section.clear_requested.connect(self._on_clear_input)
        self._input_section.context_menu_remove.connect(self._on_input_context_menu)

        # Extract section (the other sections need no reactive updates —
        # their state is read once when the pipeline starts)
        self._extract_section.config_changed.connect(self._update_extract_preview)

        # Output section
        self._output_section.browse_requested.connect(self._browse_output)
        self._output_section.run_clicked.connect(self._run_pipeline)
        self._output_section.stop_clicked.connect(self._stop_pipeline)
        self._output_section.open_output_requested.connect(self._open_output)
        self._output_section.launch_colmap_clicked.connect(self._on_launch_colmap_clicked)

        # Pipeline
        self._pipeline.progress.connect(self._on_pipeline_progress)
        self._pipeline.mask_ready.connect(self._on_pipeline_mask_ready)
        self._pipeline.pipeline_finished.connect(self._on_pipeline_finished)
        self._pipeline.preview_switch.connect(self._switch_preview_to_output)
        self._pipeline.error.connect(self._on_pipeline_error)

    # ═══════════════════════════════════════════════════════════════════
    # Input handlers
    # ═══════════════════════════════════════════════════════════════════

    def _on_clear_input(self) -> None:
        self._video_paths.clear(); self._image_paths.clear(); self._video_info.clear()
        self._all_output_images.clear()
        self._preview.clear()
        self._update_input_info()

    def _on_input_context_menu(self, idx: int) -> None:
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
        self._input_section.set_info("  ·  ".join(parts) if parts else "No input selected.")
        self._input_section.set_clear_enabled(bool(self._video_paths or self._image_paths))
        self._refresh_input_list(); self._update_image_list()

    def _refresh_input_list(self) -> None:
        sec = self._theme_color_secondary()
        items = []
        for p in self._video_paths:
            tip = self._video_tooltip(p)
            items.append((_icon_video(sec), Path(p).name, tip))
        for p in self._image_paths:
            n = len(collect_image_files([p]))
            items.append((_icon_folder(sec), f"{Path(p).name}  ({n})", ""))
        self._input_section.refresh_list(items)

    def _video_tooltip(self, path: str) -> str:
        """Build a rich tooltip showing video metadata."""
        for vi in self._video_info:
            if vi["path"] == path:
                duration = vi.get("duration_seconds", 0)
                mm, ss = int(duration // 60), int(duration % 60)
                size_mb = vi.get("file_size_mb", 0)
                return (
                    f"Resolution: {vi['width']} × {vi['height']}\n"
                    f"FPS: {vi['fps']:.2f}\n"
                    f"Frames: {vi['total_frames']}\n"
                    f"Duration: {mm}:{ss:02d}\n"
                    f"Codec: {vi.get('fourcc', 'N/A')}\n"
                    f"Size: {size_mb:.1f} MB"
                )
        return ""

    def _on_add_videos(self) -> None:
        from .utils import VIDEO_EXTENSIONS
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
        self._toolbar.set_image_count(len(self._all_output_images))
        if self._all_output_images:
            img = cv2.imread(self._all_output_images[0])
            if img is not None:
                h, w = img.shape[:2]
                self._update_image_dims(w, h)
        elif self._video_info:
            # No image folders yet — pull dims from first video's metadata
            vi = self._video_info[0]
            self._update_image_dims(vi["width"], vi["height"])
        self._preview.set_images(self._all_output_images)
        self._preview.refresh_preview()

    def _update_image_dims(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0: return
        self._resize_section.set_image_dims(w, h)
        self._camera_section.set_image_dims(w, h)

    def _update_extract_preview(self) -> None:
        if not self._video_info or not self._extract_section.is_enabled:
            self._extract_section.set_preview_text(""); return
        method = self._extract_section.method
        lines = []
        for info in self._video_info:
            try:
                if method == "interval":
                    frames = compute_frame_indices(
                        info["total_frames"], info["fps"], method="interval",
                        interval=self._extract_section.interval,
                        max_frames=self._extract_section.max_frames)
                elif method == "target_fps":
                    frames = compute_frame_indices(
                        info["total_frames"], info["fps"], method="target_fps",
                        target_fps=self._extract_section.target_fps,
                        max_frames=self._extract_section.max_frames)
                else:
                    continue
                lines.append(f"{Path(info['path']).name}: {len(frames)} frames")
            except Exception:
                pass
        self._extract_section.set_preview_text("  ·  ".join(lines) if lines else "")

    # ═══════════════════════════════════════════════════════════════════
    # Preview bridge
    # ═══════════════════════════════════════════════════════════════════

    def _switch_preview_to_output(self, images_dir: str) -> None:
        images = collect_image_files([images_dir])
        if not images: return
        self._all_output_images = images
        self._toolbar.set_image_count(len(images))
        self._preview.set_images(images)
        self._preview.set_current_image(images[0])
        img = cv2.imread(images[0])
        if img is not None:
            h, w = img.shape[:2]
            self._camera_section.set_image_dims(w, h)
        masks_dir = os.path.join(os.path.dirname(images_dir), "masks")
        if os.path.isdir(masks_dir):
            stem_to_path = {Path(ip).stem: ip for ip in images}
            for mp in sorted(os.listdir(masks_dir)):
                if mp.endswith(MASK_SUFFIX):
                    ip = stem_to_path.get(mp[:-len(MASK_SUFFIX)])
                    if ip:
                        self._mask_cache[ip] = os.path.join(masks_dir, mp)
        self._preview.set_mask_cache(self._mask_cache)
        self._preview.refresh_preview()

    # ═══════════════════════════════════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════════════════════════════════

    def _load_models(self) -> None:
        from .model_downloader import discover_models
        self._model_configs = discover_models()
        self._seg_section.populate_models(self._model_configs)
        self._toolbar.set_model_count(self._seg_section.cmb_sam_model.count())
        self._seg_section.apply_preset()

    # ═══════════════════════════════════════════════════════════════════
    # Output
    # ═══════════════════════════════════════════════════════════════════

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._output_dir = folder
            self._output_section.set_output_dir(folder)

    def _open_output(self) -> None:
        if not self._output_dir: return
        target = self._output_dir if os.path.isdir(self._output_dir) else os.path.dirname(self._output_dir)
        if not target or not os.path.isdir(target):
            QMessageBox.warning(self, "Open Output", f"Folder not found:\n{self._output_dir}"); return
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

    # ═══════════════════════════════════════════════════════════════════
    # Pipeline bridge
    # ═══════════════════════════════════════════════════════════════════

    def _gather_config(self) -> PipelineConfig:
        return PipelineConfig(
            video_paths=self._video_paths,
            image_paths=self._image_paths,
            output_dir=self._output_dir,
            extract_enabled=self._extract_section.is_enabled,
            extract_method=self._extract_section.method,
            extract_interval=self._extract_section.interval,
            extract_target_fps=self._extract_section.target_fps,
            extract_max_frames=self._extract_section.max_frames,
            extract_format=self._extract_section.output_format,
            extract_jpg_quality=self._extract_section.jpg_quality,
            resize_enabled=self._resize_section.is_enabled,
            resize_mode=self._resize_section.mode,
            resize_max_dim=self._resize_section.max_dim,
            resize_factor=self._resize_section.ds_factor,
            seg_enabled=self._seg_section.is_enabled,
            seg_model_config=self._seg_section.model_config,
            seg_target_classes=self._seg_section.target_classes,
            seg_confidence=self._seg_section.confidence,
            camera_model_id=self._camera_section.camera_model_id or DEFAULT_CAMERA_MODEL_ID,
            camera_params=self._camera_section.camera_params,
        )

    def _run_pipeline(self) -> None:
        if not self._output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output directory first."); return
        if not self._video_paths and not self._image_paths:
            QMessageBox.warning(self, "Missing Input", "Add videos or images first."); return
        if self._seg_section.is_enabled and not self._seg_section.target_classes:
            QMessageBox.warning(self, "No Target Classes",
                "Segmentation enabled but no target classes selected."); return

        self._output_section.set_busy(True)
        self._mask_cache.clear()
        self._preview.set_mask_cache(self._mask_cache)

        config = self._gather_config()
        self._pipeline.run(config)

    def _stop_pipeline(self) -> None:
        """Cancel the running pipeline."""
        self._pipeline.cancel()
        self._output_section.set_busy(False)
        self.status_bar.showMessage("Pipeline stopped by user")

    def _on_pipeline_progress(self, pct: int, msg: str) -> None:
        self._output_section.set_progress(pct, msg)
        self.status_bar.showMessage(msg)
        # Update toolbar image count in real-time during extraction so the
        # user sees frames accumulating rather than a static number.
        # Throttled: the count comes from a directory scan, which would be
        # O(N²) if run on every per-frame progress tick.
        now = time.monotonic()
        if self._output_dir and now - self._last_count_scan >= 0.5:
            self._last_count_scan = now
            images_dir = output_layout(self._output_dir)[0]
            if os.path.isdir(images_dir):
                count = len(collect_image_files([images_dir]))
                self._toolbar.set_image_count(count)
                # Also refresh the preview panel count text without a full
                # image re-render (lightweight). Works even before the first
                # preview image appears — shows a count badge on the empty
                # preview area.
                self._preview.update_image_count(count)

    def _on_pipeline_mask_ready(self, img_path: str, mask_path: str) -> None:
        self._mask_cache[img_path] = mask_path
        self._preview.set_mask_cache(self._mask_cache)
        self._preview.set_current_image(img_path)
        self._preview.refresh_preview()

    def _on_pipeline_finished(self, db_path: str) -> None:
        images_dir = output_layout(self._output_dir)[0]
        self._output_section.show_result(db_path, images_dir)
        self._preview.set_mask_cache(self._mask_cache)

        if self._output_section.launch_colmap:
            self._launch_colmap(db_path, images_dir)

    def _on_launch_colmap_clicked(self) -> None:
        """Launch COLMAP using the last built database + images."""
        db_path = self._output_section._db_path
        images_dir = self._output_section._images_dir
        if not db_path or not os.path.isfile(db_path):
            self.status_bar.showMessage("No database yet — build it first.")
            return
        self._launch_colmap(db_path, images_dir)

    def _launch_colmap(self, db_path: str, images_dir: str) -> None:
        """Launch COLMAP GUI pointing at the built database and images."""
        masks_dir = os.path.join(os.path.dirname(images_dir), "masks")
        cmd = colmap_gui_command(db_path, images_dir, masks_dir)

        colmap_exe = shutil.which("colmap")
        if colmap_exe is None:
            self.status_bar.showMessage(
                "COLMAP not found on PATH — install it or run manually: " + " ".join(cmd)
            )
            return

        try:
            subprocess.Popen(
                [colmap_exe, *cmd[1:]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.status_bar.showMessage(f"COLMAP launched: {db_path}")
        except Exception as e:
            logger.warning("Failed to launch COLMAP: %s", e)
            self.status_bar.showMessage(f"COLMAP launch failed: {e}")

    def _on_pipeline_error(self, msg: str) -> None:
        self._output_section.reset()
        QMessageBox.critical(self, "Error", msg)

    # ═══════════════════════════════════════════════════════════════════
    # Scroll blocker
    # ═══════════════════════════════════════════════════════════════════

    def _install_scroll_blocker(self) -> None:
        from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
        for widget in self.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
            widget.installEventFilter(self._scroll_blocker)
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
