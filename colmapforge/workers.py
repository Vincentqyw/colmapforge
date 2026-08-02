"""
QThread workers: frame extraction, segmentation, DB creation.

Thin QRunnable wrappers around the pure functions in ``pipeline_core``:
each worker only translates progress/image-done callbacks into Qt signals
and exposes cooperative cancellation via its ``_running`` flag.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Signal containers
# ═══════════════════════════════════════════════════════════════════════

class FrameExtractionSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

class SegmentationSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    image_done = pyqtSignal(str, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

class DatabaseBuildSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)


# ═══════════════════════════════════════════════════════════════════════
# Frame Extraction Worker
# ═══════════════════════════════════════════════════════════════════════

class FrameExtractionWorker(QRunnable):
    """Extract frames from video files with optional inline resize."""

    def __init__(
        self, video_paths: list[str], output_dir: str, *,
        method: str, interval: int, target_fps: float,
        max_frames: int | None,
        resize_mode: str, resize_max_dim: int, resize_factor: int,
        output_format: str, jpg_quality: int,
    ) -> None:
        super().__init__()
        self.video_paths = video_paths; self.output_dir = output_dir
        self.method = method; self.interval = interval
        self.target_fps = target_fps; self.max_frames = max_frames
        self.resize_mode = resize_mode
        self.resize_max_dim = resize_max_dim
        self.resize_factor = resize_factor
        self.output_format = output_format; self.jpg_quality = jpg_quality
        self.signals = FrameExtractionSignals(); self._running = False

    @pyqtSlot()
    def run(self) -> None:
        from .pipeline_core import extract_frames
        self._running = True; self.signals.started.emit()
        try:
            all_paths = extract_frames(
                video_paths=self.video_paths, output_dir=self.output_dir,
                method=self.method, interval=self.interval,
                target_fps=self.target_fps, max_frames=self.max_frames,
                resize_mode=self.resize_mode, resize_max_dim=self.resize_max_dim,
                resize_factor=self.resize_factor, output_format=self.output_format,
                jpg_quality=self.jpg_quality,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
                cancel_check=lambda: not self._running,
            )
            if not self._running:
                logger.info("Frame extraction stopped by user")
                return
            self.signals.progress.emit(100, "Frame extraction complete!")
            self.signals.finished.emit(all_paths)
        except Exception as e:
            logger.exception("Frame extraction failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# Segmentation Worker
# ═══════════════════════════════════════════════════════════════════════

class SegmentationWorker(QRunnable):
    """Run segmentation; SAM3-vs-SkyWater dispatch and model auto-download
    live in ``pipeline_core.run_segmentation``."""

    def __init__(
        self,
        image_paths: list[str],
        mask_output_dir: str,
        model_config: dict,
        target_classes: list[str],
        confidence_threshold: float,
    ) -> None:
        super().__init__()
        self._image_paths = image_paths
        self._mask_output_dir = mask_output_dir
        self._model_config = model_config
        self._target_classes = target_classes
        self._confidence_threshold = confidence_threshold
        self.signals = SegmentationSignals(); self._running = False

    @pyqtSlot()
    def run(self) -> None:
        from .pipeline_core import run_segmentation
        self._running = True; self.signals.started.emit()
        try:
            mask_paths = run_segmentation(
                image_paths=self._image_paths,
                mask_output_dir=self._mask_output_dir,
                model_config=self._model_config,
                target_classes=self._target_classes,
                confidence_threshold=self._confidence_threshold,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
                image_done_cb=lambda ip, mp: self.signals.image_done.emit(ip, mp),
                cancel_check=lambda: not self._running,
            )
            if not self._running:
                logger.info("Segmentation stopped by user")
                return
            self.signals.finished.emit(mask_paths)
        except Exception as e:
            logger.exception("Segmentation failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# Database Build Worker
# ═══════════════════════════════════════════════════════════════════════

class DatabaseBuildWorker(QRunnable):
    """Build a COLMAP database from images."""

    def __init__(
        self, image_dir: str, db_path: str,
        camera_model_id: int, camera_params: list[float] | None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir; self.db_path = db_path
        self.camera_model_id = camera_model_id; self.camera_params = camera_params
        self.signals = DatabaseBuildSignals(); self._running = False

    @pyqtSlot()
    def run(self) -> None:
        from .pipeline_core import build_database
        self._running = True; self.signals.started.emit()
        try:
            build_database(
                image_dir=self.image_dir, db_path=self.db_path,
                camera_model_id=self.camera_model_id,
                camera_params=self.camera_params,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
            )
            if not self._running:
                logger.info("Database build stopped by user")
                return
            self.signals.finished.emit(self.db_path)
        except Exception as e:
            logger.exception("Database build failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False
