"""
QThread workers: frame extraction, SAM / SkyWater segmentation, DB creation.
"""

from __future__ import annotations

import logging
import os

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
        self, video_paths: list[str], output_dir: str,
        method: str = "interval", interval: int = 1,
        target_fps: float = 1.0, start_seconds: float = 0.0,
        end_seconds: float | None = None, max_frames: int | None = None,
        resize_mode: str = "", resize_max_dim: int = 0,
        resize_factor: int = 1,
        output_format: str = ".jpg", jpg_quality: int = 95,
    ) -> None:
        super().__init__()
        self.video_paths = video_paths; self.output_dir = output_dir
        self.method = method; self.interval = interval
        self.target_fps = target_fps; self.start_seconds = start_seconds
        self.end_seconds = end_seconds; self.max_frames = max_frames
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
                target_fps=self.target_fps, start_seconds=self.start_seconds,
                end_seconds=self.end_seconds, max_frames=self.max_frames,
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
# SAM Segmentation Worker
# ═══════════════════════════════════════════════════════════════════════

class SAMWorker(QRunnable):
    """Run SAM3 text-prompt segmentation on a directory of images.

    SAM3 is the only supported SAM family: its decoder has a native language
    encoder, so class names are passed straight to it as text prompts.
    """

    def __init__(
        self, image_paths: list[str], mask_output_dir: str,
        model_config: dict, target_classes: list[str],
        confidence_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths; self.mask_output_dir = mask_output_dir
        self.model_config = model_config; self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        self.signals = SegmentationSignals(); self._running = False

    @pyqtSlot()
    def run(self) -> None:
        from .pipeline_core import run_sam_segmentation
        self._running = True; self.signals.started.emit()
        os.makedirs(self.mask_output_dir, exist_ok=True)

        try:
            mask_paths = run_sam_segmentation(
                image_paths=self.image_paths,
                mask_output_dir=self.mask_output_dir,
                model_config=self.model_config,
                target_classes=self.target_classes,
                confidence_threshold=self.confidence_threshold,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
                image_done_cb=lambda ip, mp: self.signals.image_done.emit(ip, mp),
                cancel_check=lambda: not self._running,
            )
            if not self._running:
                logger.info("SAM segmentation stopped by user")
                return
            self.signals.finished.emit(mask_paths)
        except Exception as e:
            logger.exception("SAM worker failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# SkyWater Segmentation Worker (SegFormer-B2, 4-class: Sky/Water/Person)
# ═══════════════════════════════════════════════════════════════════════

class SkyWaterWorker(QRunnable):
    """Run SkyWater SegFormer ONNX model for fast sky/water/person segmentation."""

    def __init__(
        self, image_paths: list[str], mask_output_dir: str,
        model_path: str, target_classes: list[str],
    ) -> None:
        super().__init__()
        self.image_paths = image_paths; self.mask_output_dir = mask_output_dir
        self.model_path = model_path; self.target_classes = target_classes
        self.signals = SegmentationSignals(); self._running = False

    @pyqtSlot()
    def run(self) -> None:
        from .pipeline_core import run_skywater_segmentation
        self._running = True; self.signals.started.emit()
        os.makedirs(self.mask_output_dir, exist_ok=True)

        try:
            mask_paths = run_skywater_segmentation(
                image_paths=self.image_paths,
                mask_output_dir=self.mask_output_dir,
                model_path=self.model_path,
                target_classes=self.target_classes,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
                image_done_cb=lambda ip, mp: self.signals.image_done.emit(ip, mp),
                cancel_check=lambda: not self._running,
            )
            if not self._running:
                logger.info("SkyWater segmentation stopped by user")
                return
            self.signals.finished.emit(mask_paths)
        except Exception as e:
            logger.exception("SkyWater worker failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# Unified dispatcher
# ═══════════════════════════════════════════════════════════════════════

class SegmentationWorker(QRunnable):
    """Dispatches to SAM or SkyWater worker based on model config."""

    def __init__(
        self,
        image_paths: list[str],
        mask_output_dir: str,
        model_config: dict,
        target_classes: list[str],
        confidence_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self._image_paths = image_paths
        self._mask_output_dir = mask_output_dir
        self._model_config = model_config
        self._target_classes = target_classes
        self._confidence_threshold = confidence_threshold
        self.signals = SegmentationSignals()
        self._worker = None; self._running = False

    @pyqtSlot()
    def run(self) -> None:
        self._running = True; self.signals.started.emit()

        model_config = self._model_config
        model_path = model_config.get("model_path", "")

        is_skywater = (
            "skywater" in model_config.get("type", "").lower() or
            "skywater" in model_config.get("name", "").lower() or
            "skywater" in model_path.lower()
        )

        # ── SkyWater: auto-download from HuggingFace if not yet cached ──
        # model_path may be None when the user picked SkyWater before the
        # first download. Resolve via the model registry + HF cache.
        def _on_progress(msg: str, done: int, total: int) -> None:
            pct = 0 if total <= 0 else int(100 * done / total)
            self.signals.progress.emit(pct, msg)

        from .model_downloader import download_model_entry
        logical_name = model_config.get("logical_name", "")
        need_download = False
        if is_skywater:
            # SkyWater: model_path may be None before the first download.
            need_download = not model_path or not os.path.isfile(model_path)
        else:
            # SAM: download when the zip hasn't been fetched yet.
            decoder_path = model_config.get("decoder_model_path", "")
            need_download = (
                not model_config.get("has_downloaded", True)
                or not decoder_path or not os.path.isfile(decoder_path)
            )

        if need_download and logical_name:
            try:
                self.signals.progress.emit(0, f"Downloading model: {logical_name}...")
                updated = download_model_entry(logical_name, progress_callback=_on_progress)
                model_config.update(updated)
            except Exception as e:
                logger.exception("Model download failed for %s", logical_name)
                self.signals.error.emit(f"Model download failed: {e}")
                return

        if not self._running:
            # Stopped (e.g. during the model download above) — do not start
            # the inner worker, and do not signal finished.
            logger.info("Segmentation stopped by user")
            return

        if is_skywater:
            model_path = model_config.get("model_path") or model_config.get("decoder_model_path")
            self._worker = SkyWaterWorker(
                image_paths=self._image_paths,
                mask_output_dir=self._mask_output_dir,
                model_path=model_path,
                target_classes=self._target_classes,
            )
        else:
            self._worker = SAMWorker(
                image_paths=self._image_paths,
                mask_output_dir=self._mask_output_dir,
                model_config=model_config,
                target_classes=self._target_classes,
                confidence_threshold=self._confidence_threshold,
            )

        self._worker.signals = self.signals
        self._worker.run()

    def stop(self) -> None:
        self._running = False
        if self._worker: self._worker.stop()


# ═══════════════════════════════════════════════════════════════════════
# Database Build Worker
# ═══════════════════════════════════════════════════════════════════════

class DatabaseBuildWorker(QRunnable):
    """Build a COLMAP database from images."""

    def __init__(
        self, image_dir: str, db_path: str,
        camera_model_id: int, camera_params: list[float] | None = None,
        prior_focal_length: int = 0,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir; self.db_path = db_path
        self.camera_model_id = camera_model_id; self.camera_params = camera_params
        self.prior_focal_length = prior_focal_length
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
                prior_focal_length=self.prior_focal_length,
                progress_cb=lambda pct, msg: self.signals.progress.emit(pct, msg),
            )
            if not self._running:
                logger.info("Database build stopped by user")
                return
            self.signals.finished.emit(self.db_path)
        except Exception as e:
            logger.exception("Database build failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False
