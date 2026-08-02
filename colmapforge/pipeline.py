"""
Pipeline orchestrator — non-visual QObject that chains workers.

Receives a flat PipelineConfig, runs the extraction→resize→segmentation→database
chain, and emits progress/result signals back to the UI layer.
"""

from __future__ import annotations

import logging
import os
import shutil

from PyQt6.QtCore import QObject, QThreadPool, pyqtSignal

from .pipeline_core import (
    PipelineConfig,
    apply_resize,
    copy_input_images,
    is_skywater_config,
    output_layout,
)
from .utils import collect_image_files
from .workers import (
    DatabaseBuildWorker, FrameExtractionWorker, SegmentationWorker,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator(QObject):
    """Coordinates the extraction → segmentation → database build chain."""

    progress = pyqtSignal(int, str)           # pct, message
    mask_ready = pyqtSignal(str, str)         # image_path, mask_path
    pipeline_finished = pyqtSignal(str)       # db_path
    preview_switch = pyqtSignal(str)          # images_dir (to refresh preview)
    error = pyqtSignal(str)                   # error message

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread_pool = QThreadPool()
        self._active_worker = None
        self._phase = ""
        self._last_resize_cfg = None
        self._config: PipelineConfig | None = None
        self._preview_primed = False
        self._last_preview_decile = -1

    # ── public API ──

    def run(self, config: PipelineConfig) -> None:
        """Validate and start the pipeline."""
        self._config = config
        self._preview_primed = False  # reset for fresh extraction runs
        self._last_preview_decile = -1

        if not config.output_dir:
            self.error.emit("Select an output directory first.")
            return

        images_dir, masks_dir, db_path = output_layout(config.output_dir)
        images_exist = os.path.isdir(images_dir) and collect_image_files([images_dir])

        if images_exist and config.resize_signature() == self._last_resize_cfg:
            # Re-build with unchanged resize config: re-run seg/db only.
            if config.seg_enabled and not config.seg_target_classes:
                self.error.emit("Segmentation enabled but no target classes selected.")
                return
            self._reset_masks_and_db(masks_dir, db_path)
            self._phase = "re-segmentation"
            self.preview_switch.emit(images_dir)
            self._run_seg_or_db(images_dir, masks_dir, db_path)
            return

        # Fresh run, or resize config changed → rebuild images from inputs.
        if not config.video_paths and not config.image_paths:
            self.error.emit("Add videos or images first.")
            return
        if images_exist:
            shutil.rmtree(images_dir)
            self._reset_masks_and_db(masks_dir, db_path)
        os.makedirs(images_dir, exist_ok=True)
        copy_input_images(config.image_paths, images_dir)
        if config.video_paths:
            self._run_extraction(images_dir, masks_dir, db_path)
        else:
            apply_resize(images_dir, **config.resize_kwargs())
            self.preview_switch.emit(images_dir)
            self._run_seg_or_db(images_dir, masks_dir, db_path)

    def cancel(self) -> None:
        if self._active_worker and hasattr(self._active_worker, "stop"):
            self._active_worker.stop()

    # ── helpers ──

    @staticmethod
    def _reset_masks_and_db(masks_dir: str, db_path: str) -> None:
        if os.path.isdir(masks_dir):
            shutil.rmtree(masks_dir)
        os.makedirs(masks_dir, exist_ok=True)
        if os.path.isfile(db_path):
            os.remove(db_path)

    # ── stage: extraction ──

    def _run_extraction(self, images_dir, masks_dir, db_path) -> None:
        cfg = self._config
        self._phase = "extraction"
        worker = FrameExtractionWorker(
            video_paths=cfg.video_paths, output_dir=images_dir,
            method=cfg.extract_method, interval=cfg.extract_interval,
            target_fps=cfg.extract_target_fps,
            max_frames=cfg.extract_max_frames,
            **cfg.resize_kwargs(),
            output_format=cfg.extract_format,
            jpg_quality=cfg.extract_jpg_quality,
        )
        self._active_worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda p: self._on_extract_done(p, images_dir, masks_dir, db_path))
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_extract_done(self, paths, images_dir, masks_dir, db_path) -> None:
        self.preview_switch.emit(images_dir)
        self._run_seg_or_db(images_dir, masks_dir, db_path)

    # ── stage: segmentation ──

    def _run_seg_or_db(self, images_dir, masks_dir, db_path) -> None:
        cfg = self._config
        if cfg.seg_enabled and cfg.seg_target_classes:
            model_cfg = cfg.seg_model_config
            if not model_cfg:
                self._on_error("No segmentation model selected"); return

            resolved = dict(model_cfg)

            if not is_skywater_config(model_cfg):
                dp = model_cfg.get("decoder_model_path", "")
                # Allow not-yet-downloaded models — the worker downloads them.
                if model_cfg.get("has_downloaded") is not False:
                    if not dp or not os.path.isfile(dp):
                        self._on_error(f"Model file not found: {dp}"); return
            else:
                sw_path = model_cfg.get("model_path")
                if not sw_path or not os.path.isfile(sw_path):
                    info_name = model_cfg.get("logical_name", "skywater_segformer_b2_fp16")
                    from .model_downloader import get_model_info
                    meta = get_model_info(info_name) or {}
                    size_hint = meta.get("size_hint_mb", "?")
                    # Delegate the confirmation dialog to the UI layer
                    reply = self._confirm_skywater_download(info_name, size_hint)
                    if not reply:
                        self._on_error("SkyWater download cancelled by user"); return

            self._phase = "segmentation"
            worker = SegmentationWorker(
                image_paths=collect_image_files([images_dir]), mask_output_dir=masks_dir,
                model_config=resolved, target_classes=cfg.seg_target_classes,
                confidence_threshold=cfg.seg_confidence)
            self._active_worker = worker
            worker.signals.progress.connect(self._on_progress)
            worker.signals.image_done.connect(self._on_mask_ready)
            worker.signals.finished.connect(lambda p: self._on_seg_done(p, images_dir, db_path))
            worker.signals.error.connect(self._on_error)
            self._thread_pool.start(worker)
        else:
            self._run_db(images_dir, db_path)

    def _confirm_skywater_download(self, logical_name: str, size_hint: str) -> bool:
        """Override point for SkyWater download confirmation.

        By default returns True (auto-download). The MainWindow subclass or
        widget can connect to a hook here. For now we emit a signal and let
        the caller decide before calling run().
        """
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            None, "Download SkyWater Model",
            f"The SkyWater SegFormer-B2 model (~{size_hint} MB) will be "
            "downloaded from HuggingFace (Realcat/skywater_seg) and cached "
            "for future use.\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_mask_ready(self, img_path: str, mask_path: str) -> None:
        self.mask_ready.emit(img_path, mask_path)

    def _on_seg_done(self, paths, images_dir, db_path) -> None:
        self._run_db(images_dir, db_path)

    # ── stage: database ──

    def _run_db(self, images_dir, db_path) -> None:
        cfg = self._config
        self._phase = "database"
        worker = DatabaseBuildWorker(
            image_dir=images_dir, db_path=db_path,
            camera_model_id=cfg.camera_model_id, camera_params=cfg.camera_params,
        )
        self._active_worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_db_done)
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_db_done(self, db_path: str) -> None:
        self._last_resize_cfg = self._config.resize_signature()
        self.pipeline_finished.emit(db_path)

    # ── shared signals ──

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.emit(pct, msg)
        if self._phase == "extraction" and self._config:
            # Refresh the preview whenever we cross a 10 % boundary so the
            # user sees frames accumulating. Also fire at the very first
            # frame (pct > 0) so the preview switches away from the initial
            # placeholder as soon as images exist.
            decile = pct // 10
            if pct > 0 and (decile != self._last_preview_decile or not self._preview_primed):
                self._preview_primed = True
                self._last_preview_decile = decile
                images_dir = output_layout(self._config.output_dir)[0]
                if os.path.isdir(images_dir):
                    self.preview_switch.emit(images_dir)

    def _on_error(self, msg: str) -> None:
        self.error.emit(f"Pipeline failed ({self._phase}):\n{msg}")
