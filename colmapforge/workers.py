"""
QThread workers: frame extraction, SAM / SkyWater segmentation, DB creation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
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
        from .utils import analyze_video, compute_frame_indices, resize_image
        self._running = True; self.signals.started.emit()
        os.makedirs(self.output_dir, exist_ok=True)
        all_paths: list[str] = []

        try:
            total = len(self.video_paths)
            for vi, vp in enumerate(self.video_paths):
                if not self._running: break
                self.signals.progress.emit(int(vi / total * 100), f"Analyzing: {Path(vp).name}")
                info = analyze_video(vp)
                frames = compute_frame_indices(
                    info["total_frames"], info["fps"], method=self.method,
                    interval=self.interval, target_fps=self.target_fps,
                    start_seconds=self.start_seconds, end_seconds=self.end_seconds,
                    max_frames=self.max_frames)
                cap = cv2.VideoCapture(vp)
                try:
                    for fi, fn in enumerate(frames):
                        if not self._running: break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                        ret, frame = cap.read()
                        if not ret: continue
                        # inline resize
                        if self.resize_mode == "max_dim" and self.resize_max_dim:
                            frame = resize_image(frame, max_dim=self.resize_max_dim)
                        elif self.resize_mode == "downscale" and self.resize_factor > 1:
                            h, w = frame.shape[:2]
                            frame = resize_image(frame, width=w // self.resize_factor,
                                                  height=h // self.resize_factor,
                                                  keep_aspect=False)
                        name = f"{Path(vp).stem}_f{fn:06d}{self.output_format}"
                        out = os.path.join(self.output_dir, name)
                        params = []
                        if self.output_format.lower() in (".jpg", ".jpeg"):
                            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                        elif self.output_format.lower() == ".png":
                            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                        cv2.imwrite(out, frame, params); all_paths.append(out)
                        pct = int((vi + fi / len(frames)) / total * 100)
                        self.signals.progress.emit(pct,
                            f"Extracting: {Path(vp).name} ({fi + 1}/{len(frames)})")
                finally:
                    cap.release()
            self.signals.progress.emit(100, "Frame extraction complete!")
            self.signals.finished.emit(all_paths)
        except Exception as e:
            logger.exception("Frame extraction failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# SAM Segmentation Worker
# ═══════════════════════════════════════════════════════════════════════

# Long-side bound for mask accumulation: SAM3's image encoder resizes to its
# native 1008x1008 regardless, so this bounds decoder output + compositing
# without any encoder-side cost. Masks are upscaled back to the frame size.
_SAM_MAX_INFER_DIM = 1024


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
        self._running = True; self.signals.started.emit()
        os.makedirs(self.mask_output_dir, exist_ok=True)
        mask_paths: list[str] = []

        try:
            model = self._load_sam()
            total = len(self.image_paths)
            if total == 0:
                self.signals.finished.emit(mask_paths)
                return

            # Validate the model/prompt wiring once on the first image. A
            # systemic failure (bad shape, wrong input names) would otherwise
            # surface only after the whole batch silently produced blank masks.
            try:
                self._predict_sam(model, self.image_paths[0])
            except Exception as e:
                logger.exception("SAM validation failed on first image")
                self.signals.error.emit(
                    f"SAM failed on first image: {type(e).__name__}: {e}")
                return

            failed = 0
            for idx, img_path in enumerate(self.image_paths):
                if not self._running: break
                pct = int(idx / total * 100)
                name = Path(img_path).name
                self.signals.progress.emit(pct, f"SAM: {name} ({idx + 1}/{total})")
                try:
                    mask = self._predict_sam(model, img_path)
                    mp = self._save(img_path, mask); mask_paths.append(mp)
                    self.signals.image_done.emit(img_path, mp)
                except Exception as e:
                    # A blank mask is a *valid* result ("nothing matched"), so a
                    # swallowed per-image crash is indistinguishable from a
                    # working run that found nothing. Keep going; only the
                    # first image aborts (systemic failure, caught above).
                    failed += 1
                    logger.warning("SAM failed for %s: %s", img_path, e)
                    img = cv2.imread(img_path)
                    if img is not None:
                        mp = self._save(img_path, np.zeros(img.shape[:2], dtype=np.uint8))
                        mask_paths.append(mp)

            self.signals.progress.emit(
                100, f"SAM segmentation complete ({failed}/{total} images failed)"
                if failed else "SAM segmentation complete!")
            self.signals.finished.emit(mask_paths)
        except Exception as e:
            logger.exception("SAM worker failed"); self.signals.error.emit(str(e))

    def _load_sam(self):
        """Build the SAM3 model, refusing configs without a language encoder.

        Without the language encoder the decoder runs on zero-filled language
        tensors: it still emits masks, so prompts would look "ignored" rather
        than failing. Refuse up front instead.
        """
        from .sam_backends.sam3_onnx import SegmentAnything3ONNX
        dp = self.model_config["decoder_model_path"]
        lang_path = self.model_config.get("language_encoder_path")
        if not lang_path or not os.path.isfile(lang_path):
            raise RuntimeError(
                "SAM3 model has no language encoder — text prompts cannot "
                "be applied.\n"
                f"Looked for it in: {self.model_config.get('config_file') or dp}\n"
                "Re-download the model, or pick a different one.")
        return SegmentAnything3ONNX(
            image_encoder_path=self.model_config["encoder_model_path"],
            decoder_model_path=dp,
            language_encoder_path=lang_path)

    def _predict_sam(self, model, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None: raise RuntimeError(f"Cannot read: {img_path}")
        orig_h, orig_w = img.shape[:2]

        # The image encoder always resizes to its 1008x1008 native size, so
        # downscaling the input here costs nothing on the encoder side but
        # keeps the decoder output and mask accumulation bounded on large
        # frames. Final mask is upscaled back to the original size.
        h, w = orig_h, orig_w
        if max(h, w) > _SAM_MAX_INFER_DIM:
            scale = _SAM_MAX_INFER_DIM / max(h, w)
            h, w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        combined = np.zeros((h, w), dtype=np.uint8)

        for ci, cls_name in enumerate(self.target_classes):
            if not self._running: break
            if ci == 0:
                embedding = model.encode(img_rgb, text_prompt=cls_name)
            else:
                embedding = model.update_language(embedding, cls_name)
            masks = model.predict_masks(
                embedding, prompt=[],
                confidence_threshold=self.confidence_threshold)
            for m in masks:
                if m.ndim == 3 and m.shape[0] == 1: m = m[0]
                combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

        if (h, w) != (orig_h, orig_w):
            combined = cv2.resize(
                combined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        logger.debug("SAM3 %s: %d classes, mask %.1f%% non-zero",
                     Path(img_path).name, len(self.target_classes),
                     100 * np.count_nonzero(combined) / combined.size)
        return combined

    def _save(self, img_path: str, mask: np.ndarray) -> str:
        out = os.path.join(self.mask_output_dir, f"{Path(img_path).stem}_mask.png")
        cv2.imwrite(out, mask); return out

    def stop(self) -> None: self._running = False


# ═══════════════════════════════════════════════════════════════════════
# SkyWater Segmentation Worker (SegFormer-B2, 4-class: Sky/Water/Person)
# ═══════════════════════════════════════════════════════════════════════

SW_INPUT_SIZE = (384, 384)
SW_CLASSES = ["Background", "Sky", "Water", "Person"]
SW_CLASS_IDS = {name.lower(): i for i, name in enumerate(SW_CLASSES)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        self._running = True; self.signals.started.emit()
        os.makedirs(self.mask_output_dir, exist_ok=True)
        mask_paths: list[str] = []

        try:
            from .onnx_utils import create_inference_session
            target_ids = set()
            for cls_name in self.target_classes:
                cid = SW_CLASS_IDS.get(cls_name.lower())
                if cid is not None: target_ids.add(cid)
            # always map: sky, water, person → if user selects any, include it
            if not target_ids:
                target_ids = {1, 2, 3}  # default: mask sky+water+person

            # GPU-accelerated session (CUDA → CoreML → CPU fallback).
            sess = create_inference_session(self.model_path)
            providers_used = sess.get_providers()
            logger.info("SkyWater session providers: %s", providers_used)
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            total = len(self.image_paths)

            for idx, img_path in enumerate(self.image_paths):
                if not self._running: break
                pct = int(idx / total * 100)
                name = Path(img_path).name
                self.signals.progress.emit(pct, f"SkyWater: {name} ({idx + 1}/{total})")

                img = cv2.imread(img_path)
                if img is None: continue
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # preprocess
                inp = cv2.resize(img_rgb, SW_INPUT_SIZE).astype(np.float32) / 255.0
                inp = (inp - IMAGENET_MEAN) / IMAGENET_STD
                inp = inp.transpose(2, 0, 1)[np.newaxis, ...]

                # inference
                logits = sess.run([out_name], {in_name: inp})[0]  # (1,4,384,384)
                mask_small = np.argmax(logits[0], axis=0).astype(np.uint8)  # (384,384)
                # resize back + filter to target classes
                mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
                final = np.zeros((h, w), dtype=np.uint8)
                for cid in target_ids:
                    final[mask_full == cid] = 255

                mp = os.path.join(self.mask_output_dir, f"{Path(img_path).stem}_mask.png")
                cv2.imwrite(mp, final); mask_paths.append(mp)
                self.signals.image_done.emit(img_path, mp)

            self.signals.progress.emit(100, "SkyWater segmentation complete!")
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

        if is_skywater:
            logical_name = model_config.get("logical_name", "skywater_segformer_b2_fp16")
            if not model_path or not os.path.isfile(model_path):
                try:
                    from .model_downloader import download_model
                    self.signals.progress.emit(
                        0, "Downloading SkyWater model from HuggingFace..."
                    )

                    model_path = download_model(logical_name, progress_callback=_on_progress)
                    # Persist back into config so re-builds skip the download check
                    model_config["model_path"] = model_path
                    model_config["encoder_model_path"] = model_path
                    model_config["decoder_model_path"] = model_path
                except Exception as e:
                    logger.exception("SkyWater model download failed")
                    self.signals.error.emit(f"SkyWater download failed: {e}")
                    return

            self._worker = SkyWaterWorker(
                image_paths=self._image_paths,
                mask_output_dir=self._mask_output_dir,
                model_path=model_path,
                target_classes=self._target_classes,
            )
        else:
            # ── SAM: auto-download zip from HuggingFace on first use ──
            has_downloaded = model_config.get("has_downloaded", True)
            decoder_path = model_config.get("decoder_model_path", "")
            logical_name = model_config.get("logical_name", "")
            if (not has_downloaded or not decoder_path or not os.path.isfile(decoder_path)) and logical_name:
                try:
                    from .model_downloader import download_zip_model
                    self.signals.progress.emit(0, f"Downloading SAM model: {logical_name}...")
                    updated = download_zip_model(logical_name, progress_callback=_on_progress)
                    model_config.update(updated)
                except Exception as e:
                    logger.exception("SAM model download failed")
                    self.signals.error.emit(f"SAM model download failed: {e}")
                    return

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
        from .camera_models import get_camera_model
        from .colmap_database import ColmapDatabase
        from .utils import collect_image_files
        self._running = True; self.signals.started.emit()
        try:
            self.signals.progress.emit(10, "Collecting images...")
            paths = collect_image_files([self.image_dir], recursive=False)
            if not paths: raise FileNotFoundError(f"No images in {self.image_dir}")
            self.signals.progress.emit(30, f"Found {len(paths)} images")
            model = get_camera_model(self.camera_model_id)
            self.signals.progress.emit(50, "Building database...")
            db = ColmapDatabase(self.db_path)
            db.reset()  # rebuild → fresh DB (avoids UNIQUE collisions on images.name)
            with db:
                r = db.build_project(self.image_dir, model, self.camera_params, self.prior_focal_length)
            self.signals.progress.emit(100, f"Ready: {r['image_count']} images, {model.name}")
            self.signals.finished.emit(self.db_path)
        except Exception as e:
            logger.exception("Database build failed"); self.signals.error.emit(str(e))

    def stop(self) -> None: self._running = False
