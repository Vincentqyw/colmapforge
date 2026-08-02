"""
Pure-python pipeline functions — no Qt dependency.

Each function accepts a ``progress_cb(pct: int, message: str)`` callback
and an optional ``cancel_event(→ bool)`` for cooperative cancellation.
These are the computational core; the Qt workers in ``workers.py`` wrap
them with signal emissions, and the CLI runner in ``cli.py`` calls them
directly.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera_models import DEFAULT_CAMERA_MODEL_ID, get_camera_model
from .colmap_database import ColmapDatabase
from .utils import (
    analyze_video,
    collect_image_files,
    compute_frame_indices,
    resize_image,
)

logger = logging.getLogger(__name__)

# ── Type aliases ───────────────────────────────────────────────────────

ProgressCB = Callable[[int, str], None] | None
ImageDoneCB = Callable[[str, str], None] | None
CancelCheck = Callable[[], bool] | None

# Long-side bound for SAM3 inference.
_SAM_MAX_INFER_DIM = 1024


# ═══════════════════════════════════════════════════════════════════════
# Configuration & shared conventions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """All parameters needed to run the full preprocessor pipeline."""
    video_paths: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    output_dir: str = ""

    extract_enabled: bool = True
    extract_method: str = "interval"
    extract_interval: int = 60
    extract_target_fps: float = 2.0
    extract_max_frames: int | None = None
    extract_format: str = ".jpg"
    extract_jpg_quality: int = 95

    resize_enabled: bool = False
    resize_mode: str = "downscale"
    resize_max_dim: int = 2000
    resize_factor: int = 4

    seg_enabled: bool = False
    seg_model_config: dict | None = None
    seg_target_classes: list[str] = field(default_factory=list)
    seg_confidence: float = 0.3

    camera_model_id: int = DEFAULT_CAMERA_MODEL_ID
    camera_params: list[float] = field(default_factory=list)

    def resize_kwargs(self) -> dict:
        """``extract_frames``/``apply_resize`` kwargs honoring ``resize_enabled``."""
        mode = self.resize_mode if self.resize_enabled else ""
        return {
            "resize_mode": mode,
            "resize_max_dim": self.resize_max_dim if mode == "max_dim" else 0,
            "resize_factor": self.resize_factor if mode == "downscale" else 1,
        }

    def resize_signature(self):
        """Hashable resize identity for change detection (None when disabled)."""
        if not self.resize_enabled:
            return None
        if self.resize_mode == "max_dim":
            return ("max_dim", self.resize_max_dim)
        if self.resize_mode == "downscale":
            return ("downscale", self.resize_factor)
        return (self.resize_mode,)


MASK_SUFFIX = "_mask.png"


def output_layout(output_dir: str) -> tuple[str, str, str]:
    """Canonical output layout: ``(images_dir, masks_dir, db_path)``."""
    return (
        os.path.join(output_dir, "images"),
        os.path.join(output_dir, "masks"),
        os.path.join(output_dir, "database.db"),
    )


def mask_path_for(image_path: str, mask_output_dir: str) -> str:
    """Mask path convention shared by the mask writers and the preview reader."""
    return os.path.join(mask_output_dir, f"{Path(image_path).stem}{MASK_SUFFIX}")


def is_skywater_config(model_config: dict) -> bool:
    """True when *model_config* refers to a SkyWater (SegFormer) model."""
    return (
        "skywater" in model_config.get("type", "").lower()
        or "skywater" in model_config.get("name", "").lower()
        or "skywater" in (model_config.get("model_path") or "").lower()
    )


def is_yoloworld_sam_config(model_config: dict) -> bool:
    """True when *model_config* refers to a YOLO-World + SAM1/2 cascade."""
    return model_config.get("type", "") == "yoloworld_sam"


# ═══════════════════════════════════════════════════════════════════════
# Frame Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_frames(
    video_paths: list[str],
    output_dir: str,
    *,
    method: str = "interval",
    interval: int = 60,
    target_fps: float = 2.0,
    max_frames: int | None = None,
    resize_mode: str = "",
    resize_max_dim: int = 0,
    resize_factor: int = 1,
    output_format: str = ".jpg",
    jpg_quality: int = 95,
    progress_cb: ProgressCB = None,
    cancel_check: CancelCheck = None,
) -> list[str]:
    """Extract frames from *video_paths* into *output_dir*.

    Returns the list of output image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_paths: list[str] = []
    total = len(video_paths)

    for vi, vp in enumerate(video_paths):
        if cancel_check and cancel_check():
            logger.info("Frame extraction cancelled")
            return all_paths

        if progress_cb:
            progress_cb(int(vi / total * 100), f"Analyzing: {Path(vp).name}")

        info = analyze_video(vp)
        frames = compute_frame_indices(
            info["total_frames"], info["fps"],
            method=method, interval=interval, target_fps=target_fps,
            max_frames=max_frames,
        )

        cap = cv2.VideoCapture(vp)
        try:
            for fi, fn in enumerate(frames):
                if cancel_check and cancel_check():
                    return all_paths

                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Inline resize
                if resize_mode == "max_dim" and resize_max_dim:
                    frame = resize_image(frame, max_dim=resize_max_dim)
                elif resize_mode == "downscale" and resize_factor > 1:
                    h, w = frame.shape[:2]
                    frame = resize_image(
                        frame, width=w // resize_factor,
                        height=h // resize_factor, keep_aspect=False,
                    )

                name = f"{Path(vp).stem}_f{fn:06d}{output_format}"
                out = os.path.join(output_dir, name)
                params: list[int] = []
                if output_format.lower() in (".jpg", ".jpeg"):
                    params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
                elif output_format.lower() == ".png":
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                cv2.imwrite(out, frame, params)
                all_paths.append(out)

                if progress_cb:
                    pct = int((vi + fi / len(frames)) / total * 100)
                    progress_cb(pct,
                        f"Extracting: {Path(vp).name} ({fi + 1}/{len(frames)})")
        finally:
            cap.release()

    if progress_cb:
        progress_cb(100, "Frame extraction complete!")
    return all_paths


# ═══════════════════════════════════════════════════════════════════════
# Segmentation — SAM3
# ═══════════════════════════════════════════════════════════════════════

def _load_sam3_model(model_config: dict):
    """Build a SAM3 model from *model_config*, raising early if the language
    encoder is missing (text prompts would silently produce junk otherwise)."""
    from .sam_backends.sam3_onnx import SegmentAnything3ONNX

    dp = model_config["decoder_model_path"]
    lang_path = model_config.get("language_encoder_path")
    if not lang_path or not os.path.isfile(lang_path):
        raise RuntimeError(
            "SAM3 model has no language encoder — text prompts cannot "
            "be applied.\n"
            f"Looked for it in: {model_config.get('config_file') or dp}\n"
            "Re-download the model, or pick a different one.")
    return SegmentAnything3ONNX(
        image_encoder_path=model_config["encoder_model_path"],
        decoder_model_path=dp,
        language_encoder_path=lang_path,
    )


def _predict_sam3_mask(
    model, img_path: str, target_classes: list[str],
    confidence_threshold: float, cancel_check: CancelCheck,
) -> np.ndarray:
    """Run SAM3 on a single image and return a combined uint8 mask."""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Cannot read: {img_path}")
    orig_h, orig_w = img.shape[:2]

    h, w = orig_h, orig_w
    if max(h, w) > _SAM_MAX_INFER_DIM:
        scale = _SAM_MAX_INFER_DIM / max(h, w)
        h, w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    combined = np.zeros((h, w), dtype=np.uint8)

    for ci, cls_name in enumerate(target_classes):
        if cancel_check and cancel_check():
            break
        if ci == 0:
            embedding = model.encode(img_rgb, text_prompt=cls_name)
        else:
            embedding = model.update_language(embedding, cls_name)
        masks = model.predict_masks(
            embedding, confidence_threshold=confidence_threshold)
        for m in masks:
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

    if (h, w) != (orig_h, orig_w):
        combined = cv2.resize(
            combined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    logger.debug("SAM3 %s: %d classes, mask %.1f%% non-zero",
                 Path(img_path).name, len(target_classes),
                 100 * np.count_nonzero(combined) / combined.size)
    return combined


def run_sam_segmentation(
    image_paths: list[str],
    mask_output_dir: str,
    model_config: dict,
    target_classes: list[str],
    confidence_threshold: float = 0.3,
    *,
    progress_cb: ProgressCB = None,
    image_done_cb: ImageDoneCB = None,
    cancel_check: CancelCheck = None,
) -> list[str]:
    """Run SAM3 segmentation on *image_paths*, writing masks to *mask_output_dir*.

    Returns the list of mask image paths.
    """
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_paths: list[str] = []
    total = len(image_paths)

    if total == 0:
        return mask_paths

    model = _load_sam3_model(model_config)

    failed = 0
    for idx, img_path in enumerate(image_paths):
        if cancel_check and cancel_check():
            logger.info("SAM3 segmentation cancelled")
            return mask_paths

        pct = int(idx / total * 100)
        name = Path(img_path).name
        if progress_cb:
            progress_cb(pct, f"SAM: {name} ({idx + 1}/{total})")

        try:
            mask = _predict_sam3_mask(
                model, img_path, target_classes, confidence_threshold, cancel_check)
        except Exception as e:
            if idx == 0:
                # A failure on the very first image means the setup itself is
                # broken (model/providers) — abort instead of writing empty
                # masks for the whole set.
                logger.exception("SAM3 failed on first image")
                raise RuntimeError(
                    f"SAM3 failed on first image: {type(e).__name__}: {e}") from e
            failed += 1
            logger.warning("SAM3 failed for %s: %s", img_path, e)
            img = cv2.imread(img_path)
            if img is None:
                continue
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        mp = mask_path_for(img_path, mask_output_dir)
        cv2.imwrite(mp, mask)
        mask_paths.append(mp)
        if image_done_cb:
            image_done_cb(img_path, mp)

    if progress_cb:
        progress_cb(100,
            f"SAM segmentation complete ({failed}/{total} images failed)"
            if failed else "SAM segmentation complete!")
    return mask_paths


# ═══════════════════════════════════════════════════════════════════════
# Segmentation — YOLO-World + SAM1/2 cascade
# ═══════════════════════════════════════════════════════════════════════

def _load_box_sam_model(model_config: dict):
    """Build a box-promptable SAM from *model_config*.

    EfficientViT-SAM is declared explicitly (``sam_backend`` key from the
    registry); SAM1 vs SAM2 is auto-detected from the decoder's ONNX graph
    inputs (SAM2 decoders take ``high_res_feats_0``; SAM1 decoders do not).
    """
    dp = model_config["decoder_model_path"]

    if model_config.get("sam_backend") == "efficientvit":
        from .sam_backends.efficientvit_sam_onnx import EfficientViTSAMONNX
        return EfficientViTSAMONNX(
            encoder_model_path=model_config["encoder_model_path"],
            decoder_model_path=dp)
    if model_config.get("sam_backend") == "edgetam":
        from .sam_backends.edgetam_onnx import EdgeTAMONNX
        return EdgeTAMONNX(
            encoder_model_path=model_config["encoder_model_path"],
            decoder_model_path=dp)

    import onnx

    inputs = {i.name for i in onnx.load(dp).graph.input}
    if "high_res_feats_0" in inputs:
        from .sam_backends.sam2_onnx import SegmentAnything2ONNX
        return SegmentAnything2ONNX(
            encoder_model_path=model_config["encoder_model_path"],
            decoder_model_path=dp)
    from .sam_backends.sam_onnx import SegmentAnythingONNX
    return SegmentAnythingONNX(
        encoder_model_path=model_config["encoder_model_path"],
        decoder_model_path=dp)


def _predict_yoloworld_sam_mask(
    detector, sam, img_path: str, target_classes: list[str],
    confidence_threshold: float, cancel_check: CancelCheck,
) -> np.ndarray:
    """Detect *target_classes* with YOLO-World, mask each box with SAM,
    and return the combined uint8 mask."""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Cannot read: {img_path}")
    h, w = img.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect(
        img_rgb, target_classes, score_threshold=confidence_threshold)
    if not detections:
        return combined

    boxes = [
        [float(v) for v in box] for box, _cls, _score in detections
        if box[2] - box[0] >= 1 and box[3] - box[1] >= 1
    ]
    if not boxes:
        return combined

    embedding = sam.encode(img)  # per-image; the decoder runs per box
    if hasattr(sam, "predict_boxes"):
        # Batched fast path (EfficientViT-SAM): one decoder call for all boxes.
        for m in sam.predict_boxes(embedding, boxes):
            combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)
    else:
        for box in boxes:
            if cancel_check and cancel_check():
                break
            masks = np.asarray(sam.predict_masks(
                embedding, [{"type": "rectangle", "data": box}]))
            for m in masks.reshape(-1, *masks.shape[-2:]):
                combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

    logger.debug("YOLO-World+SAM %s: %d boxes, mask %.1f%% non-zero",
                 Path(img_path).name, len(detections),
                 100 * np.count_nonzero(combined) / combined.size)
    return combined


def run_yoloworld_sam_segmentation(
    image_paths: list[str],
    mask_output_dir: str,
    model_config: dict,
    target_classes: list[str],
    confidence_threshold: float = 0.3,
    *,
    progress_cb: ProgressCB = None,
    image_done_cb: ImageDoneCB = None,
    cancel_check: CancelCheck = None,
) -> list[str]:
    """Run the YOLO-World → SAM1/2 cascade on *image_paths*.

    Class names drive YOLO-World's open-vocabulary detector; each detected
    box prompts SAM for a mask, and the per-image union is written to
    *mask_output_dir*. Returns the list of mask paths.
    """
    from .sam_backends.yoloworld_onnx import YoloWorldONNX

    os.makedirs(mask_output_dir, exist_ok=True)
    mask_paths: list[str] = []
    total = len(image_paths)

    if total == 0:
        return mask_paths

    detector = YoloWorldONNX(
        model_config["yoloworld_model_path"], model_config["text_encoder_path"])
    sam = _load_box_sam_model(model_config)

    failed = 0
    for idx, img_path in enumerate(image_paths):
        if cancel_check and cancel_check():
            logger.info("YOLO-World+SAM segmentation cancelled")
            return mask_paths

        pct = int(idx / total * 100)
        name = Path(img_path).name
        if progress_cb:
            progress_cb(pct, f"YOLO-World+SAM: {name} ({idx + 1}/{total})")

        try:
            mask = _predict_yoloworld_sam_mask(
                detector, sam, img_path, target_classes,
                confidence_threshold, cancel_check)
        except Exception as e:
            if idx == 0:
                # A failure on the very first image means the setup itself is
                # broken (model/providers) — abort instead of writing empty
                # masks for the whole set.
                logger.exception("YOLO-World+SAM failed on first image")
                raise RuntimeError(
                    f"YOLO-World+SAM failed on first image: "
                    f"{type(e).__name__}: {e}") from e
            failed += 1
            logger.warning("YOLO-World+SAM failed for %s: %s", img_path, e)
            img = cv2.imread(img_path)
            if img is None:
                continue
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        mp = mask_path_for(img_path, mask_output_dir)
        cv2.imwrite(mp, mask)
        mask_paths.append(mp)
        if image_done_cb:
            image_done_cb(img_path, mp)

    if progress_cb:
        progress_cb(100,
            f"YOLO-World+SAM complete ({failed}/{total} images failed)"
            if failed else "YOLO-World+SAM segmentation complete!")
    return mask_paths


# ═══════════════════════════════════════════════════════════════════════
# Segmentation — SkyWater  (SegFormer-B2)
# ═══════════════════════════════════════════════════════════════════════

SW_INPUT_SIZE = (384, 384)
SW_CLASSES = ["Background", "Sky", "Water", "Person"]
SW_CLASS_IDS = {name.lower(): i for i, name in enumerate(SW_CLASSES)}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_skywater_segmentation(
    image_paths: list[str],
    mask_output_dir: str,
    model_path: str,
    target_classes: list[str],
    *,
    progress_cb: ProgressCB = None,
    image_done_cb: ImageDoneCB = None,
    cancel_check: CancelCheck = None,
) -> list[str]:
    """Run SkyWater SegFormer-B2 segmentation on *image_paths*.

    Writes masks to *mask_output_dir* and returns the list of mask paths.
    """
    from .onnx_utils import create_inference_session

    os.makedirs(mask_output_dir, exist_ok=True)
    mask_paths: list[str] = []
    total = len(image_paths)

    if total == 0:
        return mask_paths

    # Resolve class IDs
    target_ids: set[int] = set()
    for cls_name in target_classes:
        cid = SW_CLASS_IDS.get(cls_name.lower())
        if cid is not None:
            target_ids.add(cid)
    if not target_ids:
        target_ids = {1, 2, 3}  # default: mask sky+water+person

    sess = create_inference_session(model_path)
    providers_used = sess.get_providers()
    logger.info("SkyWater session providers: %s", providers_used)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for idx, img_path in enumerate(image_paths):
        if cancel_check and cancel_check():
            logger.info("SkyWater segmentation cancelled")
            return mask_paths

        pct = int(idx / total * 100)
        name = Path(img_path).name
        if progress_cb:
            progress_cb(pct, f"SkyWater: {name} ({idx + 1}/{total})")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        inp = cv2.resize(img_rgb, SW_INPUT_SIZE).astype(np.float32) / 255.0
        inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]

        # Inference
        logits = sess.run([out_name], {in_name: inp})[0]  # (1,4,384,384)
        mask_small = np.argmax(logits[0], axis=0).astype(np.uint8)  # (384,384)

        # Resize back + filter to target classes
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        final = np.zeros((h, w), dtype=np.uint8)
        for cid in target_ids:
            final[mask_full == cid] = 255

        mp = mask_path_for(img_path, mask_output_dir)
        cv2.imwrite(mp, final)
        mask_paths.append(mp)
        if image_done_cb:
            image_done_cb(img_path, mp)

    if progress_cb:
        progress_cb(100, "SkyWater segmentation complete!")
    return mask_paths


# ═══════════════════════════════════════════════════════════════════════
# Unified segmentation dispatcher
# ═══════════════════════════════════════════════════════════════════════

def run_segmentation(
    image_paths: list[str],
    mask_output_dir: str,
    model_config: dict,
    target_classes: list[str],
    confidence_threshold: float = 0.3,
    *,
    progress_cb: ProgressCB = None,
    image_done_cb: ImageDoneCB = None,
    cancel_check: CancelCheck = None,
) -> list[str]:
    """Dispatch to SAM3, the YOLO-World+SAM cascade, or SkyWater."""
    is_skywater = is_skywater_config(model_config)
    is_cascade = is_yoloworld_sam_config(model_config)
    model_path = model_config.get("model_path") or ""

    # Auto-download if needed
    from .model_downloader import download_model_entry
    logical_name = model_config.get("logical_name", "")
    if is_skywater:
        need_download = not model_path or not os.path.isfile(model_path)
    elif is_cascade:
        parts = [model_config.get(k) for k in (
            "yoloworld_model_path", "text_encoder_path",
            "encoder_model_path", "decoder_model_path")]
        need_download = not all(p and os.path.isfile(p) for p in parts)
    else:
        decoder_path = model_config.get("decoder_model_path", "")
        need_download = (
            not model_config.get("has_downloaded", True)
            or not decoder_path or not os.path.isfile(decoder_path)
        )

    if need_download and logical_name:
        if progress_cb:
            progress_cb(0, f"Downloading model: {logical_name}...")

        def _dl_progress(msg: str, done: int, total: int) -> None:
            pct = 0 if total <= 0 else int(100 * done / total)
            if progress_cb:
                progress_cb(pct, msg)

        updated = download_model_entry(logical_name, progress_callback=_dl_progress)
        model_config.update(updated)

    if cancel_check and cancel_check():
        logger.info("Segmentation cancelled")
        return []

    if is_skywater:
        mp = model_config.get("model_path") or model_config.get("decoder_model_path")
        return run_skywater_segmentation(
            image_paths, mask_output_dir, mp, target_classes,
            progress_cb=progress_cb, image_done_cb=image_done_cb,
            cancel_check=cancel_check,
        )
    elif is_cascade:
        return run_yoloworld_sam_segmentation(
            image_paths, mask_output_dir, model_config,
            target_classes, confidence_threshold,
            progress_cb=progress_cb, image_done_cb=image_done_cb,
            cancel_check=cancel_check,
        )
    else:
        return run_sam_segmentation(
            image_paths, mask_output_dir, model_config,
            target_classes, confidence_threshold,
            progress_cb=progress_cb, image_done_cb=image_done_cb,
            cancel_check=cancel_check,
        )


# ═══════════════════════════════════════════════════════════════════════
# Database Build
# ═══════════════════════════════════════════════════════════════════════

def build_database(
    image_dir: str,
    db_path: str,
    camera_model_id: int = DEFAULT_CAMERA_MODEL_ID,
    camera_params: list[float] | None = None,
    *,
    progress_cb: ProgressCB = None,
) -> str:
    """Build a COLMAP-compatible SQLite database from images in *image_dir*.

    Returns the path to the created database.
    """
    if progress_cb:
        progress_cb(10, "Collecting images...")

    paths = collect_image_files([image_dir], recursive=False)
    if not paths:
        raise FileNotFoundError(f"No images in {image_dir}")

    if progress_cb:
        progress_cb(30, f"Found {len(paths)} images")

    model = get_camera_model(camera_model_id)

    if progress_cb:
        progress_cb(50, "Building database...")

    db = ColmapDatabase(db_path)
    db.reset()
    with db:
        result = db.build_project(image_dir, model, camera_params)

    if progress_cb:
        progress_cb(100, f"Ready: {result['image_count']} images, {model.name}")

    return db_path


# ═══════════════════════════════════════════════════════════════════════
# Resize (applied to existing images on disk)
# ═══════════════════════════════════════════════════════════════════════

def apply_resize(
    images_dir: str,
    resize_mode: str = "",
    resize_max_dim: int = 2000,
    resize_factor: int = 4,
    *,
    progress_cb: ProgressCB = None,
) -> None:
    """Resize all images in *images_dir* in-place."""
    if not resize_mode:
        return

    all_paths = collect_image_files([images_dir])
    total = len(all_paths)

    for idx, p in enumerate(all_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        if resize_mode == "downscale":
            if resize_factor <= 1:
                continue
            img = resize_image(img, width=w // resize_factor,
                               height=h // resize_factor, keep_aspect=False)
        elif resize_mode == "max_dim":
            img = resize_image(img, max_dim=resize_max_dim)

        cv2.imwrite(p, img)
        if progress_cb:
            progress_cb(int((idx + 1) / total * 100),
                        f"Resizing: {Path(p).name} ({idx + 1}/{total})")


# ═══════════════════════════════════════════════════════════════════════
# Image copy helper (from input folders to output images dir)
# ═══════════════════════════════════════════════════════════════════════

def copy_input_images(image_paths: list[str], images_dir: str) -> list[str]:
    """Copy images from input folders into *images_dir*.

    Returns the list of destination paths.
    """
    os.makedirs(images_dir, exist_ok=True)
    copied: list[str] = []
    for folder in image_paths:
        for src in collect_image_files([folder]):
            dst = os.path.join(images_dir, os.path.basename(src))
            if src != dst:
                shutil.copy2(src, dst)
            copied.append(dst)
    return copied
