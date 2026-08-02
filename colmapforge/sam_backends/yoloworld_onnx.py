"""
YOLO-World v2 open-vocabulary detector (ONNX).

Uses the export from https://github.com/wkentaro/yolo-world-onnx, which takes
the image AND per-class CLIP text features as inputs — the vocabulary is set
per call from free-text class names, no re-export needed for new prompts.

Inference flow (mirrors osam's reference driver):
  class names → CLIP ViT-B/32 textual encoder → L2-normed text features
  image → letterbox to the detector input size → YOLO head → per-class NMS
  → boxes mapped back to original pixel coordinates.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..onnx_utils import create_inference_session

logger = logging.getLogger(__name__)

# Letterbox fill value used by the YOLO-World reference pipeline.
_PAD_VALUE = 114


class YoloWorldONNX:
    """Text-prompted detector: CLIP text encoder + YOLO-World v2 head."""

    def __init__(self, model_path: str, text_encoder_path: str) -> None:
        self.session = create_inference_session(model_path)
        # The CLIP textual encoder (int64 token inputs) breaks under the
        # CoreML EP; it runs once per vocabulary (cached), so CPU is fine.
        self.text_session = create_inference_session(text_encoder_path, force_cpu=True)

        # Detector input: "images" [1, 3, S, S]
        shape = self.session.get_inputs()[0].shape
        try:
            self.input_size = int(shape[2])
        except (TypeError, ValueError):
            self.input_size = 640

        # Text features depend only on the class names — cache per vocabulary
        # so a video run pays for CLIP encoding once, not once per frame.
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}

    # ── text encoding ────────────────────────────────────────────────

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """CLIP-encode *texts* (plus a trailing padding class) → (N+1, 512)."""
        key = tuple(texts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        try:
            from osam._models.yoloworld.clip import tokenize
        except ImportError as e:
            raise RuntimeError(
                "YOLO-World text prompts require the 'osam' package for CLIP "
                "tokenization — reinstall dependencies (uv sync) and retry."
            ) from e

        # The trailing " " is the padding/background class expected by the
        # export (same trick as the osam reference driver).
        tokens = tokenize(texts=list(texts) + [" "])
        feats = self.text_session.run(None, {"input": tokens})[0]
        feats = feats / np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
        feats = feats.astype(np.float32)
        self._text_cache[key] = feats
        return feats

    # ── detection ────────────────────────────────────────────────────

    def detect(
        self,
        img_rgb: np.ndarray,
        texts: list[str],
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> list[tuple[np.ndarray, int, float]]:
        """Detect *texts* in an RGB image.

        Returns a list of ``(box_xyxy, class_index, score)`` with boxes as
        int arrays in original pixel coordinates.
        """
        text_features = self.encode_texts(texts)
        blob, orig_hw, pad_hw = self._letterbox(img_rgb)

        scores, boxes = self.session.run(
            ["scores", "boxes"],
            {"images": blob[None], "text_features": text_features[None]},
        )
        scores, boxes = scores[0], boxes[0]  # (N, C+1), (N, 4) xyxy
        scores = scores[:, : len(texts)]     # drop the padding class

        boxes, scores, labels = _nms_per_class(
            boxes, scores, iou_threshold, score_threshold, max_detections)
        boxes = self._unletterbox(boxes, orig_hw, pad_hw)

        logger.debug("YOLO-World: %d detections for %s", len(boxes), texts)
        return list(zip(boxes, labels, scores))

    # ── geometry helpers ─────────────────────────────────────────────

    def _letterbox(self, img: np.ndarray):
        """Resize long side to input_size, pad centered → (CHW float, hw, pad_hw)."""
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)
        pad_h = self.input_size - resized.shape[0]
        pad_w = self.input_size - resized.shape[1]
        padded = np.pad(
            resized,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2),
             (0, 0)),
            mode="constant", constant_values=_PAD_VALUE,
        )
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob, (h, w), (pad_h, pad_w)

    def _unletterbox(self, boxes: np.ndarray, orig_hw, pad_hw) -> np.ndarray:
        """Map boxes from letterboxed space back to original pixels."""
        if len(boxes) == 0:
            return boxes
        boxes = boxes.astype(np.float64)
        boxes -= np.array([pad_hw[1] // 2, pad_hw[0] // 2] * 2)
        boxes /= self.input_size / max(orig_hw)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_hw[1])
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_hw[0])
        return boxes.round().astype(int)


def _nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy class-wise NMS. *scores* is (N, C); returns kept (boxes, scores, labels)."""
    keep_boxes, keep_scores, keep_labels = [], [], []
    for c in range(scores.shape[1]):
        cls_scores = scores[:, c]
        idx = np.where(cls_scores >= score_threshold)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-cls_scores[idx])]
        while order.size:
            i = order[0]
            keep_boxes.append(boxes[i])
            keep_scores.append(float(cls_scores[i]))
            keep_labels.append(c)
            if order.size == 1:
                break
            ious = _iou(boxes[i], boxes[order[1:]])
            order = order[1:][ious <= iou_threshold]

    if not keep_boxes:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    boxes_np = np.asarray(keep_boxes, dtype=np.float32)
    scores_np = np.asarray(keep_scores, dtype=np.float32)
    labels_np = np.asarray(keep_labels, dtype=np.int64)
    if len(scores_np) > max_detections:
        top = np.argsort(-scores_np)[:max_detections]
        boxes_np, scores_np, labels_np = boxes_np[top], scores_np[top], labels_np[top]
    return boxes_np, scores_np, labels_np


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an (N, 4) array of xyxy boxes."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)
