"""
EfficientViT-SAM (MIT Han Lab) ONNX backend — a fast SAM drop-in.

Official ONNX exports from https://huggingface.co/mit-han-lab/efficientvit-sam
(Apache-2.0). Interface quirks verified against the exported graphs:

- encoder: ``input_image`` [B, 3, S, S] float32, ImageNet-normalized, long
  side resized to S (512 for L-series, 1024 for XL) and padded bottom-right;
- decoder: ``point_coords`` are in the standard SAM **1024-long-side frame**
  regardless of the encoder input size, and it accepts a batch of prompts —
  one call masks every detector box at once.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..onnx_utils import create_inference_session

logger = logging.getLogger(__name__)

# ImageNet statistics on the 0–255 scale (applied outside the graph).
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# SAM's canonical prompt-coordinate frame (long side), independent of the
# encoder resolution — feeding encoder-frame coords shifts every mask.
_COORD_FRAME = 1024


class EfficientViTSAMONNX:
    """Box-promptable EfficientViT-SAM with a batched decoder fast path."""

    def __init__(self, encoder_model_path: str, decoder_model_path: str) -> None:
        self.encoder_session = create_inference_session(encoder_model_path)
        self.decoder_session = create_inference_session(decoder_model_path)
        shape = self.encoder_session.get_inputs()[0].shape  # [B, 3, S, S]
        try:
            self.input_size = int(shape[2])
        except (TypeError, ValueError):
            self.input_size = 512

    def encode(self, cv_image: np.ndarray) -> dict:
        """Embed a BGR image (as returned by ``cv2.imread``)."""
        h, w = cv_image.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))

        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        resized = (resized.astype(np.float32) - _MEAN) / _STD
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
        padded[:nh, :nw] = resized

        embedding = self.encoder_session.run(
            None, {"input_image": padded.transpose(2, 0, 1)[None]})[0]
        return {
            "image_embedding": embedding,
            "original_size": (h, w),
            "resized_size": (nh, nw),
            "coord_scale": _COORD_FRAME / max(h, w),
        }

    def predict_masks(self, embedding: dict, prompt) -> np.ndarray:
        """Run the decoder for point/rectangle marks (same format as SAM1/2).

        Returns a float array of shape ``(1, N, H, W)`` at the original image
        size; positive values are foreground.
        """
        boxes = [m["data"] for m in prompt if m["type"] == "rectangle"]
        if boxes:
            return self.predict_boxes(embedding, boxes)[None]
        points = [m["data"] for m in prompt if m["type"] == "point"]
        labels = [[float(m["label"]) for m in prompt if m["type"] == "point"]]
        coords = np.asarray(points, np.float32)[None] * embedding["coord_scale"]
        masks = self._decode(embedding, coords, np.asarray(labels, np.float32))
        return masks[None]

    def predict_boxes(self, embedding: dict, boxes) -> np.ndarray:
        """Batched fast path: one decoder call for all *boxes* (xyxy, original px).

        Returns float masks of shape ``(len(boxes), H, W)``.
        """
        s = embedding["coord_scale"]
        coords = np.array(
            [[[b[0], b[1]], [b[2], b[3]]] for b in boxes], np.float32) * s
        labels = np.tile(np.array([[2.0, 3.0]], np.float32), (len(boxes), 1))
        return self._decode(embedding, coords, labels)

    def _decode(self, embedding: dict, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
        masks, _ious = self.decoder_session.run(None, {
            "image_embeddings": embedding["image_embedding"],
            "point_coords": coords.astype(np.float32),
            "point_labels": labels.astype(np.float32),
        })
        h, w = embedding["original_size"]
        nh, nw = embedding["resized_size"]
        out = np.empty((masks.shape[0], h, w), dtype=np.float32)
        for i, m in enumerate(masks[:, 0]):
            # Low-res logits cover the padded square frame: upsample to the
            # encoder frame, drop the padding, then restore original size.
            m = cv2.resize(m, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
            out[i] = cv2.resize(m[:nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)
        return out
