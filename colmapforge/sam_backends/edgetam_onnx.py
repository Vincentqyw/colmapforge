"""
EdgeTAM (Meta) ONNX backend — image mode.

Uses the export from https://huggingface.co/onnx-community/EdgeTAM-ONNX
(Apache-2.0): ``vision_encoder`` + ``prompt_encoder_mask_decoder`` in the
HF SAM2 layout. Interface quirks verified against the exported graphs:

- the encoder takes a **square** 1024×1024 resize (no aspect-preserving
  letterbox), ImageNet-normalized after 1/255 rescale;
- prompts are in the 1024×1024 frame, so x and y scale independently;
- ``input_points``/``input_labels`` are required inputs with a fixed
  point-batch of 1 — boxes therefore cannot be batched together with the
  padding point, and the decoder runs once per box;
- outputs are multimask (3 candidates) + IoU scores; the best candidate
  per prompt is selected by predicted IoU.

Note: this export contains no memory-attention/memory-encoder graphs, so
EdgeTAM's video propagation is NOT available here — image mode only.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..onnx_utils import create_inference_session

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 1024


class EdgeTAMONNX:
    """Box-promptable EdgeTAM (image mode)."""

    def __init__(self, encoder_model_path: str, decoder_model_path: str) -> None:
        self.encoder_session = create_inference_session(encoder_model_path)
        self.decoder_session = create_inference_session(decoder_model_path)

    def encode(self, cv_image: np.ndarray) -> dict:
        """Embed a BGR image (as returned by ``cv2.imread``)."""
        h, w = cv_image.shape[:2]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        feats = self.encoder_session.run(None, {"pixel_values": x})
        return {
            "image_embeddings.0": feats[0],
            "image_embeddings.1": feats[1],
            "image_embeddings.2": feats[2],
            "original_size": (h, w),
        }

    def predict_masks(self, embedding: dict, prompt) -> np.ndarray:
        """Run the decoder for rectangle marks (same format as SAM1/2).

        Returns a float array of shape ``(1, N, H, W)`` at the original image
        size; positive values are foreground.
        """
        h, w = embedding["original_size"]
        boxes = [m["data"] for m in prompt if m["type"] == "rectangle"]
        out = np.empty((len(boxes), h, w), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            box = np.array(
                [[[x1 * _INPUT_SIZE / w, y1 * _INPUT_SIZE / h,
                   x2 * _INPUT_SIZE / w, y2 * _INPUT_SIZE / h]]], np.float32)
            ious, masks = self.decoder_session.run(
                ["iou_scores", "pred_masks"],
                {
                    "image_embeddings.0": embedding["image_embeddings.0"],
                    "image_embeddings.1": embedding["image_embeddings.1"],
                    "image_embeddings.2": embedding["image_embeddings.2"],
                    "input_boxes": box,
                    # Required inputs; a single label -1 point is ignored by
                    # the prompt encoder (SAM padding convention).
                    "input_points": np.zeros((1, 1, 1, 2), np.float32),
                    "input_labels": np.full((1, 1, 1), -1, np.int64),
                },
            )
            best = masks[0, 0][int(ious[0, 0].argmax())]
            # Square resize on encode → masks map to the full image directly.
            out[i] = cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR)
        return out[None]
