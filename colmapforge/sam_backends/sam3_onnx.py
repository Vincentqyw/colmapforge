import functools
import logging
import os
import subprocess
from typing import Any

import cv2
import numpy as np

from ..onnx_utils import create_inference_session

logger = logging.getLogger(__name__)

# SAM3 OOMs on <8GB GPUs, so only run on GPU when there is enough VRAM.
_SAM3_MIN_VRAM_GB = 8.0


@functools.cache
def _sam3_force_cpu() -> bool:
    """Decide whether SAM3 sessions must stay on CPU.

    Override with COLMAPFORGE_SAM3_DEVICE=cpu|cuda; otherwise use the GPU
    only when total VRAM >= 8GB (SAM3 ViT-H OOMs below that).
    """
    override = os.environ.get("COLMAPFORGE_SAM3_DEVICE", "").lower()
    if override == "cpu":
        return True
    if override in ("cuda", "gpu"):
        return False
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        vram_gb = max(float(x) for x in out.split()) / 1024.0
    except Exception:
        return True
    force_cpu = vram_gb < _SAM3_MIN_VRAM_GB
    logger.info("SAM3 device: %s (VRAM %.1f GB)",
                "CPU" if force_cpu else "CUDA", vram_gb)
    return force_cpu


class SegmentAnything3ONNX:
    """Segmentation model using Segment Anything 3 (SAM3)"""

    def __init__(
        self,
        image_encoder_path,
        decoder_model_path,
        language_encoder_path=None,
    ) -> None:
        self.image_encoder = SAM3ImageEncoder(image_encoder_path)
        self.language_encoder = None
        if language_encoder_path:
            self.language_encoder = SAM3LanguageEncoder(language_encoder_path)
        self.decoder = SAM3ImageDecoder(decoder_model_path)
        # Language features depend only on the prompt text, so cache them —
        # segmenting N images with C classes then costs C encoder runs, not N×C.
        self._language_cache: dict[str, dict[str, Any]] = {}

    def encode(self, cv_image: np.ndarray, text_prompt=None) -> dict[str, Any]:
        """Encode an image (and optional text prompt) into an embedding dict.

        Parameters
        ----------
        cv_image:
            RGB uint8 image as returned by ``qt_img_to_rgb_cv_img``.
        text_prompt:
            Natural-language description of the target object.
            Falls back to ``"visual"`` when *None*.
        """
        original_size = cv_image.shape[:2]
        image_encoder_outputs = self.image_encoder(cv_image)

        embedding: dict[str, Any] = {
            "vision_pos_enc_0": image_encoder_outputs[0],
            "vision_pos_enc_1": image_encoder_outputs[1],
            "vision_pos_enc_2": image_encoder_outputs[2],
            "backbone_fpn_0": image_encoder_outputs[3],
            "backbone_fpn_1": image_encoder_outputs[4],
            "backbone_fpn_2": image_encoder_outputs[5],
            "original_size": original_size,
        }
        return self._apply_language(embedding, text_prompt)

    def _apply_language(self, embedding: dict, text_prompt: str | None) -> dict:
        """Fill *embedding*'s language inputs from *text_prompt*.

        Shared by :meth:`encode` and :meth:`update_language` so the two
        language keys stay in sync.  The image tensors are shared by
        reference (shallow copy), so re-encoding a different class term is
        cheap — only the three small ``language_*`` keys are replaced.
        """
        new_embedding = dict(embedding)  # shallow copy; image tensors shared
        new_embedding["language_mask"] = None
        new_embedding["language_features"] = None
        new_embedding["language_embeds"] = None
        if self.language_encoder is not None:
            text = text_prompt or "visual"
            cached = self._language_cache.get(text)
            if cached is None:
                cached = self.language_encoder.encode(text)
                self._language_cache[text] = cached
            new_embedding.update(cached)
        return new_embedding

    def predict_masks(
        self,
        embedding: dict[str, Any],
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Run the decoder; detections come from the language features.

        Returns
        -------
        Boolean mask array of shape ``(N, 1, H, W)``.  May be empty
        (shape ``(0, 1, H, W)``) when no confident detections are found.
        """
        original_size = embedding["original_size"]
        # Dummy box (box_masks=True → text-only prompt, no real box).
        box_coords_np = np.zeros((1, 1, 4), dtype=np.float32)
        box_labels_np = np.array([[1]], dtype=np.int64)
        box_masks_np = np.array([[True]], dtype=np.bool_)

        masks, scores, _ = self.decoder(
            original_size,
            embedding["vision_pos_enc_0"],
            embedding["vision_pos_enc_1"],
            embedding["vision_pos_enc_2"],
            embedding["backbone_fpn_0"],
            embedding["backbone_fpn_1"],
            embedding["backbone_fpn_2"],
            embedding.get("language_mask"),
            embedding.get("language_features"),
            embedding.get("language_embeds"),
            box_coords_np,
            box_labels_np,
            box_masks_np,
        )

        # Filter by confidence threshold.
        if len(scores) > 0:
            keep = np.where(scores > confidence_threshold)[0]
            if len(keep) > 0:
                masks = masks[keep]
            else:
                masks = np.zeros((0,) + masks.shape[1:], dtype=masks.dtype)

        # Guarantee masks come back at the input image resolution so callers
        # can composite them directly without a shape check.
        return self.transform_masks(masks, original_size)

    def update_language(self, embedding: dict, text_prompt: str) -> dict:
        """Re-encode just the language features for a different class term.

        See :meth:`_apply_language` — the image encoding is not repeated.
        """
        return self._apply_language(embedding, text_prompt)

    def transform_masks(self, masks, original_size):
        """Resize masks to *original_size* if the decoder emitted them at its
        native resolution instead (some exports do).

        Masks are boolean ``(N, 1, H, W)``; nearest-neighbour keeps them
        binary.  Shapes already matching *original_size* are returned as-is.
        """
        if masks.shape[2:] == tuple(original_size):
            return masks
        resized = np.empty(
            (masks.shape[0], masks.shape[1]) + tuple(original_size), dtype=masks.dtype)
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                resized[i, j] = cv2.resize(
                    masks[i, j].astype(np.uint8), (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(masks.dtype)
        return resized


class SAM3ImageEncoder:
    """Runs the SAM3 image backbone ONNX model.

    Expected model input
    --------------------
    name  : ``"image"``
    shape : ``[3, 1008, 1008]``
    dtype : ``tensor(uint8)`` (the model includes normalisation internally)
            or ``tensor(float)`` for older exports without normalisation.
    """

    def __init__(self, path: str) -> None:
        self.session = create_inference_session(path, force_cpu=_sam3_force_cpu())
        encoder_input = self.session.get_inputs()[0]
        self.input_name: str = encoder_input.name
        self.input_shape = encoder_input.shape
        self.input_type: str = encoder_input.type

        # Determine H/W from the ONNX shape.
        # Current export: [3, H, W]  (no batch dimension)
        # Legacy export:  [1, 3, H, W]
        if len(self.input_shape) == 3:
            self.input_height: int = int(self.input_shape[1]) or 1008
            self.input_width: int = int(self.input_shape[2]) or 1008
        elif len(self.input_shape) >= 4:
            self.input_height = int(self.input_shape[2]) or 1008
            self.input_width = int(self.input_shape[3]) or 1008
        else:
            self.input_height = 1008
            self.input_width = 1008

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        input_tensor = self.prepare_input(image)
        return self.session.run(None, {self.input_name: input_tensor})

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare image tensor for the ONNX encoder.

        The anylabeling pipeline passes an RGB image from
        ``qt_img_to_rgb_cv_img``.  Since the SAM3 normalisation uses equal
        mean/std across all channels (0.5, 0.5, 0.5), RGB vs BGR order
        has no effect on the normalised values, so no colour-space
        conversion is required here.
        """
        input_img = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # (H, W, C) → (C, H, W)
        input_img = input_img.transpose(2, 0, 1)

        if self.input_type == "tensor(float)":
            # Older export without normalisation inside the model.
            # Apply (x/255 − 0.5) / 0.5 to map [0,255] → [−1, 1].
            input_tensor = ((input_img / 255.0) - 0.5) / 0.5
            input_tensor = input_tensor.astype(np.float32)
        else:
            # Current export bakes normalisation in – pass raw uint8.
            input_tensor = input_img.astype(np.uint8)

        return input_tensor


class SAM3LanguageEncoder:
    """Runs the SAM3 language-encoder ONNX model.

    Expected model input
    --------------------
    name  : ``"tokens"``
    shape : ``[1, 32]``
    dtype : int64

    Tokenisation uses the vendored CLIP BPE tokeniser
    (``colmapforge.sam_backends.clip``) at ``context_length=32``, matching
    the reference driver these exports were validated against — a mismatched
    vocabulary silently produces meaningless language features rather than
    an error.
    """

    #: Output count varies by export: some publish only
    #: (mask, features), others add a third ``text_embeds`` tensor.
    _OUTPUT_KEYS = ("language_mask", "language_features", "language_embeds")

    def __init__(self, path: str) -> None:
        self.session = create_inference_session(path, force_cpu=_sam3_force_cpu())

    def encode(self, text: str) -> dict[str, Any]:
        """Encode *text* into the decoder's language inputs.

        Returns a dict keyed by decoder input name.  Exports that omit
        ``text_embeds`` simply yield two entries; the decoder substitutes a
        zero tensor for anything missing, so positional indexing (which used
        to assume three outputs) is avoided here.
        """
        return dict(zip(self._OUTPUT_KEYS, self(text)))

    def __call__(self, text: str) -> list[np.ndarray]:
        from .clip import tokenize

        tokens = tokenize([text], context_length=32)
        return self.session.run(None, {"tokens": tokens})


class SAM3ImageDecoder:
    """Runs the SAM3 decoder ONNX model.

    Expected output order (ONNX export names):
        [0] boxes  – float (N, 4)
        [1] scores – float (N,)
        [2] masks  – bool  (N, 1, H, W)

    ``__call__`` returns ``(masks, scores, boxes)`` for caller convenience.
    """

    def __init__(self, path: str) -> None:
        self.session = create_inference_session(path, force_cpu=_sam3_force_cpu())
        self.input_names: list[str] = [i.name for i in self.session.get_inputs()]

    def __call__(
        self,
        original_size,
        vision_pos_enc_0,
        vision_pos_enc_1,
        vision_pos_enc_2,
        backbone_fpn_0,
        backbone_fpn_1,
        backbone_fpn_2,
        language_mask,
        language_features,
        language_embeds,
        box_coords,
        box_labels,
        box_masks,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs: dict[str, Any] = {
            "original_height": np.array(original_size[0], dtype=np.int64),
            "original_width": np.array(original_size[1], dtype=np.int64),
            "vision_pos_enc_0": vision_pos_enc_0,
            "vision_pos_enc_1": vision_pos_enc_1,
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        }

        # Supply dummy tensors for language inputs when no encoder was used.
        # Shapes match the actual ONNX decoder's expected inputs (verified by
        # inspecting sam3_decoder.onnx with onnxruntime):
        #   language_mask     – bool  [1, 32]
        #   language_features – float [32, 1, 256]
        if "language_mask" in self.input_names and inputs["language_mask"] is None:
            inputs["language_mask"] = np.zeros((1, 32), dtype=np.bool_)
        if (
            "language_features" in self.input_names
            and inputs["language_features"] is None
        ):
            inputs["language_features"] = np.zeros((32, 1, 256), dtype=np.float32)
        if "language_embeds" in self.input_names and inputs["language_embeds"] is None:
            inputs["language_embeds"] = np.zeros((32, 1, 1024), dtype=np.float32)

        # Only forward inputs that the model actually requires – onnxsim may
        # have removed some (e.g. vision_pos_enc_0/1, language_embeds) during
        # simplification.
        model_inputs = {
            k: v for k, v in inputs.items() if k in self.input_names and v is not None
        }
        outputs = self.session.run(None, model_inputs)
        # ONNX export order: [0]=boxes, [1]=scores, [2]=masks
        # Return as (masks, scores, boxes).
        return outputs[2], outputs[1], outputs[0]
