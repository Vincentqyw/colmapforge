"""CLIP BPE tokenizer (vendored).

Verbatim copy of OpenAI CLIP's tokenizer (MIT, see LICENSE) as adapted by
the ``osam`` project — pure stdlib + numpy, no extra dependencies. Used to
tokenize text prompts for the CLIP ViT-B/32 text encoder (YOLO-World
cascades) and the SAM3 language encoder. Kept verbatim so token ids stay
byte-identical to the drivers these ONNX exports were validated against.
"""

from .clip import tokenize  # noqa: F401
