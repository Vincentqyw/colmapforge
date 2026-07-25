"""HuggingFace Hub model downloader.

Downloads model weights on first use and caches them via the standard
HF cache (``~/.cache/huggingface/hub`` on Linux/macOS,
``%USERPROFILE%\\.cache\\huggingface\\hub`` on Windows). Re-downloads
are skipped automatically when the ETag matches.

This is what makes the app truly standalone — no hardcoded local paths,
no manual model placement. The first time a user picks SkyWater, the
weights are fetched from ``Realcat/skywater_seg`` and cached forever.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────
# To add a new downloadable model, append an entry here. Each entry maps
# a stable logical name to (repo_id, filename, display_name).
DOWNLOADABLE_MODELS: dict[str, dict] = {
    "skywater_segformer_b2_fp16": {
        "repo_id": "Realcat/skywater_seg",
        "filename": "skywater_segformer_b2_fp16.onnx",
        "display_name": "SkyWater SegFormer-B2 (FP16, ~48 MB)",
        "size_hint_mb": 48,
    },
    "skywater_segformer_b2_fp32": {
        "repo_id": "Realcat/skywater_seg",
        "filename": "skywater_segformer_b2_fp32.onnx",
        "display_name": "SkyWater SegFormer-B2 (FP32, ~95 MB)",
        "size_hint_mb": 95,
    },
}


# ── Public API ────────────────────────────────────────────────────────

def is_huggingface_hub_available() -> bool:
    """Return ``True`` if the ``huggingface_hub`` package can be imported."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_info(logical_name: str) -> dict | None:
    """Return the registry entry for *logical_name*, or ``None``."""
    return DOWNLOADABLE_MODELS.get(logical_name)


def resolve_cached_path(logical_name: str) -> str | None:
    """Return the local path if the model is already in the HF cache.

    Returns ``None`` when:
    - the logical name is unknown
    - huggingface_hub is not installed
    - the file has not been downloaded yet
    """
    info = DOWNLOADABLE_MODELS.get(logical_name)
    if info is None:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(
            repo_id=info["repo_id"],
            filename=info["filename"],
        )
        # try_to_load_from_cache returns a CommitNotFound / None subclass
        # when the file is not cached; check with isfile for safety.
        if path and isinstance(path, str) and os.path.isfile(path):
            return path
    except Exception as e:
        logger.debug("Cache lookup failed for %s: %s", logical_name, e)
    return None


def download_model(
    logical_name: str,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> str:
    """Download a model from HuggingFace Hub. Returns the local path.

    Parameters
    ----------
    logical_name:
        Key in :data:`DOWNLOADABLE_MODELS`.
    progress_callback:
        Optional ``(message, downloaded_bytes, total_bytes)`` callback.
        ``total_bytes`` is 0 when unknown. If huggingface_hub does not
        expose a compatible progress hook, the callback is invoked once
        at the start with a "downloading" message and once at the end.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    RuntimeError
        On network/HTTP errors, with a manual-download URL in the message.
    """
    info = DOWNLOADABLE_MODELS.get(logical_name)
    if info is None:
        raise ValueError(f"Unknown model: {logical_name}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is not installed. Install with:\n"
            "  uv pip install huggingface_hub"
        ) from e

    repo_id = info["repo_id"]
    filename = info["filename"]

    # Fast path: already cached
    cached = resolve_cached_path(logical_name)
    if cached:
        logger.info("Model %s already cached at %s", logical_name, cached)
        if progress_callback:
            progress_callback(f"Using cached {filename}", 0, 0)
        return cached

    logger.info("Downloading %s/%s ...", repo_id, filename)
    if progress_callback:
        progress_callback(
            f"Downloading {filename} (~{info.get('size_hint_mb', '?')} MB)...",
            0, 0,
        )

    try:
        # huggingface_hub >= 0.24 accepts a callback via the download API
        # but the exact shape varies. Use the simple form — HF prints its
        # own tqdm bar to stderr — and emit a final "complete" callback.
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {repo_id}/{filename}: {e}\n"
            "Check your network connection, or download manually from:\n"
            f"  https://huggingface.co/{repo_id}/resolve/main/{filename}"
        ) from e

    if progress_callback:
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        progress_callback(f"Downloaded {filename}", size, size)

    logger.info("Model cached at: %s", path)
    return path
