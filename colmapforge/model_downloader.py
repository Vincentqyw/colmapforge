"""Model discovery and download — SAM (zip) + SkyWater (HF Hub).

On first launch, scans the built-in ``models.yaml`` registry and creates
stub config.yaml files under ``~/.colmapforge/models/<name>/`` for
any model not yet downloaded. Downloads happen lazily on first use (when
the user clicks Run), with progress reported to the toolbar.

This is what makes the app truly standalone: no hardcoded local paths,
no manual model placement required.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import os
import pathlib
import shutil
import tempfile
import time
import urllib.request
import zipfile
from typing import Callable

import yaml

from . import configs as _configs_pkg

logger = logging.getLogger(__name__)

# ── CONSTANTS ────────────────────────────────────────────────────────

MODELS_ROOT = os.path.join(os.path.expanduser("~"), ".colmapforge", "models")

# Legacy path from before the rename (also used by AnyLabeling).
_OLD_MODELS_ROOT = os.path.join(os.path.expanduser("~"), "anylabeling_data", "models")


def _migrate_old_models() -> None:
    """One-time migration: move SAM models from the legacy location
    ``~/anylabeling_data/models/`` to the new ``~/.colmapforge/models/``.

    Only folders matching names in the SAM registry are migrated;
    unrelated models (e.g. YOLO) stay where they are.
    """
    if not os.path.isdir(_OLD_MODELS_ROOT):
        return

    registry = _load_sam_registry()
    known_names = {e["name"] for e in registry}

    pathlib.Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)

    for model_name in sorted(os.listdir(_OLD_MODELS_ROOT)):
        if model_name not in known_names:
            continue
        old_dir = os.path.join(_OLD_MODELS_ROOT, model_name)
        new_dir = os.path.join(MODELS_ROOT, model_name)
        if not os.path.isdir(old_dir) or os.path.exists(new_dir):
            continue
        try:
            shutil.move(old_dir, new_dir)
            logger.info("Migrated model %s → %s", old_dir, new_dir)
        except OSError as e:
            logger.warning("Failed to migrate %s: %s", model_name, e)


# ── Registry ─────────────────────────────────────────────────────────

# Cached at import time from the bundled models.yaml.
_SAM_REGISTRY: list[dict] = []


def _load_sam_registry() -> list[dict]:
    """Load SAM model entries from the bundled models.yaml."""
    global _SAM_REGISTRY
    if _SAM_REGISTRY:
        return _SAM_REGISTRY
    try:
        with pkg_resources.open_text(_configs_pkg, "models.yaml") as f:
            _SAM_REGISTRY = yaml.safe_load(f) or []
    except Exception as e:
        logger.warning("Failed to load models.yaml: %s", e)
        _SAM_REGISTRY = []
    return _SAM_REGISTRY


# SkyWater models (single-file HF Hub downloads, not zip-based).
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


# ── Public API: HF Hub (SkyWater) ────────────────────────────────────


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
        ``total_bytes`` is 0 when unknown.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    RuntimeError
        On network/HTTP errors.
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


# ── Public API: SAM model discovery ──────────────────────────────────


def _build_sam_config(entry: dict, model_dir: str) -> dict:
    """Build a model config dict from a downloaded SAM model directory.

    Reads the local config.yaml (written inside the zip) and resolves all
    ONNX paths to absolute paths. Falls back to filename heuristics when
    explicit paths are missing.
    """
    cfg_path = os.path.join(model_dir, "config.yaml")
    cfg: dict = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    onnx_files = sorted(
        f for f in os.listdir(model_dir)
        if f.endswith(".onnx")
    )

    encoder = cfg.get("encoder_model_path", "")
    decoder = cfg.get("decoder_model_path", "")
    lang = cfg.get("language_encoder_path", "")

    ep = os.path.join(model_dir, encoder) if encoder else ""
    dp = os.path.join(model_dir, decoder) if decoder else ""
    lp = os.path.join(model_dir, lang) if lang else ""

    # Fallback: search ONNX files by naming convention
    if not dp or not os.path.isfile(dp):
        decs = [f for f in onnx_files if "decoder" in f.lower()]
        if decs:
            dp = os.path.join(model_dir, decs[0])
    if not ep or not os.path.isfile(ep):
        encs = [
            f for f in onnx_files
            if "encoder" in f.lower() and "language" not in f.lower()
        ]
        if encs:
            ep = os.path.join(model_dir, encs[0])
    if not lp or not os.path.isfile(lp):
        langs = [f for f in onnx_files if "language" in f.lower() or "text" in f.lower()]
        if langs:
            lp = os.path.join(model_dir, langs[0])

    merged: dict = {
        "type": entry.get("type", "segment_anything"),
        "name": entry["name"],
        "display_name": entry.get("display_name", entry["name"]),
        "encoder_model_path": ep,
        "decoder_model_path": dp,
        "logical_name": entry["name"],
        "has_downloaded": True,
    }
    if lp and os.path.isfile(lp):
        merged["language_encoder_path"] = lp
    return merged


def _build_stub_config(entry: dict, model_dir: str) -> dict:
    """Build a placeholder config for a model that has not been downloaded yet.

    Writes a stub config.yaml to disk so the model appears in future
    launches even if the app is restarted before download.
    """
    cfg_path = os.path.join(model_dir, "config.yaml")

    # Keep any existing on-disk config (might have been written by
    # AnyLabeling or a previous launch).
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                existing = yaml.safe_load(f) or {}
            if existing.get("has_downloaded", False):
                # Already downloaded — use the full config builder.
                return _build_sam_config(entry, model_dir)
        except Exception:
            pass

    size_hint = entry.get("size_hint_mb", 0)
    size_tag = f" [Download ~{size_hint} MB]" if size_hint else " [Download]"

    stub = dict(entry)
    stub["config_file"] = os.path.abspath(cfg_path)
    stub["has_downloaded"] = False
    stub["is_custom_model"] = False

    # Write stub if it doesn't exist or has_downloaded is missing
    needs_write = True
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                disk_cfg = yaml.safe_load(f) or {}
            if disk_cfg.get("has_downloaded") is False:
                needs_write = False  # stub is fine
        except Exception:
            pass

    if needs_write:
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(stub, f, default_flow_style=False)

    return {
        "type": entry.get("type", "segment_anything"),
        "name": entry["name"],
        "display_name": entry.get("display_name", entry["name"]) + size_tag,
        "encoder_model_path": "",
        "decoder_model_path": "",
        "logical_name": entry["name"],
        "has_downloaded": False,
        "download_url": entry.get("download_url", ""),
        "size_hint_mb": size_hint,
    }


def discover_models() -> list[dict]:
    """Discover all available models (SAM + SkyWater).

    Scans the bundled model registry, checks which models already exist
    on disk, and returns a flat list of model config dicts ready for the
    segmentation section UI.

    For models not yet downloaded, stub config.yaml files are created so
    they persist across restarts.
    """
    # One-time migration from the legacy anylabeling_data location.
    _migrate_old_models()
    configs: list[dict] = []

    # ── SAM models from registry ──────────────────────────────────
    registry = _load_sam_registry()
    for entry in registry:
        name = entry["name"]
        model_dir = os.path.join(MODELS_ROOT, name)
        cfg_path = os.path.join(model_dir, "config.yaml")

        # Check if already downloaded (has ONNX files)
        if os.path.isdir(model_dir):
            onnx_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(".onnx")
            ] if os.path.isdir(model_dir) else []
            if onnx_files:
                configs.append(_build_sam_config(entry, model_dir))
                continue
            # Has directory but no ONNX files — check config.yaml for
            # has_downloaded flag.
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8-sig") as f:
                        disk_cfg = yaml.safe_load(f) or {}
                    if disk_cfg.get("has_downloaded", False):
                        configs.append(_build_sam_config(entry, model_dir))
                        continue
                except Exception:
                    pass

        # Not yet downloaded — create stub
        configs.append(_build_stub_config(entry, model_dir))

    # ── SkyWater entry ────────────────────────────────────────────
    sw_logical = "skywater_segformer_b2_fp16"
    sw_cached = resolve_cached_path(sw_logical)
    sw_info = DOWNLOADABLE_MODELS.get(sw_logical, {})
    sw_size = sw_info.get("size_hint_mb", 48)

    if sw_cached:
        configs.append({
            "type": "skywater",
            "name": "SkyWater SegFormer-B2",
            "display_name": "SkyWater (Sky/Water/Person) [FAST]",
            "model_path": sw_cached,
            "encoder_model_path": sw_cached,
            "decoder_model_path": sw_cached,
            "logical_name": sw_logical,
            "has_downloaded": True,
        })
    elif is_huggingface_hub_available():
        configs.append({
            "type": "skywater",
            "name": "SkyWater SegFormer-B2",
            "display_name": f"SkyWater (Sky/Water/Person) [Download ~{sw_size} MB]",
            "model_path": None,
            "encoder_model_path": None,
            "decoder_model_path": None,
            "logical_name": sw_logical,
            "has_downloaded": False,
        })

    return configs


# ── Public API: SAM zip download ─────────────────────────────────────


def download_zip_model(
    logical_name: str,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Download a SAM model zip from HuggingFace and extract it.

    Parameters
    ----------
    logical_name:
        The model ``name`` from models.yaml (e.g. ``"sam_vit_b_01ec64"``).
    progress_callback:
        Optional ``(message, done_pct, total_pct)`` callback where
        *done_pct* and *total_pct* are integers 0–100.

    Returns
    -------
    dict
        Model config with absolute paths to the downloaded ONNX files.
        Keys: ``type``, ``name``, ``display_name``, ``encoder_model_path``,
        ``decoder_model_path``, ``language_encoder_path`` (optional),
        ``logical_name``, ``has_downloaded``.

    Raises
    ------
    ValueError
        If *logical_name* is not in the SAM registry.
    RuntimeError
        On download/extraction failures.
    """
    registry = _load_sam_registry()
    entry = None
    for e in registry:
        if e["name"] == logical_name:
            entry = e
            break
    if entry is None:
        raise ValueError(f"Unknown SAM model: {logical_name}")

    model_dir = os.path.join(MODELS_ROOT, logical_name)
    download_url = entry.get("download_url", "")

    if not download_url:
        raise ValueError(f"No download_url for model: {logical_name}")

    # Fast path: already downloaded?
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.isdir(model_dir):
        onnx_files = [
            f for f in os.listdir(model_dir) if f.endswith(".onnx")
        ]
        if onnx_files:
            if progress_callback:
                progress_callback(f"Using cached {logical_name}", 100, 100)
            return _build_sam_config(entry, model_dir)
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8-sig") as f:
                    disk_cfg = yaml.safe_load(f) or {}
                if disk_cfg.get("has_downloaded", False):
                    if progress_callback:
                        progress_callback(f"Using cached {logical_name}", 100, 100)
                    return _build_sam_config(entry, model_dir)
            except Exception:
                pass

    size_hint = entry.get("size_hint_mb", 0)
    display = entry.get("display_name", logical_name)

    # ── Download ──────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="colmap_sam_")
    try:
        zip_path = os.path.join(tmp_dir, "model.zip")

        # Progress reporter (throttled to ~5 FPS)
        last_update = [0.0]

        def _progress(count: int, block_size: int, total_size: int) -> None:
            now = time.time()
            if now - last_update[0] < 0.2:
                return
            last_update[0] = now
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                if progress_callback:
                    progress_callback(
                        f"Downloading {display}... {pct}%",
                        pct, 100,
                    )

        if progress_callback:
            progress_callback(
                f"Downloading {display} (~{size_hint} MB)...", 0, 100,
            )

        logger.info("Downloading %s from %s", logical_name, download_url)
        try:
            urllib.request.urlretrieve(download_url, zip_path, reporthook=_progress)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {display}: {e}\n"
                f"URL: {download_url}\n"
                "Check your network connection."
            ) from e

        # ── Extract ────────────────────────────────────────────────
        if progress_callback:
            progress_callback(f"Extracting {display}...", 100, 100)

        extract_dir = os.path.join(tmp_dir, "extract")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find the model folder (the one containing config.yaml)
        model_folder = None
        for root, _dirs, files in os.walk(extract_dir):
            if "config.yaml" in files:
                model_folder = root
                break
        if model_folder is None:
            raise RuntimeError(
                f"Could not find config.yaml in the zip for {display}.\n"
                "The zip may be corrupt or in an unexpected format."
            )

        # ── Move into place ────────────────────────────────────────
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        shutil.move(model_folder, model_dir)

        # ── Update local config ────────────────────────────────────
        local_cfg_path = os.path.join(model_dir, "config.yaml")
        local_cfg: dict = {}
        if os.path.isfile(local_cfg_path):
            with open(local_cfg_path, encoding="utf-8-sig") as f:
                local_cfg = yaml.safe_load(f) or {}
        local_cfg["has_downloaded"] = True
        local_cfg["config_file"] = os.path.abspath(local_cfg_path)
        with open(local_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(local_cfg, f, default_flow_style=False)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if progress_callback:
        progress_callback(f"Ready: {display}", 100, 100)

    return _build_sam_config(entry, model_dir)
