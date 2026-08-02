"""Model discovery and download — SAM (zip), SkyWater (single ONNX file),
and YOLO-World + SAM cascades (composed from hidden component entries).

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


# ── Registry helpers ─────────────────────────────────────────────────


def get_model_info(logical_name: str) -> dict | None:
    """Return the registry entry for *logical_name*, or ``None``."""
    return _resolve_entry(logical_name)


def _resolve_entry(logical_name: str) -> dict | None:
    """Find a registry entry by its ``name``."""
    for entry in _load_sam_registry():
        if entry.get("name") == logical_name:
            return entry
    return None


def _is_zip_entry(entry: dict) -> bool:
    """Download format is inferred from the URL suffix."""
    return entry.get("download_url", "").endswith(".zip")


# ── Public API: model discovery ──────────────────────────────────────


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


def _cached_sam_config(entry: dict, model_dir: str) -> dict | None:
    """Return the built config when a zip model is already on disk, else None.

    "On disk" means either extracted ONNX files exist, or the local
    config.yaml says ``has_downloaded: true``.
    """
    if not os.path.isdir(model_dir):
        return None
    if any(f.endswith(".onnx") for f in os.listdir(model_dir)):
        return _build_sam_config(entry, model_dir)
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                disk_cfg = yaml.safe_load(f) or {}
            if disk_cfg.get("has_downloaded", False):
                return _build_sam_config(entry, model_dir)
        except Exception:
            pass
    return None


def _skywater_config(entry: dict, model_path: str | None) -> dict:
    """Build a SkyWater config (single-file model)."""
    size_tag = ""
    if not model_path:
        size = entry.get("size_hint_mb", 0)
        size_tag = f" [Download ~{size} MB]" if size else " [Download]"
    return {
        "type": "skywater",
        "name": entry["name"],
        "display_name": entry.get("display_name", entry["name"]) + size_tag,
        "model_path": model_path,
        "encoder_model_path": model_path,
        "decoder_model_path": model_path,
        "logical_name": entry["name"],
        "has_downloaded": model_path is not None,
    }


def _single_file_config(entry: dict, local_path: str) -> dict:
    """Config for a downloaded single-file model of any type."""
    if entry.get("type") == "skywater":
        return _skywater_config(entry, local_path)
    return {
        "type": entry.get("type", ""),
        "name": entry["name"],
        "display_name": entry.get("display_name", entry["name"]),
        "model_path": local_path,
        "logical_name": entry["name"],
        "has_downloaded": True,
    }


# Component type → sam_backend key consumed by pipeline_core._load_box_sam_model.
_SAM_BACKEND_BY_TYPE = {"efficientvit_sam": "efficientvit", "edgetam": "edgetam"}


def _multifile_paths(entry: dict) -> dict[str, str]:
    """basename → local path for a multi-file (``files:``) component entry."""
    model_dir = os.path.join(MODELS_ROOT, entry["name"])
    return {os.path.basename(u): os.path.join(model_dir, os.path.basename(u))
            for u in entry["files"]}


def _component_local_config(comp_entry: dict) -> dict | None:
    """Local (already-downloaded) paths for one cascade component, else None."""
    model_dir = os.path.join(MODELS_ROOT, comp_entry["name"])
    if comp_entry.get("files"):
        # Multi-file entry (encoder + decoder, possibly external .onnx_data).
        paths = _multifile_paths(comp_entry)
        if all(os.path.isfile(p) for p in paths.values()):
            return {
                "encoder_model_path": paths[comp_entry["encoder_filename"]],
                "decoder_model_path": paths[comp_entry["decoder_filename"]],
            }
        return None
    if _is_zip_entry(comp_entry):
        return _cached_sam_config(comp_entry, model_dir)
    filename = comp_entry.get("filename") or os.path.basename(comp_entry["download_url"])
    local = os.path.join(model_dir, filename)
    return {"model_path": local} if os.path.isfile(local) else None


def _cascade_config(entry: dict) -> dict:
    """Build a YOLO-World + SAM cascade config by resolving its components.

    Each component is a normal (hidden) registry entry; the cascade merges
    their local paths. ``has_downloaded`` is True only when every component
    is already on disk.
    """
    merged: dict = {
        "type": "yoloworld_sam",
        "name": entry["name"],
        "logical_name": entry["name"],
        "size_hint_mb": entry.get("size_hint_mb", 0),
    }
    if "default_confidence" in entry:
        merged["default_confidence"] = entry["default_confidence"]
    downloaded = True
    for comp_name in entry.get("components", []):
        comp = _resolve_entry(comp_name)
        if comp is None:
            logger.warning("Cascade %s: unknown component %s", entry["name"], comp_name)
            downloaded = False
            continue
        local = _component_local_config(comp)
        if local is None:
            downloaded = False
            continue
        ctype = comp.get("type", "")
        if ctype == "yoloworld":
            merged["yoloworld_model_path"] = local["model_path"]
        elif ctype == "text_encoder":
            merged["text_encoder_path"] = local["model_path"]
        elif ctype == "segment_anything" or ctype in _SAM_BACKEND_BY_TYPE:
            merged["encoder_model_path"] = local.get("encoder_model_path", "")
            merged["decoder_model_path"] = local.get("decoder_model_path", "")
            if ctype in _SAM_BACKEND_BY_TYPE:
                merged["sam_backend"] = _SAM_BACKEND_BY_TYPE[ctype]

    size_tag = ""
    if not downloaded:
        size = entry.get("size_hint_mb", 0)
        size_tag = f" [Download ~{size} MB]" if size else " [Download]"
    merged["display_name"] = entry.get("display_name", entry["name"]) + size_tag
    merged["has_downloaded"] = downloaded
    return merged


def discover_models() -> list[dict]:
    """Discover all available models (SkyWater + SAM3) from the registry.

    Scans the bundled models.yaml, checks which models already exist
    on disk, and returns a flat list of model config dicts ready for the
    segmentation section UI.  Order follows the registry, so the first
    entry (SkyWater) is the default selection.

    For models not yet downloaded, stub config.yaml files are created so
    they persist across restarts.
    """
    # One-time migration from the legacy anylabeling_data location.
    _migrate_old_models()
    configs: list[dict] = []

    for entry in _load_sam_registry():
        if entry.get("hidden"):
            continue  # cascade components — not directly selectable
        name = entry["name"]
        model_dir = os.path.join(MODELS_ROOT, name)

        if entry.get("type") == "yoloworld_sam":
            # Cascade: composed from hidden component entries.
            configs.append(_cascade_config(entry))
        elif _is_zip_entry(entry):
            # SAM3: zip → downloaded when ONNX files exist on disk.
            cached = _cached_sam_config(entry, model_dir)
            configs.append(cached if cached else _build_stub_config(entry, model_dir))
        else:
            # SkyWater: single-file model → downloaded when the file exists.
            local = os.path.join(model_dir, entry.get("filename", name + ".onnx"))
            configs.append(_skywater_config(entry, local if os.path.isfile(local) else None))

    return configs


# ── Public API: unified model download ───────────────────────────────


def _download_file(
    url: str, dest_path: str,
    display: str, size_hint: int,
    progress_callback: Callable[[str, int, int], None] | None,
) -> None:
    """Download *url* to *dest_path* with a throttled progress callback."""
    if progress_callback:
        progress_callback(
            f"Downloading {display} (~{size_hint} MB)...", 0, 100,
        )

    last_update = [0.0]

    def _progress(count: int, block_size: int, total_size: int) -> None:
        now = time.time()
        if now - last_update[0] < 0.2:
            return
        last_update[0] = now
        if total_size > 0 and progress_callback:
            pct = min(100, int(count * block_size * 100 / total_size))
            progress_callback(f"Downloading {display}... {pct}%", pct, 100)

    pathlib.Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", display, url)
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {display}: {e}\n"
            f"URL: {url}\n"
            "Check your network connection."
        ) from e


def _download_zip(entry: dict, model_dir: str, progress_callback) -> None:
    """Download and extract a zip model (SAM3)."""
    display = entry.get("display_name", entry["name"])
    size_hint = entry.get("size_hint_mb", 0)
    tmp_dir = tempfile.mkdtemp(prefix="colmap_sam_")
    try:
        zip_path = os.path.join(tmp_dir, "model.zip")
        _download_file(entry["download_url"], zip_path, display, size_hint, progress_callback)

        if progress_callback:
            # Reset to 0 % — zip extraction can take a while and we cannot
            # track its internal progress, so showing 100 % would mislead
            # the user into thinking the download is finished.
            progress_callback(f"Extracting {display}...", 0, 100)
        extract_dir = os.path.join(tmp_dir, "extract")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

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

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        shutil.move(model_folder, model_dir)

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


def _download_single(entry: dict, model_dir: str, progress_callback) -> None:
    """Download a single-file model (SkyWater)."""
    filename = entry.get("filename") or os.path.basename(entry["download_url"])
    dest = os.path.join(model_dir, filename)
    _download_file(
        entry["download_url"], dest,
        entry.get("display_name", entry["name"]),
        entry.get("size_hint_mb", 0), progress_callback,
    )


def download_model_entry(
    logical_name: str,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Download a model from the registry, returning its config dict.

    The download format (zip + extract vs single file) is inferred from the
    entry's ``download_url`` suffix.  Already-downloaded models short-circuit.

    Parameters
    ----------
    logical_name:
        The model ``name`` from models.yaml.
    progress_callback:
        Optional ``(message, done_pct, total_pct)`` callback (0–100).

    Raises
    ------
    ValueError
        If *logical_name* is not in the registry.
    RuntimeError
        On download/extraction failures.
    """
    entry = _resolve_entry(logical_name)
    if entry is None:
        raise ValueError(f"Unknown model: {logical_name}")

    if entry.get("type") == "yoloworld_sam":
        # Cascade: download every component, then merge their paths.
        for comp_name in entry.get("components", []):
            download_model_entry(comp_name, progress_callback=progress_callback)
        cfg = _cascade_config(entry)
        if progress_callback:
            progress_callback(f"Ready: {entry.get('display_name', logical_name)}", 100, 100)
        return cfg

    if entry.get("files"):
        # Multi-file component (encoder + decoder, possibly external data).
        display = entry.get("display_name", logical_name)
        for url, dest in zip(entry["files"], _multifile_paths(entry).values()):
            if os.path.isfile(dest):
                continue
            _download_file(url, dest, f"{display} ({os.path.basename(dest)})",
                           entry.get("size_hint_mb", 0), progress_callback)
        return {
            "type": entry.get("type", ""),
            "name": entry["name"],
            "display_name": display,
            "logical_name": entry["name"],
            "has_downloaded": True,
            **(_component_local_config(entry) or {}),
        }

    if not entry.get("download_url"):
        raise ValueError(f"No download_url for model: {logical_name}")

    model_dir = os.path.join(MODELS_ROOT, logical_name)

    if _is_zip_entry(entry):
        cached = _cached_sam_config(entry, model_dir)
        if cached:
            if progress_callback:
                progress_callback(f"Using cached {logical_name}", 100, 100)
            return cached
        _download_zip(entry, model_dir, progress_callback)
        if progress_callback:
            progress_callback(f"Ready: {entry['display_name']}", 100, 100)
        return _build_sam_config(entry, model_dir)
    else:
        filename = entry.get("filename") or os.path.basename(entry["download_url"])
        local = os.path.join(model_dir, filename)
        if os.path.isfile(local):
            if progress_callback:
                progress_callback(f"Using cached {logical_name}", 100, 100)
        else:
            _download_single(entry, model_dir, progress_callback)
            if progress_callback:
                progress_callback(f"Ready: {entry['display_name']}", 100, 100)
        return _single_file_config(entry, local)
