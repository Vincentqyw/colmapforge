"""
CLI pipeline runner — headless alternative to the PyQt6 GUI.

Usage::

    colmapforge run --video vid.mp4 --output-dir out/ --seg-model skywater_segformer_b2_fp16 --seg-classes sky water

    colmapforge run --image-dir photos/ --output-dir out/ --no-seg --camera-model PINHOLE

    colmapforge run --help  # full option listing
"""

from __future__ import annotations

import argparse
import logging
import os

from .camera_models import CAMERA_MODELS, CAMERA_MODEL_BY_NAME, DEFAULT_CAMERA_MODEL_ID
from .model_downloader import discover_models, download_model_entry
from .pipeline_core import (
    PipelineConfig,
    apply_resize,
    build_database,
    copy_input_images,
    extract_frames,
    output_layout,
    run_segmentation,
)
from .utils import collect_image_files, colmap_gui_command

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``run`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="colmapforge run",
        description="COLMAP Forge — preprocess video/images for COLMAP SfM (CLI mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )

    # ── Input ──
    grp_in = parser.add_argument_group("Input")
    grp_in.add_argument("--video", nargs="+", default=[], metavar="PATH",
                        help="Video file(s) to extract frames from")
    grp_in.add_argument("--image-dir", nargs="+", default=[], metavar="DIR",
                        help="Image folder(s) to process")

    # ── Output ──
    grp_out = parser.add_argument_group("Output")
    grp_out.add_argument("--output-dir", "-o", default=None, metavar="DIR",
                         help="Output directory (required unless --list-* is used)")

    # ── Frame Extraction ──
    grp_ext = parser.add_argument_group("Frame Extraction")
    grp_ext.add_argument("--no-extract", action="store_true",
                         help="Skip frame extraction (use existing images)")
    grp_ext.add_argument("--extract-method", default="interval",
                         choices=["interval", "target_fps"],
                         help="Extraction method (default: interval)")
    grp_ext.add_argument("--extract-interval", type=int, default=60, metavar="N",
                         help="Extract every N frames (method=interval, default: 60)")
    grp_ext.add_argument("--extract-target-fps", type=float, default=2.0, metavar="FPS",
                         help="Target FPS for extraction (method=target_fps, default: 2.0)")
    grp_ext.add_argument("--extract-max-frames", type=int, default=None, metavar="N",
                         help="Maximum frames to extract per video")
    grp_ext.add_argument("--extract-format", default=".jpg", choices=[".jpg", ".png"],
                         help="Output image format (default: .jpg)")
    grp_ext.add_argument("--extract-jpg-quality", type=int, default=95, metavar="Q",
                         help="JPEG quality 1–100 (default: 95)")

    # ── Resize ──
    grp_rz = parser.add_argument_group("Resize")
    grp_rz.add_argument("--resize", action="store_true", default=False,
                        help="Enable image resize (disabled by default)")
    grp_rz.add_argument("--resize-mode", default="max_dim",
                        choices=["max_dim", "downscale"],
                        help="Resize mode (default: max_dim)")
    grp_rz.add_argument("--resize-max-dim", type=int, default=2000, metavar="PX",
                        help="Max dimension in pixels (mode=max_dim, default: 2000)")
    grp_rz.add_argument("--resize-factor", type=int, default=4, metavar="N",
                        choices=[1, 2, 4, 8],
                        help="Downscale factor (mode=downscale, default: 4)")

    # ── Segmentation ──
    grp_seg = parser.add_argument_group("Segmentation")
    grp_seg.add_argument("--no-seg", action="store_true",
                         help="Skip segmentation step")
    grp_seg.add_argument("--seg-model", default=None, metavar="NAME",
                         help="Model name from registry (e.g. skywater_segformer_b2_fp16, "
                              "yoloworld_mobile_sam, sam3_vit_h_20260220)")
    grp_seg.add_argument("--seg-classes", nargs="+", default=[], metavar="CLASS",
                         help="Target classes to mask (e.g. sky water person)")
    grp_seg.add_argument("--seg-confidence", type=float, default=None, metavar="F",
                         help="Confidence threshold 0.0–1.0: SAM3 mask score or "
                              "YOLO-World box score (default: model-specific — "
                              "0.3 for SAM3, 0.1 for YOLO-World cascades)")
    grp_seg.add_argument("--list-models", action="store_true",
                         help="List available segmentation models and exit")

    # ── Camera ──
    grp_cam = parser.add_argument_group("Camera Model")
    grp_cam.add_argument("--camera-model", default=str(DEFAULT_CAMERA_MODEL_ID), metavar="ID_OR_NAME",
                         help="Camera model ID (0–17) or name, e.g. SIMPLE_RADIAL (default: 2)")
    grp_cam.add_argument("--camera-params", type=float, nargs="*", default=None, metavar="F",
                         help="Custom camera intrinsic parameters (computed from image dims if omitted)")
    grp_cam.add_argument("--list-cameras", action="store_true",
                         help="List available camera models and exit")

    # ── General ──
    grp_gen = parser.add_argument_group("General")
    grp_gen.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging level (default: INFO)")
    grp_gen.add_argument("--quiet", "-q", action="store_true",
                         help="Suppress progress bars")

    return parser


def _epilog() -> str:
    return (
        "Examples:\n"
        "  # Full pipeline: video → frames → resize → SkyWater segmentation → DB\n"
        "  colmapforge run -o out/ --video vid.mp4 "
        "--resize-mode max_dim --resize-max-dim 2000 "
        "--seg-model skywater_segformer_b2_fp16 --seg-classes sky water\n\n"
        "  # Image-only, no segmentation\n"
        "  colmapforge run -o out/ --image-dir photos/ --no-seg\n\n"
        "  # SAM3 text-prompt segmentation\n"
        "  colmapforge run -o out/ --video vid.mp4 "
        "--seg-model sam3_vit_h_20260220 --seg-classes person car "
        "--seg-confidence 0.4\n\n"
        "  # YOLO-World + SAM cascade (text prompts, faster than SAM3)\n"
        "  colmapforge run -o out/ --video vid.mp4 "
        "--seg-model yoloworld_mobile_sam --seg-classes person car\n\n"
        "  # List available models / cameras\n"
        "  colmapforge run --list-models\n"
        "  colmapforge run --list-cameras\n\n"
        "Launch the GUI by running 'colmapforge' without arguments."
    )


# ═══════════════════════════════════════════════════════════════════════
# Model / camera listing helpers
# ═══════════════════════════════════════════════════════════════════════

def print_models() -> None:
    """Print available segmentation models to stdout."""
    models = discover_models()
    if not models:
        print("No models found in registry.")
        return
    print(f"{'Model':<40} {'Type':<20} {'Downloaded':<12}")
    print("-" * 72)
    for m in models:
        name = m.get("display_name", m.get("name", "?"))
        mtype = m.get("type", "?")
        downloaded = "Yes" if m.get("has_downloaded") else "No"
        print(f"{name:<40} {mtype:<20} {downloaded:<12}")


def print_cameras() -> None:
    """Print available camera models to stdout."""
    print(f"{'ID':<4} {'Name':<30} {'Params':<30} {'Fisheye':<8}")
    print("-" * 74)
    for cid in sorted(CAMERA_MODELS):
        m = CAMERA_MODELS[cid]
        params = ", ".join(m.params)
        fisheye = "Yes" if m.is_fisheye else "No"
        print(f"{cid:<4} {m.name:<30} {params:<30} {fisheye:<8}")


def resolve_camera_model(spec: str) -> tuple[int, str]:
    """Resolve a camera model from an ID or name string.

    Returns (model_id, model_name). Raises ValueError on failure.
    """
    # Try as integer ID
    try:
        cid = int(spec)
        if cid in CAMERA_MODELS:
            return cid, CAMERA_MODELS[cid].name
    except ValueError:
        pass

    # Try as name (case-insensitive)
    upper = spec.upper()
    if upper in CAMERA_MODEL_BY_NAME:
        return CAMERA_MODEL_BY_NAME[upper].model_id, upper

    # Suggest close matches
    names = list(CAMERA_MODEL_BY_NAME.keys())
    suggestions = [n for n in names if upper[:4] in n]
    hint = f" Did you mean: {', '.join(suggestions[:5])}?" if suggestions else ""
    raise ValueError(f"Unknown camera model: {spec!r}.{hint}")


def resolve_model_config(model_name: str | None, seg_enabled: bool) -> dict | None:
    """Resolve a model name to its config dict.

    Returns None if *seg_enabled* is ``False`` or *model_name* is ``None``.
    Raises ValueError if the model is not found in the registry.
    """
    if not seg_enabled or model_name is None:
        return None

    models = discover_models()
    for m in models:
        if m.get("name") == model_name or m.get("logical_name") == model_name:
            return m

    names = [m.get("name", "?") for m in models]
    suggestions = [n for n in names if model_name.lower() in n.lower()]
    hint = f" Did you mean: {', '.join(suggestions[:5])}?" if suggestions else ""
    raise ValueError(
        f"Unknown model: {model_name!r}.{hint}\n"
        f"Run 'colmapforge run --list-models' to see available models."
    )


# ═══════════════════════════════════════════════════════════════════════
# Model pre-download  (colmapforge download ...)
# ═══════════════════════════════════════════════════════════════════════

def build_download_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``download`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="colmapforge download",
        description="Pre-download segmentation models so runs never wait on the network. "
                    "Already-downloaded models (and cascade components) are skipped.",
        epilog="Examples:\n"
               "  colmapforge download --all\n"
               "  colmapforge download yoloworld_edgetam sam3_vit_h_20260220\n"
               "  colmapforge download --list\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("models", nargs="*", metavar="NAME",
                        help="Model name(s) from the registry (see --list)")
    parser.add_argument("--all", action="store_true",
                        help="Download every model in the registry")
    parser.add_argument("--list", action="store_true", dest="list_models",
                        help="List models with download status and exit")
    return parser


def _download_progress(msg: str, done: int, total: int) -> None:
    print(f"\r  {msg[:76]:<76}", end="", flush=True)


def run_download(argv: list[str] | None = None) -> int:
    """Entry point for ``colmapforge download``. Returns 0 on success."""
    parser = build_download_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    models = discover_models()
    by_name = {m["name"]: m for m in models}

    if args.list_models or (not args.models and not args.all):
        print_models()
        if not args.list_models:
            print("\nUsage: colmapforge download --all | NAME [NAME ...]")
        return 0

    names = [m["name"] for m in models] if args.all else args.models
    failed: list[str] = []
    for name in names:
        cfg = by_name.get(name)
        display = (cfg or {}).get("display_name", name)
        if cfg and cfg.get("has_downloaded"):
            print(f"[skip] {display} — already downloaded")
            continue
        size = (cfg or {}).get("size_hint_mb", 0)
        tag = f" (~{size} MB)" if size else ""
        print(f"[get ] {display}{tag}")
        try:
            # Cached cascade components short-circuit inside, so partially
            # downloaded models only fetch what is missing.
            download_model_entry(name, progress_callback=_download_progress)
            print(f"\r  {'':76}\r[done] {display}")
        except Exception as e:
            print(f"\r  {'':76}\r[fail] {display}: {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        print("Run 'colmapforge download --list' to see available models.")
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════

class CLIPipeline:
    """Synchronous COLMAP Forge pipeline (no Qt dependency)."""

    def __init__(self, config: PipelineConfig, quiet: bool = False) -> None:
        self._config = config
        self._quiet = quiet

    def run(self) -> str:
        """Run the complete pipeline.

        Returns the path to the created database.
        """
        import shutil

        cfg = self._config
        images_dir, masks_dir, db_path = output_layout(cfg.output_dir)

        # ── Check if images already exist ──
        images_exist = os.path.isdir(images_dir) and collect_image_files([images_dir])

        if images_exist:
            # Re-build: wipe existing outputs
            self._log("Existing images found — rebuilding outputs.")
            if os.path.isdir(masks_dir):
                shutil.rmtree(masks_dir)
            if os.path.isfile(db_path):
                os.remove(db_path)

            if cfg.seg_enabled and not cfg.seg_target_classes:
                raise ValueError(
                    "Segmentation enabled but no target classes selected. "
                    "Use --seg-classes to specify classes."
                )

            os.makedirs(masks_dir, exist_ok=True)
            # Re-run segmentation on existing images
            if cfg.seg_enabled and cfg.seg_target_classes and cfg.seg_model_config:
                self._run_segmentation(images_dir, masks_dir)
            self._run_database(images_dir, db_path)
            return db_path

        # ── Fresh run ──
        if not cfg.video_paths and not cfg.image_paths:
            raise ValueError("No input specified. Use --video or --image-dir.")

        os.makedirs(images_dir, exist_ok=True)

        # Step 1: Copy input images
        if cfg.image_paths:
            self._log(f"Copying images from {len(cfg.image_paths)} folder(s)...")
            copy_input_images(cfg.image_paths, images_dir)

        # Step 2: Extract frames (or apply resize to copied images)
        if cfg.video_paths and cfg.extract_enabled:
            self._run_extraction(images_dir)
        elif not cfg.video_paths:
            # Image-only input — apply resize
            if cfg.resize_enabled:
                self._run_resize(images_dir)

        # Step 3: Segmentation
        if cfg.seg_enabled and cfg.seg_target_classes and cfg.seg_model_config:
            os.makedirs(masks_dir, exist_ok=True)
            self._run_segmentation(images_dir, masks_dir)

        # Step 4: Database
        self._run_database(images_dir, db_path)

        return db_path

    # ── stage helpers ──

    def _run_extraction(self, images_dir: str) -> None:
        cfg = self._config
        self._log(f"Extracting frames from {len(cfg.video_paths)} video(s)...")
        paths = extract_frames(
            video_paths=cfg.video_paths,
            output_dir=images_dir,
            method=cfg.extract_method,
            interval=cfg.extract_interval,
            target_fps=cfg.extract_target_fps,
            max_frames=cfg.extract_max_frames,
            **cfg.resize_kwargs(),
            output_format=cfg.extract_format,
            jpg_quality=cfg.extract_jpg_quality,
            progress_cb=self._progress_cb("Extract"),
        )
        self._log(f"  Extracted {len(paths)} frames.")

    def _run_resize(self, images_dir: str) -> None:
        cfg = self._config
        self._log(f"Resizing images (mode={cfg.resize_mode})...")
        apply_resize(
            images_dir=images_dir,
            **cfg.resize_kwargs(),
            progress_cb=self._progress_cb("Resize"),
        )
        self._log("  Resize complete.")

    def _run_segmentation(self, images_dir: str, masks_dir: str) -> None:
        cfg = self._config
        image_paths = collect_image_files([images_dir])
        self._log(
            f"Running segmentation on {len(image_paths)} images "
            f"({cfg.seg_model_config.get('display_name', cfg.seg_model_config.get('name', '?'))})..."
        )
        mask_paths = run_segmentation(
            image_paths=image_paths,
            mask_output_dir=masks_dir,
            model_config=cfg.seg_model_config or {},
            target_classes=cfg.seg_target_classes,
            confidence_threshold=cfg.seg_confidence,
            progress_cb=self._progress_cb("Seg"),
        )
        self._log(f"  Generated {len(mask_paths)} masks.")

    def _run_database(self, images_dir: str, db_path: str) -> None:
        cfg = self._config
        self._log("Building COLMAP database...")
        build_database(
            image_dir=images_dir,
            db_path=db_path,
            camera_model_id=cfg.camera_model_id,
            camera_params=cfg.camera_params if cfg.camera_params else None,
            progress_cb=self._progress_cb("DB"),
        )
        self._log(f"  Database: {db_path}")

    # ── progress ──

    def _progress_cb(self, label: str):
        """Return a progress callback function.

        Uses ``tqdm`` when available and *quiet* is ``False``; otherwise
        falls back to logger info at each 10 % boundary.
        """
        try:
            from tqdm import tqdm as _tqdm_mod
            _TQDM_AVAILABLE = True
        except ImportError:
            _TQDM_AVAILABLE = False

        if self._quiet or not _TQDM_AVAILABLE:
            def _cb(pct: int, msg: str) -> None:
                if pct % 10 == 0 or pct >= 100:
                    logger.info("[%s %3d%%] %s", label, pct, msg)
            return _cb

        # tqdm-backed progress
        pbar = None
        last_pct = [0]

        def _cb(pct: int, msg: str) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = _tqdm_mod(total=100, desc=label, unit="%",
                                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")
            delta = pct - last_pct[0]
            if delta > 0:
                pbar.update(delta)
                last_pct[0] = pct
            pbar.set_postfix_str(msg[:50])
            if pct >= 100:
                pbar.close()

        return _cb

    @staticmethod
    def _log(msg: str) -> None:
        logger.info(msg)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def run_cli(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the pipeline.

    Returns 0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set up logging early
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Info-only flags ──
    if args.list_models:
        print_models()
        return 0

    if args.list_cameras:
        print_cameras()
        return 0

    # ── Validate ──
    if not args.output_dir:
        parser.error("--output-dir/-o is required (unless using --list-models or --list-cameras)")

    if not args.video and not args.image_dir:
        parser.error("At least one of --video or --image-dir is required.")

    try:
        camera_id, camera_name = resolve_camera_model(args.camera_model)
    except ValueError as e:
        parser.error(str(e))

    # ── Build config ──
    seg_enabled = not args.no_seg
    model_config = None
    if seg_enabled:
        try:
            model_config = resolve_model_config(args.seg_model, seg_enabled=True)
        except ValueError as e:
            parser.error(str(e))
        if model_config is None:
            parser.error(
                "Segmentation is enabled but no --seg-model specified.\n"
                "Use --seg-model to pick a model, or --no-seg to skip segmentation."
            )

    config = PipelineConfig(
        video_paths=list(args.video),
        image_paths=list(args.image_dir),
        output_dir=args.output_dir,

        extract_enabled=not args.no_extract,
        extract_method=args.extract_method,
        extract_interval=args.extract_interval,
        extract_target_fps=args.extract_target_fps,
        extract_max_frames=args.extract_max_frames,
        extract_format=args.extract_format,
        extract_jpg_quality=args.extract_jpg_quality,

        resize_enabled=args.resize,
        resize_mode=args.resize_mode,
        resize_max_dim=args.resize_max_dim,
        resize_factor=args.resize_factor,

        seg_enabled=seg_enabled,
        seg_model_config=model_config,
        seg_target_classes=list(args.seg_classes),
        # No explicit threshold → the model's calibrated default (0.3 unless
        # the registry entry says otherwise, e.g. YOLO-World's 0.1).
        seg_confidence=args.seg_confidence if args.seg_confidence is not None
        else (model_config or {}).get("default_confidence", 0.3),

        camera_model_id=camera_id,
        camera_params=list(args.camera_params) if args.camera_params else [],
    )

    # ── Run ──
    logger.info("COLMAP Forge CLI — starting pipeline")
    logger.info("  Input: %d video(s), %d image folder(s)",
                len(config.video_paths), len(config.image_paths))
    logger.info("  Output: %s", config.output_dir)
    logger.info("  Camera: %s (ID %d)", camera_name, camera_id)
    if config.seg_enabled and config.seg_model_config:
        logger.info("  Segmentation: %s → %s",
                    config.seg_model_config.get("display_name", "?"),
                    ", ".join(config.seg_target_classes))

    try:
        pipeline = CLIPipeline(config, quiet=args.quiet)
        db_path = pipeline.run()
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        if args.log_level == "DEBUG":
            logger.exception("Traceback:")
        return 1

    # ── Summary ──
    images_dir, masks_dir, _ = output_layout(config.output_dir)
    image_count = len(collect_image_files([images_dir])) if os.path.isdir(images_dir) else 0
    mask_count = len(collect_image_files([masks_dir])) if os.path.isdir(masks_dir) else 0

    print()
    print("═" * 60)
    print("  COLMAP Forge — Pipeline Complete")
    print("═" * 60)
    print(f"  Output directory : {config.output_dir}")
    print(f"  Images           : {image_count}")
    print(f"  Masks            : {mask_count}")
    print(f"  Database         : {db_path}")
    print(f"  Camera model     : {camera_name} (ID {camera_id})")
    print("═" * 60)
    print()
    print("Next step — run COLMAP:")
    print("  " + " ".join(colmap_gui_command(
        db_path, images_dir, masks_dir if mask_count > 0 else None)))
    print()

    return 0
