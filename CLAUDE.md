# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra cpu              # Install all dependencies (CPU ONNX Runtime)
uv sync --extra gpu              # Install all dependencies (CUDA ONNX Runtime, NVIDIA GPU)
uv run colmapforge               # Launch the GUI
uv run colmapforge run -o out/ --video vid.mp4  # Run CLI pipeline (headless)
uv run colmapforge run --help    # Show CLI options
uv run colmapforge run --list-models  # List available segmentation models
uv run colmapforge run --list-cameras # List available camera models
uv run colmapforge --log-level DEBUG  # Launch GUI with debug logging
uv run ruff check .              # Lint the project
```

User-facing install/run commands are also in README.md.

> **ONNX Runtime** — `onnxruntime>=1.28` (SAM requirement) is an **optional dependency**.
> Install via `uv sync --extra cpu` (all platforms, CoreML on macOS) or
> `uv sync --extra gpu` (Linux / Windows, NVIDIA CUDA). The two wheels share
> the same module — installing both silently breaks GPU acceleration.
> See `onnx_utils.py` for provider selection and auto-repair logic.

## Architecture

This is a **PyQt6 desktop GUI** (single-window wizard) and **CLI tool** that preprocesses video/image data for COLMAP Structure-from-Motion. It chains four async pipeline stages — frame extraction → resize → segmentation → COLMAP database export — using `QRunnable` workers on a `QThreadPool` (GUI) or a synchronous `CLIPipeline` (CLI).

### Data flow

**GUI path**: `MainWindow` (`main_window.py`) orchestrates everything. It builds the UI (left settings panel with 6 collapsible sections, right preview panel with mask overlay), then wires each section's state into a pipeline. Workers chain sequentially via Qt signals: each worker's `finished` signal connects to the next stage's `start()`.

**CLI path**: `app.py` dispatches `colmapforge run ...` to `cli.py`. `CLIPipeline` runs the same stages synchronously (no Qt), using the pure functions in `pipeline_core.py`. Progress is reported via `tqdm`.

### Key modules

| Module | Role |
|--------|------|
| `app.py` | Entry point: dispatches `colmapforge` → GUI or `colmapforge run` → CLI. Lazy Qt imports keep the CLI path lightweight. |
| `cli.py` | CLI pipeline: argparse-based argument parsing, `CLIPipeline` synchronous runner, model/camera listing. No Qt dependency. |
| `pipeline_core.py` | Pure Python pipeline functions (no Qt): `extract_frames()`, `run_sam_segmentation()`, `run_skywater_segmentation()`, `run_segmentation()`, `build_database()`, `apply_resize()`, `copy_input_images()`. Accept progress/image-done callbacks + cancel checks. Both GUI workers and CLI call these. |
| `main_window.py` | All UI + pipeline wiring. 6 collapsible card sections in a `QGridLayout` (not `QFormLayout`), preview panel, keyboard shortcuts |
| `workers.py` | `QRunnable` workers with `QObject` signal containers for thread-safe GUI updates. Delegates to `pipeline_core` functions, wrapping signal emissions around callbacks. All support cancellation via `_running` flag |
| `pipeline.py` | `PipelineConfig` dataclass + `PipelineOrchestrator` QObject that chains workers via Qt signals for the GUI path |
| `utils.py` | Pure functions: video metadata, image file collection, resize, frame index computation |
| `camera_models.py` | COLMAP `CameraModel` dataclasses (IDs 0–17, includes EUCM and EQUIRECTANGULAR beyond the standard 16). `default_params()` computes defaults from image dimensions |
| `colmap_database.py` | `ColmapDatabase` context manager: creates a COLMAP-compatible SQLite DB (WAL mode, FK enabled), stores camera params as BLOBs via `struct.pack`, converts paths to relative |
| `onnx_utils.py` | ONNX Runtime provider selection (CUDA → CoreML → CPU priority), diagnostics, and silent-overwrite auto-repair. The CPU wheel on macOS includes CoreML for Apple Silicon acceleration |
| `theme.py` | Singleton `Theme` class: `apply_theme()` loads QSS from `views/styles/`, `_make_palette()` builds Apple HIG-inspired `QPalette` for dark/light |
| `sam_backends/` | ONNX inference backends: `sam3_onnx.py` (`SegmentAnything3ONNX`, text prompts via built-in language encoder, forced CPU EP to avoid GPU OOM, language features cached per prompt), `sam_onnx.py`/`sam2_onnx.py`/`efficientvit_sam_onnx.py`/`edgetam_onnx.py` (box-promptable SAMs used by the YOLO-World cascade; EfficientViT-SAM is the fastest, takes decoder prompts in the SAM 1024-px frame, and batches all boxes in one decoder call via `predict_boxes()`; EdgeTAM is the smallest (~41 MB), image mode only — its export has no memory graphs), `yoloworld_onnx.py` (`YoloWorldONNX`, open-vocabulary detector: CLIP ViT-B/32 text encoder + YOLO-World v2 head + numpy per-class NMS, text features cached per vocabulary). SkyWater has no backend class — it is a single SegFormer session driven directly by `pipeline_core.run_skywater_segmentation()` |

### UI layout conventions

- **3-column `QGridLayout`** per section: col 0 = label (right-aligned), col 1 = input widget (stretches), col 2 = auxiliary widget
- **Collapsible sections**: header row with optional `QCheckBox` to enable/disable the section, content inside `#sectionCard` widget
- **QSS themes**: `dark.qss` and `light.qss`, use object-name selectors (`#sectionCard`, `#btnRun`, `#formLabel`) rather than type selectors
- ComboBoxes use `AdjustToMinimumContentsLengthWithIcon` + `min-width` in QSS to prevent dropdown width jumping
- See `main_window.py` for exact spacing/widget constants (`_S`, `_W`)

### Segmentation model dispatch

`pipeline_core.run_segmentation()` is the single dispatcher for both GUI and CLI: it classifies the model config via `is_skywater_config()` / `is_yoloworld_sam_config()`, auto-downloads missing weights through `model_downloader.download_model_entry()`, then runs one of three families:

- `run_skywater_segmentation()` — SegFormer, fixed sky/water/person classes
- `run_yoloworld_sam_segmentation()` — text-prompted cascade: YOLO-World detects boxes from class names, then a box-promptable SAM (`_load_box_sam_model()`: EfficientViT-SAM via the explicit `sam_backend` config key, otherwise SAM1-vs-SAM2 auto-detected from the decoder's ONNX inputs) masks each box; per-image union is written
- `run_sam_segmentation()` — SAM3, class names go straight to its language encoder

Cascade registry entries (`type: yoloworld_sam`) have no `download_url`; they list `components:` referencing hidden registry entries (detector, CLIP text encoder, SAM zip), each downloaded via its normal flow and merged by `model_downloader._cascade_config()`. Shared conventions (`output_layout()`, `mask_path_for()`/`MASK_SUFFIX`, `PipelineConfig`) also live in `pipeline_core.py`.
