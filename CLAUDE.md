# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run colmap-prep               # Launch the GUI
uv run colmap-prep --log-level DEBUG  # Launch with debug logging
uv run ruff check .              # Lint the project
```

## Architecture

This is a **PyQt6 desktop GUI** (single-window wizard) that preprocesses video/image data for COLMAP Structure-from-Motion. It chains four async pipeline stages — frame extraction → resize → segmentation → COLMAP database export — using `QRunnable` workers on a `QThreadPool`.

### Data flow

`MainWindow` (`main_window.py`) orchestrates everything. It builds the UI (left settings panel with 6 collapsible sections, right preview panel with mask overlay), then wires each section's state into a pipeline. When the user clicks "Build", `MainWindow._build()` chains the workers sequentially: each worker's `finished` signal connects to the next stage's `start()`.

- **Frame extraction**: `FrameExtractionWorker` uses `utils.compute_frame_indices()` + OpenCV `VideoCapture`.
- **Resize**: Applied inline during frame extraction or as a post-pass via `utils.resize_image()`.
- **Segmentation**: `SegmentationWorker` dispatches to either `SkyWaterWorker` (SegFormer-B2 ONNX, fast) or `SAMWorker` (SAM1/2/3 ONNX). SAM variant is auto-detected from ONNX model input names in `SAMWorker._load_sam()`. SAM1/2 use 3×3 grid point prompts; SAM3 uses text prompts via a language encoder.
- **Database**: `DatabaseBuildWorker` calls `ColmapDatabase.build_project()`, which creates `cameras`/`images` tables + rows using `CameraModel` defaults.

### Key modules

| Module | Role |
|--------|------|
| `app.py` | Entry point: argparse for `--log-level`, QApplication + Fusion style, hands off to `MainWindow` |
| `main_window.py` | All UI + pipeline wiring. ~1000 lines: 6 collapsible card sections in a `QGridLayout` (not `QFormLayout`), preview panel, keyboard shortcuts |
| `workers.py` | Five `QRunnable` workers with `QObject` signal containers for thread-safe GUI updates. All support cancellation via `_running` flag |
| `utils.py` | Pure functions: video metadata, image file collection, resize, frame index computation |
| `camera_models.py` | 18 COLMAP `CameraModel` dataclasses (IDs 0–17, includes EUCM and EQUIRECTANGULAR beyond the standard 16). `default_params()` computes defaults from image dimensions |
| `colmap_database.py` | `ColmapDatabase` context manager: creates a COLMAP-compatible SQLite DB (WAL mode, FK enabled), stores camera params as BLOBs via `struct.pack`, converts paths to relative |
| `theme.py` | Singleton `Theme` class: `apply_theme()` loads QSS from `views/styles/`, `_make_palette()` builds Apple HIG-inspired `QPalette` for dark/light, `theme_color()` helper for inline colors |
| `sam_backends/` | Three SAM ONNX inference backends (SAM1/SAM2/SAM3). SAM3 forces CPU EP to avoid GPU OOM. Each backend handles its own image preprocessing and prompt encoding |

### UI layout conventions

- **3-column `QGridLayout`** per section: col 0 = label (80px, right-aligned), col 1 = input widget (stretches), col 2 = auxiliary widget (fixed width)
- **Collapsible sections**: header row with optional `QCheckBox` to enable/disable the section, content inside `#sectionCard` widget
- **Design constants**: `_S` (Spacing: section gap 8px, row gap 2px, label width 80px), `_W` (widget widths: combo 180px, spin 80–120px)
- **QSS themes**: `dark.qss` and `light.qss` (~304 lines each), use object-name selectors (`#sectionCard`, `#btnRun`, `#formLabel`) rather than type selectors
- ComboBoxes use `AdjustToMinimumContentsLengthWithIcon` + `min-width: 180px` in QSS to prevent dropdown width jumping

### SAM variant detection

In `workers.py` `_load_sam()`, the ONNX model graph is inspected:
- `backbone_fpn_0` or `language_mask` in input names → SAM3
- `high_res_feats_0` in input names → SAM2
- Otherwise → SAM1

This determines which backend class and prompting strategy (grid vs. text) is used.
