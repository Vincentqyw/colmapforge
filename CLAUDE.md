# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run colmapforge               # Launch the GUI
uv run colmapforge --log-level DEBUG  # Launch with debug logging
uv run ruff check .              # Lint the project
```

User-facing install/run commands are also in README.md.

> **ONNX Runtime** — `onnxruntime>=1.28` (SAM requirement) is a hard dependency.
> The CPU wheel works everywhere; on macOS it includes CoreML. GPU users (Linux /
> Windows, NVIDIA) add `uv pip install onnxruntime-gpu` after `uv sync`.
> The two wheels share the same module — installing both silently breaks GPU
> acceleration. See `onnx_utils.py` for provider selection and auto-repair logic.

## Architecture

This is a **PyQt6 desktop GUI** (single-window wizard) that preprocesses video/image data for COLMAP Structure-from-Motion. It chains four async pipeline stages — frame extraction → resize → segmentation → COLMAP database export — using `QRunnable` workers on a `QThreadPool`.

### Data flow

`MainWindow` (`main_window.py`) orchestrates everything. It builds the UI (left settings panel with 6 collapsible sections, right preview panel with mask overlay), then wires each section's state into a pipeline. `MainWindow._build()` chains workers sequentially via Qt signals: each worker's `finished` signal connects to the next stage's `start()`.

### Key modules

| Module | Role |
|--------|------|
| `app.py` | Entry point: argparse for `--log-level`, QApplication + Fusion style, hands off to `MainWindow` |
| `main_window.py` | All UI + pipeline wiring. 6 collapsible card sections in a `QGridLayout` (not `QFormLayout`), preview panel, keyboard shortcuts |
| `workers.py` | `QRunnable` workers with `QObject` signal containers for thread-safe GUI updates. All support cancellation via `_running` flag |
| `utils.py` | Pure functions: video metadata, image file collection, resize, frame index computation |
| `camera_models.py` | COLMAP `CameraModel` dataclasses (IDs 0–17, includes EUCM and EQUIRECTANGULAR beyond the standard 16). `default_params()` computes defaults from image dimensions |
| `colmap_database.py` | `ColmapDatabase` context manager: creates a COLMAP-compatible SQLite DB (WAL mode, FK enabled), stores camera params as BLOBs via `struct.pack`, converts paths to relative |
| `onnx_utils.py` | ONNX Runtime provider selection (CUDA → CoreML → CPU priority), diagnostics, and silent-overwrite auto-repair. The CPU wheel on macOS includes CoreML for Apple Silicon acceleration |
| `theme.py` | Singleton `Theme` class: `apply_theme()` loads QSS from `views/styles/`, `_make_palette()` builds Apple HIG-inspired `QPalette` for dark/light |
| `sam_backends/` | Three SAM ONNX inference backends (SAM1/SAM2/SAM3). SAM3 forces CPU EP to avoid GPU OOM. Each backend handles its own image preprocessing and prompt encoding |

### UI layout conventions

- **3-column `QGridLayout`** per section: col 0 = label (right-aligned), col 1 = input widget (stretches), col 2 = auxiliary widget
- **Collapsible sections**: header row with optional `QCheckBox` to enable/disable the section, content inside `#sectionCard` widget
- **QSS themes**: `dark.qss` and `light.qss`, use object-name selectors (`#sectionCard`, `#btnRun`, `#formLabel`) rather than type selectors
- ComboBoxes use `AdjustToMinimumContentsLengthWithIcon` + `min-width` in QSS to prevent dropdown width jumping
- See `main_window.py` for exact spacing/widget constants (`_S`, `_W`)

### SAM variant detection

In `workers.py` `_load_sam()`, the ONNX model graph is inspected to determine which backend class and prompting strategy to use (grid prompts for SAM1/2, text prompts for SAM3). See the method's implementation for the exact input-name detection logic.
