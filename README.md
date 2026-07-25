# COLMAP Preprocessor

A PyQt6-based wizard tool that prepares video and image data for [COLMAP](https://github.com/colmap/colmap) Structure-from-Motion pipelines.

## Features

- **Video frame extraction** — configurable frame rates, time ranges, intervals
- **Image resizing** — max dimension or exact size with aspect ratio preservation
- **SAM-based dynamic object masking** — segment people, vehicles, sky, water, etc.
- **All 16 COLMAP camera models** — from SIMPLE_PINHOLE to FISHEYE
- **COLMAP database.db export** — ready for one-click import into COLMAP

## Quick Start

```bash
# One command — installs the project + GPU ONNX Runtime.
# uv sync respects the [tool.uv.override-dependencies] in pyproject.toml
# which prevents osam's `onnxruntime` (CPU) dependency from being pulled in
# and overwriting onnxruntime-gpu's binaries.
uv sync --extra gpu          # NVIDIA + CUDA 13 (recommended)
# uv sync --extra directml  # Windows, any GPU (AMD/Intel/NVIDIA)
# uv sync --extra cpu       # pure CPU fallback

# Launch the GUI
uv run colmap-prep

# Or with debug logging
uv run colmap-prep --log-level DEBUG
```

> **First run downloads the SkyWater model (~48 MB) from
> [huggingface.co/Realcat/skywater_seg](https://huggingface.co/Realcat/skywater_seg)
> and caches it in `~/.cache/huggingface/hub`** — no manual setup needed.
> Subsequent runs load from cache instantly.

## ONNX Runtime Backend

This project uses ONNX Runtime for SAM / SkyWater segmentation. Three
mutually-exclusive wheels exist — **install only one**:

| Extra | Package | When to use |
|---|---|---|
| `[gpu]` | `onnxruntime-gpu` | NVIDIA GPU + CUDA 13 Toolkit installed |
| `[directml]` | `onnxruntime-directml` | Windows, any GPU (AMD/Intel/NVIDIA) |
| `[cpu]` | `onnxruntime` | No GPU, or unsupported platform |

> **⚠️ Critical:** All three wheels install into the same
> `site-packages/onnxruntime/` directory. If you install more than one,
> their files (especially `capi/onnxruntime.dll` and
> `capi/onnxruntime_pybind11_state.pyd`) silently overwrite each other.
> CUDA will then disappear from `get_available_providers()` without any
> error — segmentation will run on CPU and you'll only notice it's slow.

### Verifying GPU is active

The toolbar shows a live GPU status indicator (green = CUDA/DirectML,
orange = CPU only, red = broken). Click it for full diagnostics. You can
also check at startup:

```bash
uv run python -c "from colmap_preprocessor.onnx_utils import diagnose; import json; print(json.dumps(diagnose(), indent=2, default=str))"
```

A healthy GPU setup reports `"active_provider": "CUDA"` and an empty
`"issues"` list.

### Switching backends

`uv sync --extra <name>` handles everything — including the
`override-dependencies` that prevent osam from pulling the CPU wheel.
Just switch extras:

```powershell
uv sync --extra directml   # switch from GPU to DirectML
uv sync --extra cpu        # switch to pure CPU
uv sync --extra gpu        # switch back to GPU
```

If `uv sync` ever leaves stale onnxruntime files behind (rare), force-clean:

```powershell
uv pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
Remove-Item .venv/Lib/site-packages/onnxruntime -Recurse -Force
uv sync --extra gpu
```

### Fixing the silent-overwrite problem

The app **auto-detects and self-heals** this problem at startup. If
the toolbar indicator shows red ("GPU: broken"), simply restart the
app — it will:

1. Detect that onnxruntime-gpu is installed but CUDA is missing
2. Prompt you to auto-fix (one click)
3. Uninstall the CPU wheel + reinstall onnxruntime-gpu with `--no-deps`
4. Exit and ask you to restart

You should never need to manually run uninstall/reinstall commands.
If auto-fix fails (e.g. no network), the error dialog shows the exact
manual commands to run:

```powershell
uv pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
Remove-Item .venv/Lib/site-packages/onnxruntime -Recurse -Force
uv sync --extra gpu
```

The `pyproject.toml` also declares a `[[tool.uv.override-dependencies]]`
rule that prevents osam's `onnxruntime` transitive dependency from being
pulled in during `uv sync` — so the problem shouldn't occur in the first
place. The auto-fix is the safety net for when it does (e.g. after a
manual `pip install` or upgrading from an older pyproject.toml).

### NVIDIA CUDA requirements (for `[gpu]`)

- NVIDIA driver ≥ 580 (CUDA 13)
- CUDA 13.x Toolkit on PATH (provides `cudart64_13.dll`)
- The `onnxruntime-gpu` wheel bundles cuDNN 9 + cuBLAS 13, so you do **not**
  need to install those separately

Verify CUDA Toolkit is reachable:

```powershell
Get-Command cudart64_13.dll  # should resolve to your CUDA\v13.x\bin\x64\...
```

## Workflow

1. **Input** — Add video files and/or image folders
2. **Frame Extraction** — Configure frame sampling from videos
3. **Resize** — Optionally resize images (recommended max 2000px)
4. **Segmentation** — Use SAM to mask dynamic objects (requires SAM model download)
5. **Camera Model** — Select COLMAP camera model (default: SIMPLE_RADIAL)
6. **Export** — Generate `database.db` + images folder

## Requirements

- Python ≥ 3.11
- PyQt6
- OpenCV
- ONNX Runtime (for SAM segmentation)
