# COLMAP Forge

A PyQt6-based wizard tool that prepares video and image data for [COLMAP](https://github.com/colmap/colmap) Structure-from-Motion pipelines.

## Features

- **Video frame extraction** — configurable frame rates, time ranges, intervals
- **Image resizing** — max dimension or exact size with aspect ratio preservation
- **SAM-based dynamic object masking** — segment people, vehicles, sky, water, etc.
- **18 COLMAP camera models** — from SIMPLE_PINHOLE to EQUIRECTANGULAR
- **COLMAP database.db export** — ready for one-click import into COLMAP

## Quick Start

```bash
uv sync --extra gpu          # NVIDIA + CUDA 13 (recommended)
# uv sync --extra directml  # Windows, any GPU (AMD/Intel/NVIDIA)
# uv sync --extra cpu       # pure CPU fallback

uv run colmapforge
# uv run colmapforge --log-level DEBUG
```

> **First run downloads the SkyWater model (~48 MB) from
> [huggingface.co/Realcat/skywater_seg](https://huggingface.co/Realcat/skywater_seg)
> and caches it in `~/.cache/huggingface/hub`** — no manual setup needed.
> Subsequent runs load from cache instantly.

## ONNX Runtime Backend

Three mutually-exclusive ONNX Runtime wheels exist — **install only one**:

| Extra | Package | When to use |
|---|---|---|
| `[gpu]` | `onnxruntime-gpu` | NVIDIA GPU + CUDA 13 Toolkit installed |
| `[directml]` | `onnxruntime-directml` | Windows, any GPU (AMD/Intel/NVIDIA) |
| `[cpu]` | `onnxruntime` | No GPU, or unsupported platform |

> **⚠️ Critical:** Installing more than one wheel silently breaks GPU
> acceleration — their shared `site-packages/onnxruntime/` files overwrite
> each other and CUDA disappears from `get_available_providers()` without
> error. The app auto-detects and self-heals this at startup. To prevent it
> from happening, `pyproject.toml` blocks osam's transitive `onnxruntime`
> dependency via `[[tool.uv.override-dependencies]]`.

The toolbar shows a live GPU status indicator (green = CUDA/DirectML,
orange = CPU only, red = broken). Click it for full diagnostics.

### Switching backends

```bash
uv sync --extra directml   # switch from GPU to DirectML
uv sync --extra cpu        # switch to pure CPU
uv sync --extra gpu        # switch back to GPU
```

If stale files persist, force-clean:

```powershell
uv pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
Remove-Item .venv/Lib/site-packages/onnxruntime -Recurse -Force
uv sync --extra gpu
```

### NVIDIA CUDA requirements (for `[gpu]`)

- NVIDIA driver ≥ 580 (CUDA 13)
- CUDA 13.x Toolkit on PATH (provides `cudart64_13.dll`)
- cuDNN 9 + cuBLAS 13 are bundled in the `onnxruntime-gpu` wheel

## Workflow

1. **Input** — Add video files and/or image folders
2. **Frame Extraction** — Configure frame sampling from videos
3. **Resize** — Optionally resize images (recommended max 2000px)
4. **Segmentation** — Use SAM to mask dynamic objects
5. **Camera Model** — Select COLMAP camera model (default: SIMPLE_RADIAL)
6. **Export** — Generate `database.db` + images folder

## Requirements

- Python ≥ 3.11
- PyQt6 ≥ 6.6
- OpenCV ≥ 4.8
- ONNX Runtime (see [above](#onnx-runtime-backend))
