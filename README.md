# COLMAP Forge

<p align="center">
  <img src="assets/colmapforge-gui.jpg" alt="COLMAP Forge GUI" width="1000">
</p>

A PyQt6 desktop wizard and CLI tool that prepares video and image data for [COLMAP](https://github.com/colmap/colmap) Structure-from-Motion — frame extraction, resizing, SAM-based dynamic object masking, and `database.db` export.

## Quick Start

```bash
git clone https://github.com/Vincentqyw/colmapforge.git
cd colmapforge

uv sync --extra cpu           # install with CPU ONNX Runtime (works everywhere)
uv run colmapforge            # launch the GUI
```

### CLI Mode

Run the preprocessing pipeline headlessly (no GUI required):

```bash
# Image-only, no segmentation
uv run colmapforge run -o out/ --image-dir photos/

# Full pipeline: video → max-dim resize → SkyWater segmentation → database
uv run colmapforge run -o out/ --video vid.mp4 --resize \
  --resize-mode max_dim --resize-max-dim 2000 \
  --seg-model skywater_segformer_b2_fp16 --seg-classes sky water

# SAM3 text-prompt segmentation
uv run colmapforge run -o out/ --video vid.mp4 \
  --seg-model sam3_vit_h_20260220 --seg-classes person car \
  --seg-confidence 0.4

# List available models and camera models
uv run colmapforge run --list-models
uv run colmapforge run --list-cameras

# Progress bars are shown automatically via tqdm
uv run colmapforge run -o out/ --video vid.mp4 --seg-model ...
```

> **macOS users** — CoreML (Apple Silicon GPU / Neural Engine) is included in the
> CPU wheel — no extra setup needed. **Linux / Windows users with NVIDIA
> GPUs** — use `uv sync --extra gpu` for CUDA acceleration.

> **Zero-config models** — SAM3 + SkyWater download on first use and
> cache to `~/.colmapforge/models/`.

## Workflow

1. **Input** — Add video files and/or image folders
2. **Frame Extraction** — Sample frames by interval or target FPS
3. **Resize** — Downscale with max-dimension or fixed-factor modes
4. **Segmentation** — SAM3 (text prompts) or SkyWater (fast sky/water/people)
5. **Camera** — Choose from 18 COLMAP camera models (SIMPLE_RADIAL default)
6. **Build** — One-click `database.db` + masked images export

## Features

- **4-stage pipeline** — extraction → resize → masking → database, chained async via `QThreadPool`
- **GUI + CLI** — full-featured desktop GUI or headless CLI for scripts/automation
- **Zero-config models** — SAM3, YOLO-World + SAM cascades, and SkyWater auto-download on first run
- **SAM3** — class names go straight to its native language encoder as text prompts
- **YOLO-World + SAM1/2** — open-vocabulary detection from class names, each box masked by MobileSAM or SAM2.1-Tiny; text prompts at a fraction of SAM3's cost
- **SkyWater** — SegFormer-B2 for fast sky/water/person segmentation (~48 MB)
- **18 camera models** — SIMPLE_PINHOLE through EQUIRECTANGULAR
- **GPU auto-detect** — CUDA → CoreML → CPU fallback; toolbar indicator shows active backend
- **Dark / Light themes** — Apple HIG-inspired `QPalette` + QSS, toggle with Ctrl+T
- **Preview panel** — real-time mask overlay with opacity control

## Available Models

All models are ONNX-based and download on first use
(`colmapforge run --list-models` shows the same registry from the CLI).

| Model | Type | Size |
|-------|------|------|
| [SkyWater](https://github.com/Vincentqyw/skywater_seg) SegFormer-B2 (FP16) | fixed sky/water/person | ~48 MB |
| YOLO-World + MobileSAM | text prompts (detect → box-mask) | ~720 MB |
| YOLO-World + SAM2.1-Tiny | text prompts (detect → box-mask) | ~835 MB |
| YOLO-World + [EfficientViT-SAM-L0](https://huggingface.co/mit-han-lab/efficientvit-sam) | text prompts (detect → box-mask, fastest) | ~815 MB |
| YOLO-World + [EdgeTAM](https://huggingface.co/onnx-community/EdgeTAM-ONNX) | text prompts (detect → box-mask, smallest SAM) | ~715 MB |
| SAM3 ViT-H | text prompts | ~3.0 GB |

## ONNX Runtime Backend

<details>
<summary>GPU setup details — click to expand</summary>

ONNX Runtime is not a hard dependency — install via project extras:

```bash
uv sync --extra cpu    # CPU (all platforms, CoreML on macOS)
uv sync --extra gpu    # CUDA (Linux / Windows, NVIDIA GPU)
```

**macOS** users get CoreML (Apple Silicon GPU / Neural Engine) automatically
from the CPU wheel — no extra setup needed.

For **NVIDIA GPU** acceleration (Linux / Windows):

```bash
uv sync --extra gpu
```

Two wheels exist — **install only one** (they share the same module and
silently overwrite each other's binaries):

| Package | Platform | Acceleration |
|---------|----------|-------------|
| `onnxruntime` (default) | macOS / Linux / Windows | CoreML on macOS, CPU elsewhere |
| `onnxruntime-gpu` | Linux / Windows (NVIDIA) | CUDA |

The app auto-detects the best provider: **CUDA → CoreML → CPU**.
A toolbar indicator shows the active backend (green = GPU, orange = CPU).

**Switching backends:**

```bash
uv sync --extra cpu    # → CPU / CoreML
uv sync --extra gpu    # → CUDA
```

**Force-clean stale files (if providers are wrong):**

```bash
uv pip uninstall onnxruntime onnxruntime-gpu
rm -rf .venv/lib/python*/site-packages/onnxruntime/
rm -rf .venv/lib/python*/site-packages/onnxruntime*.dist-info/
uv sync --extra gpu    # or --extra cpu
```

**NVIDIA CUDA requirements:**

- Driver ≥ 580 (CUDA 13)
- CUDA 13.x Toolkit on PATH
- cuDNN 9 + cuBLAS 13 bundled in `onnxruntime-gpu`

</details>

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + O` | Add images |
| `Ctrl + B` | Build COLMAP database |
| `Ctrl + T` | Toggle dark/light theme |

## Requirements

- Python ≥ 3.11
- PyQt6 ≥ 6.6
- OpenCV ≥ 4.8
- ONNX Runtime ≥ 1.28 (see [above](#onnx-runtime-backend))
- `tqdm` ≥ 4.66 (CLI progress bars)

## License

MIT — see [LICENSE](LICENSE) for details.
