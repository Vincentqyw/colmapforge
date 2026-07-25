"""
Utility functions: video analysis, frame extraction, image resize, I/O helpers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv",
    ".flv", ".m4v", ".mpg", ".mpeg", ".3gp", ".mts", ".m2ts", ".ts",
}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".gif", ".pgm", ".ppm", ".pbm",
}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def analyze_video(video_path: str | Path) -> dict:
    """Return fps, total_frames, duration_seconds, width, height, etc."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()

        if total_frames <= 0:
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = cap.get(cv2.CAP_PROP_FPS)

        duration = total_frames / fps if fps > 0 else 0.0
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        return {
            "path": video_path, "fps": float(fps),
            "total_frames": total_frames, "duration_seconds": duration,
            "width": width, "height": height,
            "fourcc": fourcc, "file_size_mb": file_size,
        }
    finally:
        cap.release()


def analyze_videos(video_paths: list[str]) -> list[dict]:
    """Analyze multiple video files, skipping failures."""
    results = []
    for path in video_paths:
        try:
            results.append(analyze_video(path))
        except RuntimeError as e:
            logger.warning("Skipping %s: %s", path, e)
    return results


def collect_image_files(paths: list[str], recursive: bool = False) -> list[str]:
    """Collect image files from a list of files/directories."""
    result: list[str] = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            if is_image_file(p):
                result.append(p)
        elif os.path.isdir(p):
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        if is_image_file(fp):
                            result.append(fp)
            else:
                for f in sorted(os.listdir(p)):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and is_image_file(fp):
                        result.append(fp)
    return sorted(set(result))


def resize_image(
    image: np.ndarray,
    max_dim: int | None = None,
    width: int | None = None,
    height: int | None = None,
    keep_aspect: bool = True,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize with various strategies."""
    h, w = image.shape[:2]
    if max_dim is not None:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
    elif width is not None and height is not None:
        if keep_aspect:
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = width, height
    elif width is not None:
        new_w = width
        new_h = int(h * width / w) if keep_aspect else h
    elif height is not None:
        new_w = int(w * height / h) if keep_aspect else w
        new_h = height
    else:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def compute_frame_indices(
    total_frames: int, fps: float,
    method: str = "interval", interval: int = 1,
    target_fps: float = 1.0, start_seconds: float = 0.0,
    end_seconds: float | None = None, max_frames: int | None = None,
) -> list[int]:
    """Compute which 0-based frame indices to extract from a video."""
    if method == "target_fps":
        interval = max(1, int(fps / target_fps))
        frames = list(range(0, total_frames, interval))
    elif method == "time_range":
        start_frame = max(0, int(start_seconds * fps))
        end_frame = total_frames if end_seconds is None else min(total_frames, int(end_seconds * fps))
        frames = list(range(start_frame, end_frame, max(1, interval)))
    else:
        frames = list(range(0, total_frames, max(1, interval)))

    if max_frames and max_frames > 0 and len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames


def get_output_image_path(
    output_dir: str, original_path: str,
    index: int | None = None, extension: str = ".jpg",
) -> str:
    base = Path(original_path).stem
    name = f"{base}_frame_{index:06d}{extension}" if index is not None else f"{base}{extension}"
    return os.path.join(output_dir, name)
