"""
Complete COLMAP camera model definitions — 18 models (IDs 0–17).

Based on: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
(models 0–15), plus EUCM and EQUIRECTANGULAR.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraModel:
    """Definition of a single COLMAP camera model."""

    model_id: int
    name: str
    params: list[str]
    description: str = ""
    is_fisheye: bool = False

    @property
    def num_params(self) -> int:
        return len(self.params)

    def default_params(self, width: int, height: int) -> list[float]:
        """Sensible defaults — principal point at center, f = max(w,h) * 1.2."""
        cx = width / 2.0
        cy = height / 2.0
        f = max(width, height) * 1.2
        defaults = {
            "f": f, "fx": f, "fy": f, "cx": cx, "cy": cy,
            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
            "k5": 0.0, "k6": 0.0, "p1": 0.0, "p2": 0.0,
            "omega": 0.7, "sx1": 0.0, "sy1": 0.0,
            "sx2": 0.0, "sy2": 0.0,
            "alpha": 0.5, "beta": 1.0,
        }
        return [defaults[p] for p in self.params]


# ---------------------------------------------------------------------------
# Complete registry
# ---------------------------------------------------------------------------
CAMERA_MODELS: dict[int, CameraModel] = {
    0: CameraModel(0, "SIMPLE_PINHOLE", ["f", "cx", "cy"],
                   "Single focal length, principal point. No distortion."),
    1: CameraModel(1, "PINHOLE", ["fx", "fy", "cx", "cy"],
                   "Two focal lengths, principal point. No distortion."),
    2: CameraModel(2, "SIMPLE_RADIAL", ["f", "cx", "cy", "k1"],
                   "Pinhole + one radial distortion term (k1)."),
    3: CameraModel(3, "RADIAL", ["f", "cx", "cy", "k1", "k2"],
                   "Pinhole + two radial distortion terms (k1, k2)."),
    4: CameraModel(4, "OPENCV", ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
                   "OpenCV model: 2 radial (k1,k2) + 2 tangential (p1,p2)."),
    5: CameraModel(5, "OPENCV_FISHEYE", ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
                   "OpenCV fisheye: 4 radial terms (k1–k4).", is_fisheye=True),
    6: CameraModel(6, "FULL_OPENCV",
                   ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
                   "Full OpenCV: 6 radial + 2 tangential. Best for high-accuracy cals."),
    7: CameraModel(7, "FOV", ["fx", "fy", "cx", "cy", "omega"],
                   "Field-of-View (Devernay & Faugeras). Single ω parameter.", is_fisheye=True),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", ["f", "cx", "cy", "k1"],
                   "Simple fisheye with one radial term.", is_fisheye=True),
    9: CameraModel(9, "RADIAL_FISHEYE", ["f", "cx", "cy", "k1", "k2"],
                   "Fisheye with two radial terms.", is_fisheye=True),
    10: CameraModel(10, "THIN_PRISM_FISHEYE",
                    ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"],
                    "Thin-prism fisheye: 4 radial + 2 tangential + 2 thin-prism.", is_fisheye=True),
    11: CameraModel(11, "RAD_TAN_THIN_PRISM_FISHEYE",
                    ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "sx1", "sy1", "sx2", "sy2"],
                    "Extended thin-prism with 4 radial, 2 tangential, 4 thin-prism.", is_fisheye=True),
    12: CameraModel(12, "SIMPLE_DIVISION", ["f", "cx", "cy", "k1"],
                    "Simple division model with one radial term."),
    13: CameraModel(13, "DIVISION", ["f", "cx", "cy", "k1", "k2"],
                    "Division model with two radial terms."),
    14: CameraModel(14, "SIMPLE_FISHEYE", ["f", "cx", "cy"],
                    "Simple fisheye, no distortion.", is_fisheye=True),
    15: CameraModel(15, "FISHEYE", ["fx", "fy", "cx", "cy"],
                    "Fisheye with two focal lengths, no distortion.", is_fisheye=True),
    16: CameraModel(16, "EUCM", ["fx", "fy", "cx", "cy", "alpha", "beta"],
                    "Enhanced Unified Camera Model. Covers perspective (α≤0.5) to "
                    "wide-angle/fisheye (α>0.5) with a single smooth model. "
                    "6 params: fx,fy,cx,cy,α,β.", is_fisheye=True),
    17: CameraModel(17, "EQUIRECTANGULAR", ["f", "cx", "cy"],
                    "Equirectangular (360° lat/long spherical) projection. "
                    "No distortion — images are already in equirectangular format."),
}

CAMERA_MODEL_BY_NAME = {m.name: m for m in CAMERA_MODELS.values()}
PINHOLE_MODELS = [m for m in CAMERA_MODELS.values() if not m.is_fisheye]
FISHEYE_MODELS = [m for m in CAMERA_MODELS.values() if m.is_fisheye]

# Default across GUI, CLI, and pipeline: SIMPLE_RADIAL.
DEFAULT_CAMERA_MODEL_ID = 2


def get_camera_model(model_id: int) -> CameraModel:
    if model_id not in CAMERA_MODELS:
        raise KeyError(f"Unknown camera model ID: {model_id}")
    return CAMERA_MODELS[model_id]
