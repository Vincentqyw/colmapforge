"""
COLMAP SQLite database builder.

Creates a standard database.db with cameras + images tables,
ready for import into COLMAP.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
from pathlib import Path

import cv2

from .camera_models import CameraModel

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    model       INTEGER NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    params      BLOB,
    prior_focal_length INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS images (
    image_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    camera_id   INTEGER NOT NULL,
    prior_qw    REAL, prior_qx    REAL, prior_qy    REAL, prior_qz    REAL,
    prior_tx    REAL, prior_ty    REAL, prior_tz    REAL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS keypoints (
    image_id    INTEGER PRIMARY KEY,
    rows        INTEGER, cols INTEGER, data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS descriptors (
    image_id    INTEGER PRIMARY KEY,
    rows        INTEGER, cols INTEGER, data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS matches (
    pair_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id1   INTEGER, image_id2 INTEGER,
    rows        INTEGER, cols INTEGER, data BLOB,
    FOREIGN KEY(image_id1) REFERENCES images(image_id) ON DELETE CASCADE,
    FOREIGN KEY(image_id2) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id1   INTEGER, image_id2 INTEGER,
    rows        INTEGER, cols INTEGER, data BLOB,
    config      INTEGER NOT NULL DEFAULT 0,
    F           BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB,
    FOREIGN KEY(image_id1) REFERENCES images(image_id) ON DELETE CASCADE,
    FOREIGN KEY(image_id2) REFERENCES images(image_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_images_camera ON images(camera_id);
"""


def _params_to_blob(params: list[float]) -> bytes:
    return struct.pack(f"<{len(params)}d", *params)


def _relative_path(absolute_path: str, base_dir: str) -> str:
    try:
        return str(Path(absolute_path).relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        return absolute_path.replace("\\", "/")


class ColmapDatabase:
    """Builder for a COLMAP-compatible SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "ColmapDatabase":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def open(self) -> None:
        if self._conn is not None:
            return
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def add_camera(
        self, model: CameraModel, width: int, height: int,
        params: list[float], prior_focal_length: int = 0,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?)",
            (model.model_id, width, height, _params_to_blob(params), prior_focal_length),
        )
        return cur.lastrowid

    def add_image(
        self, name: str, camera_id: int,
        prior_position: tuple[float, ...] | None = None,
    ) -> int:
        if prior_position:
            qw, qx, qy, qz, tx, ty, tz = prior_position
            cur = self._conn.execute(
                "INSERT INTO images (name, camera_id, prior_qw, prior_qx, prior_qy, "
                "prior_qz, prior_tx, prior_ty, prior_tz) VALUES (?,?,?,?,?,?,?,?,?)",
                (name, camera_id, qw, qx, qy, qz, tx, ty, tz),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO images (name, camera_id) VALUES (?,?)", (name, camera_id),
            )
        return cur.lastrowid

    def add_images_batch(
        self, image_paths: list[str], camera_id: int,
        base_dir: str | None = None,
    ) -> list[int]:
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(self.db_path))
        ids = []
        cur = self._conn.cursor()
        for path in image_paths:
            rel = _relative_path(os.path.abspath(path), base_dir)
            cur.execute(
                "INSERT INTO images (name, camera_id) VALUES (?,?)", (rel, camera_id),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()
        return ids

    def build_project(
        self, image_dir: str, camera_model: CameraModel,
        camera_params: list[float] | None = None,
        prior_focal_length: int = 0,
    ) -> dict:
        from .utils import collect_image_files

        image_paths = collect_image_files([image_dir], recursive=False)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        first = cv2.imread(image_paths[0])
        if first is None:
            raise RuntimeError(f"Cannot read image: {image_paths[0]}")
        h, w = first.shape[:2]

        if camera_params is None:
            camera_params = camera_model.default_params(w, h)

        cam_id = self.add_camera(camera_model, w, h, camera_params, prior_focal_length)
        img_ids = self.add_images_batch(image_paths, cam_id, base_dir=os.path.dirname(image_dir))

        logger.info("Database built: %d images, %s [%d], %dx%d", len(image_paths), camera_model.name, cam_id, w, h)
        return {"camera_id": cam_id, "image_ids": img_ids, "image_count": len(image_paths), "width": w, "height": h}
