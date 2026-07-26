#!/usr/bin/env python
"""
COLMAP Forge — Standalone Application Launcher.

Usage:
    colmapforge
    python -m colmapforge.app
"""

from __future__ import annotations

import argparse
import logging
import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COLMAP Forge")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting COLMAP Forge...")

    # Early ONNX Runtime diagnostic — surfaces the silent "GPU wheel got
    # overwritten by CPU wheel" issue at startup instead of letting the user
    # discover it from a slow segmentation run.
    from .onnx_utils import log_diagnostics
    log_diagnostics()

    app = QApplication(sys.argv)
    app.setApplicationName("COLMAP Forge")
    app.setOrganizationName("AnyLabeling")
    app.setStyle("Fusion")

    # Apple-style system font — must include robust fallbacks for icon glyphs
    font = QFont()
    font.setFamilies(["Segoe UI", "Segoe UI Variable", "Microsoft YaHei UI", "SF Pro Text",
                      "Helvetica Neue", "Arial", "sans-serif"])
    font.setPointSize(10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    # ── Self-healing: detect & auto-fix the silent-overwrite problem ──
    # If osam (or any other package) pulled the CPU onnxruntime wheel back
    # in and overwrote onnxruntime-gpu's binaries, this will offer to
    # uninstall the CPU wheel + reinstall GPU, then ask the user to restart.
    # Must run AFTER QApplication is created (uses QMessageBox).
    from .onnx_utils import ensure_onnxruntime_healthy
    if not ensure_onnxruntime_healthy():
        logger.warning("ONNX Runtime needs a restart after auto-fix. Exiting.")
        return 0

    from .main_window import MainWindow
    window = MainWindow()
    window.show()
    logger.info("Window shown")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
