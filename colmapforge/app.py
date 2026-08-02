#!/usr/bin/env python
"""
COLMAP Forge — Standalone Application Launcher.

Usage:
    colmapforge                       Launch the GUI
    colmapforge run [options]         Run the CLI pipeline (headless)
    colmapforge run --help            Show CLI options
    colmapforge download --all        Pre-download all models
    colmapforge download NAME [...]   Pre-download specific models
"""

from __future__ import annotations

import argparse
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _launch_gui(log_level: str) -> int:
    """Launch the PyQt6 GUI.  Imported lazily so the CLI path never pays the
    Qt startup cost."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting COLMAP Forge...")

    # Early ONNX Runtime diagnostic — surfaces the silent "GPU wheel got
    # overwritten by CPU wheel" issue at startup instead of letting the user
    # discover it from a slow segmentation run.
    from .onnx_utils import log_diagnostics
    log_diagnostics()

    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication

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
    #
    # Returns (should_continue, needs_restart):
    #   - (True, False)  → healthy, or issue exists but app can continue
    #   - (False, True)  → auto-fix applied; caller MUST exit so user restarts
    from .onnx_utils import ensure_onnxruntime_healthy
    should_continue, needs_restart = ensure_onnxruntime_healthy()
    if needs_restart:
        logger.warning("ONNX Runtime needs a restart after auto-fix. Exiting.")
        return 0
    if not should_continue:
        # Unrecoverable issue; onnx_utils already showed an error dialog.
        logger.error("ONNX Runtime is in an unrecoverable state. Exiting.")
        return 1

    from .main_window import MainWindow
    window = MainWindow()
    window.show()
    logger.info("Window shown")
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    # ── Dispatch: "colmapforge run ..." → CLI, otherwise GUI ──
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) >= 1 and argv[0] == "run":
        from .cli import run_cli
        return run_cli(argv[1:])

    if len(argv) >= 1 and argv[0] == "download":
        from .cli import run_download
        return run_download(argv[1:])

    # ── GUI mode ──
    parser = argparse.ArgumentParser(
        prog="colmapforge",
        usage=(
            "colmapforge [--log-level LEVEL]     launch the desktop GUI\n"
            "       colmapforge run [options]          run the pipeline headlessly\n"
            "       colmapforge download [options]     pre-download models"
        ),
        description=(
            "COLMAP Forge — prepare video/image data for COLMAP Structure-from-"
            "Motion: frame extraction, resize, dynamic-object masking (SkyWater / "
            "YOLO-World + SAM text prompts / SAM3), and database.db export."
        ),
        epilog=(
            "commands:\n"
            "  (none)      Launch the desktop GUI\n"
            "  run         Headless pipeline: video/images → frames → resize →\n"
            "              segmentation masks → COLMAP database.db\n"
            "  download    Pre-download segmentation models (SHA256-verified;\n"
            "              already-downloaded models are skipped)\n"
            "\n"
            "examples:\n"
            "  colmapforge                                # GUI\n"
            "  colmapforge run -o out/ --video vid.mp4 \\\n"
            "      --seg-model yoloworld_efficientvit_sam --seg-classes person car\n"
            "  colmapforge run --list-models              # show available models\n"
            "  colmapforge download --all                 # fetch every model\n"
            "\n"
            "  colmapforge run --help                     # all pipeline options\n"
            "  colmapforge download --help                # all download options"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="GUI logging level (default: INFO)")
    args = parser.parse_args(argv)
    return _launch_gui(args.log_level)


if __name__ == "__main__":
    sys.exit(main())
