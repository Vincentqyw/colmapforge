"""ONNX Runtime provider selection utilities.

Prefers GPU acceleration (CUDA → DirectML) and falls back to CPU.
TensorRT is excluded (stability issues with dynamic shapes).

This module also exposes :func:`diagnose` which inspects the local
onnxruntime installation and surfaces the *silent overwrite* failure mode
that occurs when both ``onnxruntime`` and ``onnxruntime-gpu`` are installed
— their wheels share the ``onnxruntime/`` site-packages directory and
overwrite each other's binaries, causing ``CUDAExecutionProvider`` to
quietly disappear from ``get_available_providers()`` without raising any
error.
"""
from __future__ import annotations

import importlib.metadata as _md
import logging

logger = logging.getLogger(__name__)

# ── Import onnxruntime with graceful failure ──────────────────────────
# A missing onnxruntime must NOT crash the whole app — the user can still
# browse the GUI and add inputs; segmentation will fail with a clear message.
try:
    import onnxruntime as _ort

    _ONNXRUNTIME_AVAILABLE = True
    _ONNXRUNTIME_IMPORT_ERROR: str | None = None
except ImportError as _e:
    _ort = None  # type: ignore[assignment]
    _ONNXRUNTIME_AVAILABLE = False
    _ONNXRUNTIME_IMPORT_ERROR = str(_e)


_INSTALL_HINT = (
    "ONNX Runtime is not installed. Install EXACTLY ONE of the following "
    "(they are mutually exclusive and will silently break each other):\n"
    "  uv pip install onnxruntime           # CPU only\n"
    "  uv pip install onnxruntime-gpu       # NVIDIA + CUDA 13\n"
    "  uv pip install onnxruntime-directml  # Windows, any GPU\n"
    "Or use the project extras:\n"
    '  uv pip install -e ".[cpu]"\n'
    '  uv pip install -e ".[gpu]"\n'
    '  uv pip install -e ".[directml]"'
)

# Fix recipe shown when the silent-overwrite issue is detected.
_OVERWRITE_FIX = (
    "onnxruntime-gpu is installed but CUDAExecutionProvider is missing from "
    "available providers. This is caused by the CPU wheel (onnxruntime) "
    "being installed alongside the GPU wheel and overwriting its binaries. "
    "Fix:\n"
    "  uv pip uninstall onnxruntime onnxruntime-gpu\n"
    "  Remove-Item .venv/Lib/site-packages/onnxruntime -Recurse -Force\n"
    "  uv pip install onnxruntime-gpu --no-deps"
)


# ── Public API ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return ``True`` if the ``onnxruntime`` module can be imported."""
    return _ONNXRUNTIME_AVAILABLE


def require_onnxruntime() -> None:
    """Raise a :class:`RuntimeError` with install hint if onnxruntime is missing."""
    if not _ONNXRUNTIME_AVAILABLE:
        raise RuntimeError(_INSTALL_HINT)


def get_onnx_providers(
    *, force_cpu: bool = False
) -> list[str]:
    """Return a prioritized list of ONNX Runtime execution providers.

    Order: CUDAExecutionProvider → DirectML → CPUExecutionProvider.
    If ``force_cpu=True``, only CPUExecutionProvider is returned.
    """
    require_onnxruntime()
    if force_cpu:
        return ["CPUExecutionProvider"]
    available = _ort.get_available_providers()  # type: ignore[union-attr]
    ordered: list[str] = []
    # 1) CUDA (highest throughput for typical ONNX models)
    if "CUDAExecutionProvider" in available:
        ordered.append("CUDAExecutionProvider")
    # 2) DirectML (Windows GPU fallback when CUDA is unavailable)
    if "DmlExecutionProvider" in available:
        ordered.append("DmlExecutionProvider")
    # 3) CPU (always available, guaranteed fallback)
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def create_inference_session(
    model_path: str, *, force_cpu: bool = False
):
    """Create an ONNX Runtime InferenceSession with the best available provider."""
    require_onnxruntime()
    providers = get_onnx_providers(force_cpu=force_cpu)
    return _ort.InferenceSession(model_path, providers=providers)  # type: ignore[union-attr]


# ── Diagnostics ───────────────────────────────────────────────────────

def _installed_dist_names() -> set[str]:
    """Lower-cased distribution names that map to the onnxruntime module."""
    try:
        return {d.metadata["Name"].lower() for d in _md.distributions()}
    except Exception:  # pragma: no cover — metadata API is stable
        return set()


def diagnose() -> dict:
    """Inspect the local onnxruntime installation.

    Returns a dict with:

    - ``installed``: bool — module importable?
    - ``version``: str | None
    - ``providers``: list[str] — available execution providers
    - ``active_provider``: str | None — first GPU-ish provider, else None
    - ``gpu_active``: bool — CUDA or DirectML present?
    - ``installed_wheels``: list[str] — which onnxruntime* dists are present
    - ``issues``: list[str] — human-readable problems (empty if healthy)
    - ``hint``: str | None — install/fix hint when ``issues`` is non-empty
    """
    wheels = sorted(
        n for n in _installed_dist_names()
        if n.startswith("onnxruntime")
    )

    if not _ONNXRUNTIME_AVAILABLE:
        return {
            "installed": False,
            "version": None,
            "providers": [],
            "active_provider": None,
            "gpu_active": False,
            "installed_wheels": wheels,
            "issues": [
                _ONNXRUNTIME_IMPORT_ERROR or "unknown import error"
            ],
            "hint": _INSTALL_HINT,
        }

    version = _ort.__version__  # type: ignore[union-attr]
    available = _ort.get_available_providers()  # type: ignore[union-attr]
    # Filter out TensorrtExecutionProvider — excluded due to stability issues
    available = [p for p in available if p != "TensorrtExecutionProvider"]
    has_cuda = "CUDAExecutionProvider" in available
    has_dml = "DmlExecutionProvider" in available

    active = None
    if has_cuda:
        active = "CUDA"
    elif has_dml:
        active = "DirectML"

    issues: list[str] = []
    hint: str | None = None

    # The silent-overwrite problem: GPU wheel metadata present but CUDA
    # provider missing from the runtime. This is exactly the failure mode
    # that bites new users after `pip install onnxruntime onnxruntime-gpu`.
    has_gpu_wheel = "onnxruntime-gpu" in wheels
    has_dml_wheel = "onnxruntime-directml" in wheels
    has_cpu_wheel = "onnxruntime" in wheels

    if has_gpu_wheel and not has_cuda:
        issues.append(_OVERWRITE_FIX)
        hint = _OVERWRITE_FIX
    elif has_dml_wheel and not has_dml:
        issues.append(
            "onnxruntime-directml is installed but DmlExecutionProvider is "
            "missing. Reinstall it cleanly:\n"
            "  uv pip uninstall onnxruntime onnxruntime-directml\n"
            "  uv pip install onnxruntime-directml --no-deps"
        )
        hint = issues[-1]
    elif len(wheels) > 1 and not (has_cuda or has_dml):
        # Multiple onnxruntime* dists installed and no GPU provider active.
        issues.append(
            f"Multiple onnxruntime distributions installed: {wheels}. "
            "Only ONE should be present. Uninstall all and reinstall the one "
            "you want (see _INSTALL_HINT)."
        )
        hint = _INSTALL_HINT

    return {
        "installed": True,
        "version": version,
        "providers": available,
        "active_provider": active,
        "gpu_active": bool(active),
        "has_cuda": has_cuda,
        "has_dml": has_dml,
        "installed_wheels": wheels,
        "issues": issues,
        "hint": hint,
    }


def log_diagnostics() -> dict:
    """Run :func:`diagnose` and emit log messages. Returns the diagnostic dict.

    Called once at app startup so the user can see the GPU state in the log
    without having to wait for a segmentation run to fail.
    """
    diag = diagnose()
    if not diag["installed"]:
        logger.error("ONNX Runtime not installed.\n%s", diag["hint"])
        return diag
    if diag["issues"]:
        for issue in diag["issues"]:
            logger.warning("ONNX Runtime issue: %s", issue)
    logger.info(
        "ONNX Runtime %s ready. Providers: %s",
        diag["version"], diag["providers"],
    )
    if diag["gpu_active"]:
        logger.info("GPU acceleration: ENABLED (%s)", diag["active_provider"])
    else:
        logger.info("GPU acceleration: disabled (CPU only)")
    return diag


# ── Auto-repair ───────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str]) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _detect_package_manager() -> str:
    """Return 'uv' if uv is available, else 'pip'."""
    import shutil
    if shutil.which("uv"):
        return "uv"
    return "pip"


def auto_fix_overwrite() -> tuple[bool, str]:
    """Auto-repair the silent-overwrite problem.

    Uninstalls both ``onnxruntime`` and ``onnxruntime-gpu``, removes residual
    files from ``site-packages/onnxruntime/``, then reinstalls ``onnxruntime-gpu``
    with ``--no-deps`` so no transitive CPU wheel can sneak back in.

    Returns ``(success, message)``. The caller MUST terminate the current
    process after a successful fix — the running Python interpreter still
    holds the broken ``onnxruntime`` module in memory and only a fresh
    process will pick up the repaired files.
    """
    import os
    import shutil
    import sys

    venv_python = sys.executable
    pm = _detect_package_manager()
    logger.warning("Starting auto-fix using %s. Python: %s", pm, venv_python)

    # Locate site-packages dir from the current onnxruntime install
    site_packages = None
    if _ONNXRUNTIME_AVAILABLE:
        try:
            site_packages = os.path.dirname(os.path.dirname(_ort.__file__))  # type: ignore[union-attr]
        except Exception:
            pass
    if not site_packages:
        # Fallback: derive from venv python path
        venv_root = os.path.dirname(os.path.dirname(venv_python))
        candidate = os.path.join(venv_root, "Lib", "site-packages")
        if os.path.isdir(candidate):
            site_packages = candidate
    if not site_packages:
        return False, "Could not locate site-packages directory."

    # Step 1: uninstall both wheels
    if pm == "uv":
        rc, out, err = _run_subprocess([
            "uv", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu",
            "--python", venv_python,
        ])
    else:
        rc, out, err = _run_subprocess([
            venv_python, "-m", "pip", "uninstall", "-y",
            "onnxruntime", "onnxruntime-gpu",
        ])
    if rc != 0:
        return False, f"Uninstall failed:\n{err or out}"
    logger.info("Uninstalled onnxruntime + onnxruntime-gpu")

    # Step 2: remove residual files (onnxruntime/ dir + dist-info dirs)
    ort_dir = os.path.join(site_packages, "onnxruntime")
    if os.path.isdir(ort_dir):
        shutil.rmtree(ort_dir, ignore_errors=True)
        logger.info("Removed residual %s", ort_dir)
    for name in os.listdir(site_packages):
        if name.lower().startswith("onnxruntime") and name.endswith(".dist-info"):
            shutil.rmtree(os.path.join(site_packages, name), ignore_errors=True)
            logger.info("Removed residual %s", name)

    # Step 3: reinstall GPU wheel with --no-deps (prevents osam from pulling CPU)
    if pm == "uv":
        rc, out, err = _run_subprocess([
            "uv", "pip", "install", "onnxruntime-gpu", "--no-deps",
            "--python", venv_python,
        ])
    else:
        rc, out, err = _run_subprocess([
            venv_python, "-m", "pip", "install", "onnxruntime-gpu", "--no-deps",
        ])
    if rc != 0:
        return False, f"Reinstall failed:\n{err or out}"
    logger.info("Reinstalled onnxruntime-gpu")

    return True, (
        "Successfully uninstalled the CPU onnxruntime wheel and reinstalled "
        "onnxruntime-gpu. Restart the application to activate CUDA."
    )


def ensure_onnxruntime_healthy(*, auto_fix: bool = True) -> bool:
    """Check ONNX Runtime health; optionally auto-repair and request restart.

    Returns ``True`` if healthy (or no fix was attempted). Returns ``False``
    when a fix was applied and the caller MUST exit the process so the user
    can restart with the repaired environment.

    Intended to be called early at app startup, after ``QApplication`` is
    created (so ``QMessageBox`` is available for the restart prompt).
    """
    diag = diagnose()
    if not diag["issues"]:
        return True  # healthy

    logger.warning("ONNX Runtime issue detected: %s", diag["issues"][0])

    if not auto_fix:
        return False

    # Only auto-fix the silent-overwrite problem (GPU wheel present but CUDA missing)
    wheels = diag.get("installed_wheels", [])
    has_gpu_wheel = "onnxruntime-gpu" in wheels
    has_cpu_wheel = "onnxruntime" in wheels
    if not (has_gpu_wheel and has_cpu_wheel and not diag["gpu_active"]):
        # Different problem (e.g. onnxruntime not installed at all) — can't auto-fix
        return False

    from PyQt6.QtWidgets import QMessageBox

    reply = QMessageBox.question(
        None,
        "ONNX Runtime — Auto-Fix Available",
        "ONNX Runtime GPU support is broken: the CPU wheel was installed "
        "alongside the GPU wheel and overwrote its binaries.\n\n"
        "The application can automatically fix this by:\n"
        "  1. Uninstalling both onnxruntime and onnxruntime-gpu\n"
        "  2. Removing residual files\n"
        "  3. Reinstalling onnxruntime-gpu with --no-deps\n\n"
        "After the fix, the app will exit — please restart it.\n\n"
        "Proceed with auto-fix?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        logger.warning("User declined auto-fix. App will continue with broken ONNX Runtime.")
        return False

    success, msg = auto_fix_overwrite()
    if success:
        logger.info("Auto-fix succeeded: %s", msg)
        QMessageBox.information(
            None,
            "ONNX Runtime Fixed",
            msg + "\n\nClick OK to exit. Then restart the application.",
        )
        return False  # caller must exit
    else:
        logger.error("Auto-fix failed: %s", msg)
        QMessageBox.critical(
            None,
            "Auto-Fix Failed",
            msg + "\n\nPlease fix manually:\n"
            "  uv pip uninstall onnxruntime onnxruntime-gpu\n"
            "  Remove-Item .venv/Lib/site-packages/onnxruntime -Recurse -Force\n"
            "  uv pip install onnxruntime-gpu --no-deps",
        )
        return False
