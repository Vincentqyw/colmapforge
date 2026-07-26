#!/usr/bin/env python
"""Thin launcher for PyInstaller — keeps package context intact.

PyInstaller treats the entry script as top-level __main__. If we point it
directly at colmapforge/app.py, the relative imports inside main()
(``from .onnx_utils import ...``) fail with:
    ImportError: attempted relative import with no known parent package

This launcher runs as a top-level module (no relative imports here), then
imports colmapforge.app — which establishes the package context so
all relative imports inside the package resolve correctly.
"""

from colmapforge.app import main

if __name__ == "__main__":
    raise SystemExit(main())
