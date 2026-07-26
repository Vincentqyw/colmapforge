"""
COLMAP Forge — About dialog.

Apple-style modal dialog: logo + project info + author + models +
keyboard shortcuts + license. Reuses the app's QSS section-card styling
so it stays visually consistent with the main window.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .logo import make_logo_pixmap


GITHUB_URL = "https://github.com/vincentqyw/colmapforge"


def _app_version() -> str:
    """Resolve installed package version, falling back to 1.0.0."""
    try:
        from colmapforge import __version__
        return __version__ or "1.0.0"
    except Exception:
        return "1.0.0"


# ── Section card helper ───────────────────────────────────────────────
def _section_card(title: str) -> tuple[QWidget, QVBoxLayout]:
    """A rounded section card with a header label and a body layout."""
    card = QWidget(); card.setObjectName("sectionCard")
    cl = QVBoxLayout(card)
    cl.setContentsMargins(14, 10, 14, 12)
    cl.setSpacing(4)
    head = QLabel(title.upper()); head.setObjectName("sectionHeader")
    cl.addWidget(head)
    body_wrap = QWidget()
    body = QVBoxLayout(body_wrap)
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(2)
    cl.addWidget(body_wrap)
    return card, body


# ── AboutDialog ──────────────────────────────────────────────────────
class AboutDialog(QDialog):
    """Modal About dialog — theme-aware, scrollable if viewport is small."""

    def __init__(self, parent: QWidget | None = None, is_dark: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setModal(True)
        self.setFixedSize(460, 620)
        self._is_dark = is_dark

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 16)
        outer.setSpacing(12)

        # ── Header: logo + name + version + tagline ──
        outer.addLayout(self._build_header())

        # ── Scrollable sections ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setObjectName("leftPanel")  # reuse transparent-bg rule
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        self._add_project(cl)
        self._add_author(cl)
        self._add_models(cl)
        self._add_shortcuts(cl)
        self._add_license(cl)
        cl.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # ── Close button ──
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setFixedWidth(120)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        outer.addLayout(btn_row)

    # ── Header ──
    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(14)
        row.setContentsMargins(0, 0, 0, 4)

        logo = QLabel()
        logo.setPixmap(make_logo_pixmap(72, self._is_dark))
        logo.setFixedSize(72, 72)
        row.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)

        col = QVBoxLayout(); col.setSpacing(1)
        name = QLabel("COLMAP Forge"); name.setObjectName("appTitle")
        ver = QLabel(f"Version {_app_version()}"); ver.setObjectName("hintLabel")
        tag = QLabel("SAM-based masking for COLMAP SfM"); tag.setObjectName("hintLabel")
        col.addWidget(name); col.addWidget(ver); col.addWidget(tag)
        row.addLayout(col, 1)
        row.addStretch()
        return row

    # ── Sections ──
    def _add_project(self, layout: QVBoxLayout) -> None:
        card, body = _section_card("Project")
        desc = QLabel(
            "A PyQt6-based wizard tool that prepares video and image data for "
            "COLMAP Structure-from-Motion pipelines. Supports frame extraction, "
            "image resizing, SAM-based dynamic object masking, all 16 COLMAP "
            "camera models, and COLMAP database.db export."
        )
        desc.setWordWrap(True)
        body.addWidget(desc)
        layout.addWidget(card)

    def _add_author(self, layout: QVBoxLayout) -> None:
        card, body = _section_card("Author")
        name = QLabel("Vincent Qin  ·  @vincentqyw")
        body.addWidget(name)
        body.addWidget(self._link_label(GITHUB_URL))
        layout.addWidget(card)

    def _add_models(self, layout: QVBoxLayout) -> None:
        card, body = _section_card("Models & Weights")

        # ── SkyWater SegFormer-B2 ──
        m1 = QLabel("<b>SkyWater SegFormer-B2</b> (FP16)")
        m1.setTextFormat(Qt.TextFormat.RichText)
        d1 = QLabel("Segments sky, water, and people. ~48 MB, auto-downloaded "
                    "on first use.")
        d1.setWordWrap(True); d1.setObjectName("hintLabel")
        l1 = self._link_label("https://huggingface.co/Realcat/skywater_seg")
        body.addWidget(m1); body.addWidget(d1); body.addWidget(l1)

        body.addSpacing(4)

        # ── SAM (Segment Anything) ONNX models ──
        m2 = QLabel("<b>SAM / SAM2 / SAM3</b> (ONNX)")
        m2.setTextFormat(Qt.TextFormat.RichText)
        d2 = QLabel("General-purpose interactive segmentation backends. Loaded "
                    "from ~/.colmapforge/models/ on demand.")
        d2.setWordWrap(True); d2.setObjectName("hintLabel")
        l2 = self._link_label("https://huggingface.co/vietanhdev/segment-anything-3-onnx-models")
        body.addWidget(m2); body.addWidget(d2); body.addWidget(l2)

        layout.addWidget(card)

    def _link_label(self, url: str) -> QLabel:
        """A clickable external link label styled with the accent color."""
        lbl = QLabel(f'<a href="{url}" style="color:#0a84ff;">{url}</a>')
        lbl.setOpenExternalLinks(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl.setWordWrap(True)
        return lbl

    def _add_shortcuts(self, layout: QVBoxLayout) -> None:
        card, body = _section_card("Keyboard Shortcuts")
        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setHorizontalSpacing(16); grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        shortcuts = [
            ("Ctrl + O", "Add images"),
            ("Ctrl + B", "Build COLMAP database"),
            ("Ctrl + T", "Toggle theme"),
            ("←  /  →",  "Previous / Next image"),
        ]
        for i, (key, desc) in enumerate(shortcuts):
            k = QLabel(key); k.setObjectName("hintLabel")
            d = QLabel(desc)
            grid.addWidget(k, i, 0)
            grid.addWidget(d, i, 1)
        body.addWidget(grid_wrap)
        layout.addWidget(card)

    def _add_license(self, layout: QVBoxLayout) -> None:
        card, body = _section_card("License")
        lic = QLabel("MIT License — free for academic and commercial use.")
        lic.setWordWrap(True)
        body.addWidget(lic)
        layout.addWidget(card)
