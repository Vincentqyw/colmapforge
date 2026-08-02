"""Centralized theming — Apple HIG color system."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication

_STYLES_DIR = Path(__file__).parent / "views" / "styles"


class Theme:
    """Singleton theme manager. Loads QSS from light.qss / dark.qss."""

    _instance: Theme | None = None

    def __init__(self) -> None:
        self._dark = False
        Theme._instance = self

    @property
    def is_dark(self) -> bool:
        return self._dark

    def toggle(self) -> None:
        self._dark = not self._dark

    def set_dark(self, dark: bool) -> None:
        self._dark = dark

    @classmethod
    def get(cls) -> "Theme":
        if cls._instance is None:
            cls._instance = Theme()
        return cls._instance


def _make_palette(dark: bool) -> QPalette:
    """Build a QPalette matching the Apple HIG theme."""
    p = QPalette()
    if dark:
        p.setColor(QPalette.ColorRole.Window,          QColor("#1c1c1e"))
        p.setColor(QPalette.ColorRole.WindowText,       QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.Base,             QColor("#2c2c2e"))
        p.setColor(QPalette.ColorRole.AlternateBase,    QColor("#3a3a3c"))
        p.setColor(QPalette.ColorRole.Text,             QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.Button,           QColor("#3a3a3c"))
        p.setColor(QPalette.ColorRole.ButtonText,       QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.Highlight,        QColor("#0a84ff"))
        p.setColor(QPalette.ColorRole.HighlightedText,  QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.Link,             QColor("#409cff"))
        p.setColor(QPalette.ColorRole.LinkVisited,      QColor("#0a84ff"))
        p.setColor(QPalette.ColorRole.PlaceholderText,  QColor("#48484a"))
        p.setColor(QPalette.ColorRole.ToolTipBase,      QColor("#2c2c2e"))
        p.setColor(QPalette.ColorRole.ToolTipText,      QColor("#ffffff"))
    else:
        p.setColor(QPalette.ColorRole.Window,          QColor("#f2f2f7"))
        p.setColor(QPalette.ColorRole.WindowText,       QColor("#000000"))
        p.setColor(QPalette.ColorRole.Base,             QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.AlternateBase,    QColor("#f2f2f7"))
        p.setColor(QPalette.ColorRole.Text,             QColor("#000000"))
        p.setColor(QPalette.ColorRole.Button,           QColor("#e5e5ea"))
        p.setColor(QPalette.ColorRole.ButtonText,       QColor("#000000"))
        p.setColor(QPalette.ColorRole.Highlight,        QColor("#007aff"))
        p.setColor(QPalette.ColorRole.HighlightedText,  QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.Link,             QColor("#007aff"))
        p.setColor(QPalette.ColorRole.LinkVisited,      QColor("#5856d6"))
        p.setColor(QPalette.ColorRole.PlaceholderText,  QColor("#c7c7cc"))
        p.setColor(QPalette.ColorRole.ToolTipBase,      QColor("#ffffff"))
        p.setColor(QPalette.ColorRole.ToolTipText,      QColor("#000000"))
    return p


def apply_theme(app: QApplication, theme: Theme) -> None:
    """Apply QSS + palette at QApplication level."""
    qss_file = _STYLES_DIR / ("dark.qss" if theme.is_dark else "light.qss")
    if qss_file.is_file():
        app.setStyleSheet(qss_file.read_text(encoding="utf-8"))
    app.setPalette(_make_palette(theme.is_dark))
