"""
Design constants and data for COLMAP Forge UI.
"""

# ── Design Constants ──────────────────────────────────────────────────


class _S:
    """Spacing constants."""
    SECTION_GAP = 6
    ROW_GAP = 1
    CARD_PAD = (10, 3, 10, 3)   # left top right bottom — tight
    LABEL_W = 80                  # form label fixed width (fits "Confidence", "Max frames")


class _W:
    """Widget width constants."""
    COMBO = 150
    SPIN_MIN = 70
    # No SPIN_MAX — let spinbox stretch to fill grid col 1-2, same as combo box


# ── Data Constants ────────────────────────────────────────────────────

PRESET_CLASSES = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "bird", "cat", "dog", "boat", "sky", "water",
    "reflection", "shadow", "tree branch", "cloud",
]

QUICK_PRESETS = {
    "People + Vehicles": ["person", "car", "bus", "truck", "motorcycle", "bicycle"],
    "Sky + Water": ["sky", "water", "cloud", "reflection"],
    "All Dynamics": list(PRESET_CLASSES),
    "Animals": ["bird", "cat", "dog"],
}

DEFAULT_PRESET = "All Dynamics"
