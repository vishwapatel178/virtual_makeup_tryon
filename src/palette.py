"""Clickable swatch palettes. Colors are stored as RGB (0-255)."""

from __future__ import annotations

SWATCHES = {
    "lipstick": [
        ("Classic Red", (196, 30, 58)),
        ("Rose", (201, 76, 109)),
        ("Nude", (192, 128, 129)),
        ("Berry", (123, 45, 78)),
        ("Coral", (224, 112, 96)),
        ("Mauve", (164, 93, 122)),
        ("Wine", (107, 28, 49)),
        ("Pink", (232, 138, 164)),
    ],
    "blush": [
        ("Peach", (232, 160, 144)),
        ("Pink", (224, 122, 138)),
        ("Rose", (212, 106, 122)),
        ("Coral", (224, 144, 120)),
        ("Soft", (232, 180, 184)),
        ("Apricot", (228, 150, 118)),
    ],
    "eyeshadow": [
        ("Nude", (196, 164, 132)),
        ("Taupe", (139, 115, 85)),
        ("Bronze", (184, 115, 51)),
        ("Rose Gold", (183, 110, 121)),
        ("Plum", (107, 58, 93)),
        ("Brown", (107, 68, 35)),
        ("Champagne", (212, 188, 148)),
        ("Smoke", (90, 86, 96)),
    ],
}


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"