"""Clickable swatch palettes and color conversion utilities. Colors stored as RGB (0-255)."""

from __future__ import annotations

SWATCHES: dict[str, list[tuple[str, tuple[int, int, int]]]] = {
    "lipstick": [
        ("Classic Red", (196, 30, 58)),
        ("Rose Petal", (201, 76, 109)),
        ("Soft Nude", (192, 128, 129)),
        ("Sweet Berry", (123, 45, 78)),
        ("Peach Coral", (224, 112, 96)),
        ("Soft Mauve", (164, 93, 122)),
        ("Wine Kiss", (107, 28, 49)),
        ("Baby Pink", (232, 138, 164)),
        ("Cherry Pie", (178, 34, 52)),
        ("Plum Velvet", (98, 33, 62)),
    ],
    "blush": [
        ("Peach Blossom", (232, 160, 144)),
        ("Dusty Rose", (224, 122, 138)),
        ("Sweet Rose", (212, 106, 122)),
        ("Warm Coral", (224, 144, 120)),
        ("Soft Peony", (232, 180, 184)),
        ("Apricot Flush", (228, 150, 118)),
        ("Warm Sunset", (198, 124, 95)),
    ],
    "eyeshadow": [
        ("Velvet Nude", (196, 164, 132)),
        ("Soft Taupe", (139, 115, 85)),
        ("Golden Bronze", (184, 115, 51)),
        ("Rose Gold", (183, 110, 121)),
        ("Dusty Plum", (107, 58, 93)),
        ("Cocoa Brown", (107, 68, 35)),
        ("Champagne Glow", (212, 188, 148)),
        ("Smoky Slate", (90, 86, 96)),
    ],
    "eyeliner": [
        ("Carbon Black", (18, 18, 20)),
        ("Mocha Brown", (54, 38, 32)),
        ("Midnight Blue", (24, 32, 54)),
        ("Smoky Charcoal", (48, 48, 52)),
    ],
    "foundation": [
        ("Fair Vanilla", (245, 222, 204)),
        ("Light Ivory", (240, 210, 186)),
        ("Medium Sand", (226, 186, 153)),
        ("Warm Honey", (208, 164, 128)),
        ("Sweet Caramel", (178, 132, 95)),
        ("Deep Chestnut", (138, 96, 68)),
    ],
    "brow": [
        ("Soft Brown", (75, 52, 42)),
        ("Dark Chocolate", (48, 34, 28)),
        ("Blonde Taupe", (130, 108, 88)),
        ("Ebony", (28, 24, 22)),
        ("Warm Auburn", (96, 54, 40)),
    ],
}


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (int(b), int(g), int(r))


def bgr_to_rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bgr
    return (int(r), int(g), int(b))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = [int(v) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    s = hex_code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return (255, 255, 255)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def hex_to_bgr(hex_code: str) -> tuple[int, int, int]:
    return rgb_to_bgr(hex_to_rgb(hex_code))