"""Soft region masks and color blending (lipstick / blush / eyeshadow)."""

from __future__ import annotations

import cv2
import numpy as np


def points_to_pixels(points, width: int, height: int) -> np.ndarray:
    return np.array([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def create_polygon_mask(image_shape, points, hole_points=None) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pixel_points = points_to_pixels(points, w, h)
    if len(pixel_points) >= 3:
        cv2.fillPoly(mask, [pixel_points], 255)
    if hole_points:
        hole = points_to_pixels(hole_points, w, h)
        if len(hole) >= 3:
            cv2.fillPoly(mask, [hole], 0)
    return mask


def create_lip_mask(image_shape, lip_points):
    """Original helper: filled polygon from a single point list."""
    return create_polygon_mask(image_shape, lip_points)


def create_ellipse_mask(image_shape, points, axes: tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pixel_points = points_to_pixels(points, w, h)
    if len(pixel_points) == 0:
        return mask
    center = tuple(np.mean(pixel_points, axis=0).astype(int))
    axes = (max(1, axes[0]), max(1, axes[1]))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def soften_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    kernel = max(3, int(kernel) | 1)
    return cv2.GaussianBlur(mask, (kernel, kernel), 0)


def apply_color(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    opacity: float,
    mode: str = "mix",
) -> np.ndarray:
    """Blend a BGR color into `image` using a 0-255 mask and opacity in [0, 1]."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0:
        return image

    base = image.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0) * opacity
    alpha_3 = cv2.merge([alpha, alpha, alpha])

    if mode == "overlay":
        base_n = base / 255.0
        color_n = color / 255.0
        overlay = np.where(
            base_n < 0.5,
            2.0 * base_n * color_n,
            1.0 - 2.0 * (1.0 - base_n) * (1.0 - color_n),
        )
        blended = overlay * 255.0
    elif mode == "multiply":
        blended = (base / 255.0) * color
        blended = np.clip(blended, 0, 255)
    else:
        colored = np.empty_like(base)
        colored[:] = color
        blended = colored

    result = base * (1.0 - alpha_3) + blended * alpha_3
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lip_color(image, mask, color_bgr, opacity=0.5):
    """Original helper used by the still-image CLI."""
    return apply_color(image, mask, color_bgr, opacity, mode="mix")