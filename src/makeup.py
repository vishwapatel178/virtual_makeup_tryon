"""Apply lipstick, blush, and eyeshadow from detected landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from landmarks import (
    INNER_LIP_INDICES,
    LEFT_BROW_INDICES,
    LEFT_CHEEK_INDICES,
    LEFT_EYE_INDICES,
    LEFT_UPPER_LID_INDICES,
    OUTER_LIP_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_CHEEK_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_UPPER_LID_INDICES,
    face_width_px,
    get_region_points,
)
from masking import (
    apply_color,
    create_ellipse_mask,
    create_eyeshadow_mask,
    create_polygon_mask,
    soften_mask,
)


@dataclass
class ProductState:
    enabled: bool = True
    color_bgr: tuple[int, int, int] = (80, 60, 180)
    intensity: float = 0.55


@dataclass
class MakeupSettings:
    lipstick: ProductState
    blush: ProductState
    eyeshadow: ProductState


def apply_makeup(image: np.ndarray, landmarks, settings: MakeupSettings) -> np.ndarray:
    if not landmarks:
        return image

    result = image
    h, w = image.shape[:2]
    face_w = face_width_px(landmarks, w, h)

    if settings.lipstick.enabled and settings.lipstick.intensity > 0:
        outer = get_region_points(landmarks, OUTER_LIP_INDICES)
        inner = get_region_points(landmarks, INNER_LIP_INDICES)
        mask = create_polygon_mask(image.shape, outer, inner)
        mask = soften_mask(mask, max(7, face_w // 35))
        result = apply_color(
            result,
            mask,
            settings.lipstick.color_bgr,
            settings.lipstick.intensity,
            mode="multiply",
        )

    if settings.blush.enabled and settings.blush.intensity > 0:
        axes = (max(8, int(face_w * 0.13)), max(6, int(face_w * 0.09)))
        blur_k = max(21, face_w // 8)
        for indices in (LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES):
            points = get_region_points(landmarks, indices)
            mask = create_ellipse_mask(image.shape, points, axes)
            mask = soften_mask(mask, blur_k)
            result = apply_color(
                result,
                mask,
                settings.blush.color_bgr,
                settings.blush.intensity * 0.85,
                mode="overlay",
            )

    if settings.eyeshadow.enabled and settings.eyeshadow.intensity > 0:
        pairs = (
            (LEFT_UPPER_LID_INDICES, LEFT_BROW_INDICES, LEFT_EYE_INDICES),
            (RIGHT_UPPER_LID_INDICES, RIGHT_BROW_INDICES, RIGHT_EYE_INDICES),
        )
        for lid_idx, brow_idx, eye_idx in pairs:
            lid = get_region_points(landmarks, lid_idx)
            brow = get_region_points(landmarks, brow_idx)
            eye = get_region_points(landmarks, eye_idx)
            mask = create_eyeshadow_mask(image.shape, lid, brow, eye, lid_span=0.22)
            mask = soften_mask(mask, max(11, face_w // 22))
            result = apply_color(
                result,
                mask,
                settings.eyeshadow.color_bgr,
                settings.eyeshadow.intensity * 0.38,
                mode="mix",
            )

    return result
