"""Apply foundation, blush, eyeshadow, eyeliner, brow fill, and matte lipstick."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from landmarks import (
    LEFT_BROW_INDICES,
    LEFT_CHEEK_INDICES,
    LEFT_EYE_CREASE_INDICES,
    LEFT_EYE_INDICES,
    LEFT_UPPER_LID_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_CHEEK_INDICES,
    RIGHT_EYE_CREASE_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_UPPER_LID_INDICES,
    face_width_px,
    get_region_points,
)
from masking import (
    apply_color,
    create_brow_mask,
    create_ellipse_mask,
    create_eyeliner_mask,
    create_eyeshadow_mask,
    create_foundation_mask,
    create_skin_aware_lip_mask,
    soften_mask,
)


@dataclass
class ProductState:
    enabled: bool = True
    color_bgr: tuple[int, int, int] = (80, 60, 180)
    intensity: float = 0.40
    style: str = "classic_wing"  # For eyeliner: classic_wing, tightline, bold, cat_eye
    blend_mode: str = "multiply"  # For lipstick: multiply (matte), overlay, color


@dataclass
class MakeupSettings:
    foundation: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=False,
            color_bgr=(186, 210, 240),  # Light neutral
            intensity=0.28,
        )
    )
    blush: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=True,
            color_bgr=(144, 160, 232),  # Peach Blossom
            intensity=0.28,
        )
    )
    eyeshadow: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=True,
            color_bgr=(132, 164, 196),  # Nude
            intensity=0.32,
        )
    )
    eyeliner: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=False,
            color_bgr=(20, 18, 18),  # Carbon Black
            intensity=0.80,
            style="classic_wing",
        )
    )
    brow: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=True,
            color_bgr=(42, 52, 75),  # Soft Brown
            intensity=0.25,  # Soft, natural and smooth
        )
    )
    lipstick: ProductState = field(
        default_factory=lambda: ProductState(
            enabled=True,
            color_bgr=(58, 30, 196),  # Classic Red
            intensity=0.60,
            blend_mode="multiply",  # Pure matte pigment
        )
    )


def apply_makeup(image: np.ndarray, landmarks, settings: MakeupSettings) -> np.ndarray:
    """Apply enabled makeup products in proper aesthetic layering sequence with high-speed ROI processing."""
    if not landmarks:
        return image

    result = image.copy()
    h, w = image.shape[:2]
    face_w = face_width_px(landmarks, w, h)

    # -----------------------------------------------------------------------
    # 1. Foundation / Skin Tint (Full Forehead + Face with seamless hairline falloff)
    # -----------------------------------------------------------------------
    if settings.foundation.enabled and settings.foundation.intensity > 0:
        fnd_mask = create_foundation_mask(image.shape, landmarks, face_w)
        nz = cv2.findNonZero(fnd_mask)
        if nz is not None:
            fx, fy, fw, fh = cv2.boundingRect(nz)
            roi_img = result[fy : fy + fh, fx : fx + fw]
            roi_mask = fnd_mask[fy : fy + fh, fx : fx + fw]
            blurred_skin = cv2.bilateralFilter(roi_img, d=7, sigmaColor=35, sigmaSpace=35)
            alpha_fnd = (roi_mask.astype(np.float32) * (settings.foundation.intensity * 0.38 / 255.0))[:, :, np.newaxis]
            smoothed_base = roi_img.astype(np.float32) * (1.0 - alpha_fnd) + blurred_skin.astype(np.float32) * alpha_fnd
            result[fy : fy + fh, fx : fx + fw] = np.clip(smoothed_base, 0, 255).astype(np.uint8)

            result = apply_color(
                result,
                fnd_mask,
                settings.foundation.color_bgr,
                settings.foundation.intensity * 0.32,
                mode="soft_light",
            )

    # -----------------------------------------------------------------------
    # 2. Blush (Naturally Diffused & Softly Blended into Cheek Skin)
    # -----------------------------------------------------------------------
    if settings.blush.enabled and settings.blush.intensity > 0:
        axes = (max(12, int(face_w * 0.15)), max(9, int(face_w * 0.11)))
        blush_mask = np.zeros((h, w), dtype=np.uint8)
        for indices in (LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES):
            points = get_region_points(landmarks, indices)
            mask = create_ellipse_mask(image.shape, points, axes)
            cv2.bitwise_or(blush_mask, mask, blush_mask)
        result = apply_color(
            result,
            blush_mask,
            settings.blush.color_bgr,
            settings.blush.intensity * 0.50,
            mode="soft_light",  # Diffused soft wash
        )

    # -----------------------------------------------------------------------
    # 3. Eyeshadow (Full Lid up to Crease with soft gradient)
    # -----------------------------------------------------------------------
    if settings.eyeshadow.enabled and settings.eyeshadow.intensity > 0:
        pairs = (
            (LEFT_UPPER_LID_INDICES, LEFT_EYE_CREASE_INDICES, LEFT_BROW_INDICES, LEFT_EYE_INDICES),
            (RIGHT_UPPER_LID_INDICES, RIGHT_EYE_CREASE_INDICES, RIGHT_BROW_INDICES, RIGHT_EYE_INDICES),
        )
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        blur_k = max(9, face_w // 26)
        for lid_idx, crease_idx, brow_idx, eye_idx in pairs:
            lid = get_region_points(landmarks, lid_idx)
            crease = get_region_points(landmarks, crease_idx)
            brow = get_region_points(landmarks, brow_idx)
            eye = get_region_points(landmarks, eye_idx)
            mask = create_eyeshadow_mask(image.shape, lid, crease, brow, eye)
            mask = soften_mask(mask, blur_k)
            cv2.bitwise_or(shadow_mask, mask, shadow_mask)

        result = apply_color(
            result,
            shadow_mask,
            settings.eyeshadow.color_bgr,
            settings.eyeshadow.intensity * 0.50,
            mode="overlay",
        )

    # -----------------------------------------------------------------------
    # 4. Eyeliner (Smooth, Evenly Blended Lash Line: wing, tightline, bold, cat-eye)
    # -----------------------------------------------------------------------
    if settings.eyeliner.enabled and settings.eyeliner.intensity > 0:
        el_mask = create_eyeliner_mask(
            image.shape,
            landmarks,
            style=settings.eyeliner.style,
            face_w=face_w,
        )
        result = apply_color(
            result,
            el_mask,
            settings.eyeliner.color_bgr,
            settings.eyeliner.intensity,
            mode="multiply",
        )

    # -----------------------------------------------------------------------
    # 5. Brow Fill (Softer, Less Intense, Natural and Smooth throughout)
    # -----------------------------------------------------------------------
    if settings.brow.enabled and settings.brow.intensity > 0:
        brow_mask = create_brow_mask(image.shape, landmarks, face_w)
        result = apply_color(
            result,
            brow_mask,
            settings.brow.color_bgr,
            settings.brow.intensity * 0.35,  # Soft, natural hair tint
            mode="multiply",
        )

    # -----------------------------------------------------------------------
    # 6. Lipstick (Pure Matte Finish: Clean Vermilion Border, Teeth Safe, No Gloss)
    # -----------------------------------------------------------------------
    if settings.lipstick.enabled and settings.lipstick.intensity > 0:
        lip_mask, _ = create_skin_aware_lip_mask(result, landmarks, face_w)
        lip_mode = getattr(settings.lipstick, "blend_mode", "multiply")
        result = apply_color(
            result,
            lip_mask,
            settings.lipstick.color_bgr,
            settings.lipstick.intensity,
            mode=lip_mode,
        )

    return result
