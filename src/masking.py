"""Advanced soft region masks and natural skin-aware color blending engine."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks import (
    FACE_OVAL_INDICES,
    INNER_LIP_INDICES,
    LEFT_BROW_INDICES,
    LEFT_EYE_CREASE_INDICES,
    LEFT_EYE_INDICES,
    LEFT_EYE_INNER_CORNER,
    LEFT_EYE_OUTER_CORNER,
    LEFT_UPPER_LID_INDICES,
    OUTER_LIP_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_EYE_CREASE_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_EYE_INNER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_UPPER_LID_INDICES,
    get_extended_forehead_points,
    get_region_points,
    landmark_xy,
)


def points_to_pixels(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Convert normalized (0..1) coordinate pairs to integer pixel coordinates."""
    return np.array([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def soften_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    """Apply Gaussian blur with odd kernel size to feather mask edges."""
    kernel = max(3, int(kernel) | 1)
    return cv2.GaussianBlur(mask, (kernel, kernel), 0)


def create_polygon_mask(image_shape: tuple[int, ...], points, hole_points=None) -> np.ndarray:
    """Fill a polygon on a single-channel 8-bit mask, optionally subtracting a hole."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pixel_points = points_to_pixels(points, w, h) if isinstance(points[0], (tuple, list)) else points
    if len(pixel_points) >= 3:
        cv2.fillPoly(mask, [pixel_points], 255)
    if hole_points is not None and len(hole_points) >= 3:
        hole = points_to_pixels(hole_points, w, h) if isinstance(hole_points[0], (tuple, list)) else hole_points
        cv2.fillPoly(mask, [hole], 0)
    return mask


def create_lip_mask(image_shape: tuple[int, ...], lip_points) -> np.ndarray:
    """Backward compatibility helper."""
    return create_polygon_mask(image_shape, lip_points)


def create_ellipse_mask(image_shape: tuple[int, ...], points, axes: tuple[int, int]) -> np.ndarray:
    """
    Generate an ultra-soft, naturally diffused blush mask with continuous Gaussian radial falloff.
    Diffuses seamlessly into skin pores and contours without any circular patch edges.
    """
    h, w = image_shape[:2]
    pixel_points = points_to_pixels(points, w, h)
    if len(pixel_points) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    center = tuple(np.mean(pixel_points, axis=0).astype(int))
    cx, cy = center
    rx, ry = float(max(2, axes[0])), float(max(2, axes[1]))

    # Bounding box for efficient calculation
    pad = int(max(rx, ry) * 2.2)
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)

    y_grid, x_grid = np.ogrid[y0:y1, x0:x1]
    norm_dist_sq = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2

    # Smooth Gaussian bell curve falloff: exp(-1.85 * norm_dist_sq)
    patch = np.exp(-1.85 * norm_dist_sq)
    patch = np.where(norm_dist_sq < 2.8, patch, 0.0)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = patch * 255.0

    # Multi-scale Gaussian softening
    blur_k = max(25, int(max(rx, ry) * 1.2)) | 1
    softened = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return np.clip(softened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Teeth and Mouth Interior Masking (Clean Matte Lipstick)
# ---------------------------------------------------------------------------

def create_skin_aware_lip_mask(
    image: np.ndarray,
    landmarks,
    face_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an outer lip mask strictly subtracting teeth and inner mouth cavity.
    Returns (lip_mask, empty_gloss_mask).
    """
    h, w = image.shape[:2]
    outer_pts = get_region_points(landmarks, OUTER_LIP_INDICES)
    inner_pts = get_region_points(landmarks, INNER_LIP_INDICES)

    outer_px = points_to_pixels(outer_pts, w, h)
    inner_px = points_to_pixels(inner_pts, w, h)

    # 1. Base lip polygon with inner cavity hole
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    if len(outer_px) >= 3:
        cv2.fillPoly(lip_mask, [outer_px], 255)
    if len(inner_px) >= 3:
        cv2.fillPoly(lip_mask, [inner_px], 0)

    # 2. Teeth / Oral cavity exclusion:
    if len(inner_px) >= 3:
        mouth_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mouth_roi_mask, [inner_px], 255)
        k_dilate = max(3, face_w // 70) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        dilated_mouth = cv2.dilate(mouth_roi_mask, kernel, iterations=1)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        teeth_or_cavity = ((val > 125) & (sat < 65)) | (val < 45)
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        exclusion_mask[(dilated_mouth > 0) & teeth_or_cavity] = 255
        exclusion_mask = cv2.GaussianBlur(exclusion_mask, (5, 5), 0)

        lip_mask = np.clip(lip_mask.astype(np.int16) - exclusion_mask.astype(np.int16), 0, 255).astype(np.uint8)

    # Clean vermilion border for matte lipstick (no fuzzy bleed outside lips)
    blur_k = max(3, face_w // 75) | 1
    lip_mask = cv2.GaussianBlur(lip_mask, (blur_k, blur_k), 0)

    # Gloss is removed; return empty gloss mask for backward compatibility
    gloss_mask = np.zeros((h, w), dtype=np.uint8)
    return lip_mask, gloss_mask


# ---------------------------------------------------------------------------
# Full Eyelid Eyeshadow Mask
# ---------------------------------------------------------------------------

def create_eyeshadow_mask(
    image_shape: tuple[int, ...],
    lid_points,
    crease_points,
    brow_points,
    eye_points,
) -> np.ndarray:
    """
    Rebuild full eyelid mask covering from upper lash line up to the crease
    with soft, graded edges that blend seamlessly into skin.
    """
    h, w = image_shape[:2]
    lid = np.array(lid_points, dtype=np.float32)
    brow = np.array(brow_points, dtype=np.float32)

    if len(lid) < 3 or len(brow) < 2:
        return np.zeros((h, w), dtype=np.uint8)

    span = 0.52
    upper_boundary = []
    for pt in lid:
        brow_match = brow[np.argmin(np.abs(brow[:, 0] - pt[0]))]
        upper_pt = (1.0 - span) * pt + span * brow_match
        upper_boundary.append(upper_pt)

    contour = np.vstack([np.asarray(upper_boundary), lid[::-1]])
    contour_px = points_to_pixels(contour.tolist(), w, h)
    eye_px = points_to_pixels(eye_points, w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(contour_px) >= 3:
        cv2.fillPoly(mask, [contour_px], 255)
    if len(eye_px) >= 3:
        cv2.fillPoly(mask, [eye_px], 0)

    lid_px = points_to_pixels(lid_points, w, h)
    line = np.zeros((h, w), dtype=np.uint8)
    if len(lid_px) >= 2:
        cv2.polylines(line, [lid_px], False, 255, 2)
    dist = cv2.distanceTransform(cv2.bitwise_not(line), cv2.DIST_L2, 5)

    brow_px = points_to_pixels(brow_points, w, h)
    brow_center = brow_px.mean(axis=0)
    lid_center = lid_px.mean(axis=0)
    lid_height = float(np.linalg.norm(brow_center - lid_center))
    falloff = max(8.0, lid_height * 0.75)

    weight = np.clip(1.0 - (dist / falloff) ** 1.3, 0.0, 1.0)
    faded = mask.astype(np.float32) * weight
    return np.clip(faded, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Smooth, Evenly Blended Eyeliner Shapes Generator
# ---------------------------------------------------------------------------

def _smooth_polyline(pts: np.ndarray, num_samples: int = 40) -> np.ndarray:
    """Sub-divide and smooth discrete landmark points into a silky continuous polyline."""
    if len(pts) < 3:
        return pts
    t_orig = np.linspace(0, 1, len(pts))
    t_fine = np.linspace(0, 1, num_samples)
    x_fine = np.interp(t_fine, t_orig, pts[:, 0])
    y_fine = np.interp(t_fine, t_orig, pts[:, 1])
    # Apply rolling average smoothing
    kernel = np.array([0.2, 0.6, 0.2])
    x_smooth = np.convolve(x_fine, kernel, mode="same")
    y_smooth = np.convolve(y_fine, kernel, mode="same")
    x_smooth[0], x_smooth[-1] = x_fine[0], x_fine[-1]
    y_smooth[0], y_smooth[-1] = y_fine[0], y_fine[-1]
    return np.column_stack([x_smooth, y_smooth]).astype(np.int32)


def create_eyeliner_mask(
    image_shape: tuple[int, ...],
    landmarks,
    style: str = "classic_wing",
    face_w: int = 150,
) -> np.ndarray:
    """
    Generate an evenly blended, anti-aliased eyeliner mask that hugs the lash line smoothly:
    - 'classic_wing': tapered lash line with sleek upward wing flick
    - 'tightline': delicate line hugging lash roots seamlessly
    - 'cat_eye': bold outer wing + subtle inner corner point
    - 'bold': smoky blended kohl liner
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    scale_factor = max(1.0, face_w / 200.0)

    pairs = [
        # Left eye: lash line outer -> inner, outer corner 33, inner 133, brow arch 105
        (LEFT_UPPER_LID_INDICES, LEFT_EYE_OUTER_CORNER, LEFT_EYE_INNER_CORNER, 70, 105, -1),
        # Right eye: lash line inner -> outer, inner 362, outer 263, brow arch 334
        (RIGHT_UPPER_LID_INDICES, RIGHT_EYE_OUTER_CORNER, RIGHT_EYE_INNER_CORNER, 300, 334, 1),
    ]

    for lid_indices, outer_idx, inner_idx, brow_tail_idx, brow_arch_idx, side_sign in pairs:
        lid_pts = get_region_points(landmarks, lid_indices)
        lid_px = points_to_pixels(lid_pts, w, h)
        if len(lid_px) < 3:
            continue

        # Smooth discrete points for an even, non-kinked lash curve
        smooth_line = _smooth_polyline(lid_px, num_samples=50)

        outer_pt = np.array(landmark_xy(landmarks, outer_idx, w, h), dtype=np.float32)
        inner_pt = np.array(landmark_xy(landmarks, inner_idx, w, h), dtype=np.float32)
        brow_tail = np.array(landmark_xy(landmarks, brow_tail_idx, w, h), dtype=np.float32)

        eye_vector = outer_pt - inner_pt
        eye_len = max(5.0, float(np.linalg.norm(eye_vector)))
        eye_dir = eye_vector / eye_len

        wing_vector = brow_tail - outer_pt
        wing_len = float(np.linalg.norm(wing_vector))
        wing_dir = (wing_vector / wing_len) if wing_len > 1e-3 else np.array([side_sign * 0.8, -0.6], dtype=np.float32)
        wing_dir = 0.7 * wing_dir + 0.3 * np.array([side_sign * 0.7, -0.7], dtype=np.float32)
        wing_dir /= np.linalg.norm(wing_dir)

        if style == "tightline":
            thickness = max(1, int(round(1.6 * scale_factor)))
            cv2.polylines(mask, [smooth_line], False, 255, thickness, cv2.LINE_AA)

        elif style == "classic_wing":
            thickness = max(2, int(round(2.4 * scale_factor)))
            cv2.polylines(mask, [smooth_line], False, 255, thickness, cv2.LINE_AA)

            wing_ext = eye_len * 0.28
            wing_tip = outer_pt + wing_dir * wing_ext
            wing_poly = np.array([
                outer_pt - eye_dir * (scale_factor * 2.5),
                outer_pt + np.array([-wing_dir[1], wing_dir[0]]) * (scale_factor * 1.5),
                wing_tip,
                outer_pt,
            ], dtype=np.int32)
            cv2.fillPoly(mask, [wing_poly], 255, cv2.LINE_AA)

        elif style == "cat_eye":
            thickness = max(2, int(round(3.0 * scale_factor)))
            cv2.polylines(mask, [smooth_line], False, 255, thickness, cv2.LINE_AA)

            wing_ext = eye_len * 0.38
            wing_tip = outer_pt + wing_dir * wing_ext
            wing_poly = np.array([
                outer_pt - eye_dir * (scale_factor * 3.5),
                outer_pt + np.array([-wing_dir[1], wing_dir[0]]) * (scale_factor * 2.2),
                wing_tip,
                outer_pt,
            ], dtype=np.int32)
            cv2.fillPoly(mask, [wing_poly], 255, cv2.LINE_AA)

            inner_flick_dir = -eye_dir + np.array([0.0, 0.25], dtype=np.float32)
            inner_flick_dir /= np.linalg.norm(inner_flick_dir)
            inner_tip = inner_pt + inner_flick_dir * (scale_factor * 3.0)
            inner_poly = np.array([
                inner_pt + eye_dir * 2.0,
                inner_tip,
                inner_pt,
            ], dtype=np.int32)
            cv2.fillPoly(mask, [inner_poly], 255, cv2.LINE_AA)

        else:  # bold / bold_smudged
            thickness = max(3, int(round(4.0 * scale_factor)))
            cv2.polylines(mask, [smooth_line], False, 255, thickness, cv2.LINE_AA)
            wing_ext = eye_len * 0.22
            wing_tip = outer_pt + wing_dir * wing_ext
            wing_poly = np.array([
                outer_pt - eye_dir * (scale_factor * 3.0),
                wing_tip,
                outer_pt,
            ], dtype=np.int32)
            cv2.fillPoly(mask, [wing_poly], 255, cv2.LINE_AA)

    # Evenly diffused Gaussian feathering along the lash line
    blur_k = max(3, int(scale_factor * 2.5) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return mask


# ---------------------------------------------------------------------------
# Foundation / Skin Tint (Full Forehead Coverage with Seamless Hairline Falloff)
# ---------------------------------------------------------------------------

def create_foundation_mask(image_shape: tuple[int, ...], landmarks, face_w: int) -> np.ndarray:
    """
    Construct full face mask with extended forehead reaching up to the hairline,
    strictly cutting out eyes, eyebrows, and lips, with seamless zero-edge hairline feathering.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    ext_forehead = get_extended_forehead_points(landmarks, w, h, extension_ratio=0.48)
    face_contour_pts = [landmark_xy(landmarks, idx, w, h) for idx in FACE_OVAL_INDICES]

    combined_face = ext_forehead + face_contour_pts
    hull = cv2.convexHull(np.array(combined_face, dtype=np.int32))
    cv2.fillPoly(mask, [hull], 255)

    for eye_idx in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        eye_px = points_to_pixels(get_region_points(landmarks, eye_idx), w, h)
        if len(eye_px) >= 3:
            eye_hull = cv2.convexHull(eye_px)
            cv2.fillPoly(mask, [eye_hull], 0)

    for brow_idx in (LEFT_BROW_INDICES, RIGHT_BROW_INDICES):
        brow_px = points_to_pixels(get_region_points(landmarks, brow_idx), w, h)
        if len(brow_px) >= 3:
            brow_hull = cv2.convexHull(brow_px)
            cv2.fillPoly(mask, [brow_hull], 0)

    lip_px = points_to_pixels(get_region_points(landmarks, OUTER_LIP_INDICES), w, h)
    if len(lip_px) >= 3:
        lip_hull = cv2.convexHull(lip_px)
        cv2.fillPoly(mask, [lip_hull], 0)

    hairline_y_min = float(min(pt[1] for pt in ext_forehead))
    brow_y_max = float(max(landmark_xy(landmarks, idx, w, h)[1] for idx in LEFT_BROW_INDICES + RIGHT_BROW_INDICES))
    forehead_span = max(10.0, brow_y_max - hairline_y_min)

    y_indices = np.arange(h, dtype=np.float32)
    vert_ramp = np.ones(h, dtype=np.float32)
    above_brow = y_indices < brow_y_max
    norm_y = np.clip((y_indices - hairline_y_min) / forehead_span, 0.0, 1.0)
    vert_ramp[above_brow] = (np.sin(norm_y[above_brow] * (np.pi / 2.0)) ** 2.2)
    ramp_2d = vert_ramp[:, np.newaxis]

    modulated = (mask.astype(np.float32) * ramp_2d)

    blur_k = max(31, face_w // 5) | 1
    softened = cv2.GaussianBlur(modulated, (blur_k, blur_k), 0)
    return np.clip(softened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Eyebrow Fill Mask
# ---------------------------------------------------------------------------

def create_brow_mask(image_shape: tuple[int, ...], landmarks, face_w: int) -> np.ndarray:
    """Generate soft, natural feathered mask for both eyebrows respecting hair shape."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for brow_indices in (LEFT_BROW_INDICES, RIGHT_BROW_INDICES):
        pts = get_region_points(landmarks, brow_indices)
        px = points_to_pixels(pts, w, h)
        if len(px) >= 3:
            cv2.fillPoly(mask, [px], 255)

    blur_k = max(9, face_w // 22) | 1
    return cv2.GaussianBlur(mask, (blur_k, blur_k), 0)


# ---------------------------------------------------------------------------
# Realistic Color Blending Engine
# ---------------------------------------------------------------------------

def apply_color(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    opacity: float,
    mode: str = "overlay",
) -> np.ndarray:
    """
    Blend color into image using skin-luminance preserving transfer.
    Supported modes:
    - 'multiply': darkens naturally, preserves skin folds and hair texture
    - 'overlay': contrasts and enriches color while preserving highlights & shadows
    - 'soft_light': subtle, soft, sheer cosmetic wash
    - 'color': strictly retains underlying skin lightness (L in LAB), transfers hue/chroma
    - 'mix': linear alpha composite
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0 or mask is None:
        return image

    base = image.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0) * opacity
    alpha_3 = cv2.merge([alpha, alpha, alpha])

    if mode == "multiply":
        blended = (base / 255.0) * color
        blended = np.clip(blended, 0.0, 255.0)

    elif mode == "overlay":
        base_n = base / 255.0
        color_n = color / 255.0
        overlay = np.where(
            base_n < 0.5,
            2.0 * base_n * color_n,
            1.0 - 2.0 * (1.0 - base_n) * (1.0 - color_n),
        )
        blended = np.clip(overlay * 255.0, 0.0, 255.0)

    elif mode == "soft_light":
        base_n = base / 255.0
        color_n = color / 255.0
        soft = np.where(
            color_n <= 0.5,
            base_n - (1.0 - 2.0 * color_n) * base_n * (1.0 - base_n),
            base_n + (2.0 * color_n - 1.0) * (np.sqrt(np.maximum(base_n, 0.0)) - base_n),
        )
        blended = np.clip(soft * 255.0, 0.0, 255.0)

    elif mode == "color":
        base_u8 = np.clip(base, 0, 255).astype(np.uint8)
        lab_base = cv2.cvtColor(base_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        color_img = np.full_like(base_u8, color_bgr, dtype=np.uint8)
        lab_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        lab_blended = lab_base.copy()
        lab_blended[:, :, 1] = lab_color[:, :, 1]
        lab_blended[:, :, 2] = lab_color[:, :, 2]

        blended_bgr = cv2.cvtColor(np.clip(lab_blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        blended = blended_bgr.astype(np.float32)

    elif mode == "screen":
        base_n = base / 255.0
        color_n = color / 255.0
        screen = 1.0 - (1.0 - base_n) * (1.0 - color_n)
        blended = np.clip(screen * 255.0, 0.0, 255.0)

    else:  # 'mix'
        colored = np.empty_like(base)
        colored[:] = color
        blended = colored

    result = base * (1.0 - alpha_3) + blended * alpha_3
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def apply_gloss(image: np.ndarray, gloss_mask: np.ndarray, gloss_intensity: float = 0.0) -> np.ndarray:
    """Deprecated / no-op for matte finish."""
    return image


def apply_lip_color(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], opacity: float = 0.5) -> np.ndarray:
    """Backward compatibility helper used by tests and CLI."""
    return apply_color(image, mask, color_bgr, opacity, mode="mix")
