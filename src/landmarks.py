"""MediaPipe Face Mesh (468) region indices and geometric helpers for makeup masks."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Lip Landmark Indices
# ---------------------------------------------------------------------------

# Outer lip contour (clockwise).
OUTER_LIP_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Inner lip contour — subtracted so the mouth opening & teeth stay clear.
INNER_LIP_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Lower lip center points (for realistic gloss/highlight)
LOWER_LIP_GLOSS_INDICES = [14, 17, 84, 181, 314, 405]
UPPER_LIP_CUPID_INDICES = [0, 37, 267, 39, 269]

# Backward compatibility with earlier pipeline
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]

# ---------------------------------------------------------------------------
# Cheek & Face Contour Indices
# ---------------------------------------------------------------------------

CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]
RIGHT_CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]  # Observer left
LEFT_CHEEK_INDICES = [345, 346, 347, 330, 266, 425, 411, 352]  # Observer right

# Complete 36-point face silhouette
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

# Forehead arch anchor points for full-forehead extension to hairline
FOREHEAD_TOP_INDICES = [103, 67, 109, 10, 338, 297, 332]
MID_EYES_IDX = 168
NOSE_BRIDGE_IDX = 6
CHIN_IDX = 152

# ---------------------------------------------------------------------------
# Eye & Eyelid Indices
# ---------------------------------------------------------------------------

LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]

# Upper lash line (from outer corner to inner corner)
LEFT_UPPER_LID_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133]
RIGHT_UPPER_LID_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362]

# Full lid crease & upper socket (for deep, full-coverage eyeshadow)
LEFT_EYE_CREASE_INDICES = [247, 30, 29, 27, 28, 56, 190, 243, 112, 26]
RIGHT_EYE_CREASE_INDICES = [467, 260, 259, 257, 258, 286, 414, 463, 341, 256]

# Eye corners for eyeliner wing calculations
LEFT_EYE_OUTER_CORNER = 33
LEFT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 263
RIGHT_EYE_INNER_CORNER = 362

# ---------------------------------------------------------------------------
# Eyebrow Indices
# ---------------------------------------------------------------------------

# Full closed eyebrow contours (top edge + bottom edge)
LEFT_BROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Eyebrow arch anchors
LEFT_BROW_ARCH = 105
RIGHT_BROW_ARCH = 334

# ---------------------------------------------------------------------------
# Highlighter Indices
# ---------------------------------------------------------------------------

LEFT_CHEEK_HIGHLIGHT = [116, 117, 123, 147, 213, 192]
RIGHT_CHEEK_HIGHLIGHT = [345, 346, 352, 376, 433, 416]
NOSE_HIGHLIGHT = [168, 6, 197, 195, 5, 4, 1]
BROW_BONE_LEFT = [105, 66, 107]
BROW_BONE_RIGHT = [334, 296, 336]

# Face width references
FACE_LEFT_IDX = 234
FACE_RIGHT_IDX = 454


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_region_points(landmarks, indices: list[int]) -> list[tuple[float, float]]:
    """Return [(x, y), ...] normalized coordinates for the first face in landmarks."""
    face = landmarks[0]
    return [(face[i].x, face[i].y) for i in indices]


def get_region_points_3d(landmarks, indices: list[int]) -> list[tuple[float, float, float]]:
    """Return [(x, y, z), ...] normalized coordinates for the first face in landmarks."""
    face = landmarks[0]
    return [(face[i].x, face[i].y, getattr(face[i], "z", 0.0)) for i in indices]


def landmark_xy(landmarks, index: int, width: int, height: int) -> tuple[int, int]:
    """Return pixel (x, y) coordinate for landmark index."""
    point = landmarks[0][index]
    return int(point.x * width), int(point.y * height)


def face_width_px(landmarks, width: int, height: int) -> int:
    """Return pixel distance between left and right temples/cheeks."""
    left = landmark_xy(landmarks, FACE_LEFT_IDX, width, height)
    right = landmark_xy(landmarks, FACE_RIGHT_IDX, width, height)
    return max(1, int(((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5))


def face_height_px(landmarks, width: int, height: int) -> int:
    """Return pixel distance between forehead top (landmark 10) and chin (landmark 152)."""
    top = landmark_xy(landmarks, 10, width, height)
    bottom = landmark_xy(landmarks, CHIN_IDX, width, height)
    return max(1, int(((bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2) ** 0.5))


def get_extended_forehead_points(
    landmarks,
    width: int,
    height: int,
    extension_ratio: float = 0.42,
) -> list[tuple[int, int]]:
    """
    Extrapolate forehead arch points upward towards the hairline so foundation/tint
    covers the complete forehead area naturally.
    """
    face = landmarks[0]
    mid_eyes = np.array([face[MID_EYES_IDX].x * width, face[MID_EYES_IDX].y * height], dtype=np.float32)
    top_mid = np.array([face[10].x * width, face[10].y * height], dtype=np.float32)
    
    # Vector from mid-eyes to top of mesh forehead
    forehead_vec = top_mid - mid_eyes
    forehead_len = max(5.0, float(np.linalg.norm(forehead_vec)))
    ext_dist = forehead_len * extension_ratio

    extended_points = []
    for idx in FOREHEAD_TOP_INDICES:
        pt = np.array([face[idx].x * width, face[idx].y * height], dtype=np.float32)
        # Vector pointing radially away from center of eyes/nose
        radial = pt - mid_eyes
        r_norm = np.linalg.norm(radial)
        if r_norm > 1e-3:
            radial_dir = radial / r_norm
        else:
            radial_dir = np.array([0.0, -1.0], dtype=np.float32)
        
        # Upward bias
        up_dir = np.array([0.0, -1.0], dtype=np.float32)
        combined_dir = 0.65 * radial_dir + 0.35 * up_dir
        combined_norm = np.linalg.norm(combined_dir)
        if combined_norm > 1e-3:
            combined_dir /= combined_norm
            
        ext_pt = pt + combined_dir * ext_dist
        extended_points.append((int(np.clip(ext_pt[0], 0, width - 1)), int(np.clip(ext_pt[1], 0, height - 1))))
        
    return extended_points