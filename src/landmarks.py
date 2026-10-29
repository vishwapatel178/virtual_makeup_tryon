"""MediaPipe Face Mesh (468) region indices used for makeup masks."""

# Outer lip contour (clockwise).
OUTER_LIP_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Inner lip contour — subtracted so the mouth opening stays clear.
INNER_LIP_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Kept for compatibility with the original CLI pipeline.
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]

CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]
RIGHT_CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]
LEFT_CHEEK_INDICES = [345, 346, 347, 330, 266, 425, 411, 352]

LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]

# Upper lash line (outer → inner). Eyeshadow sits just above this, not up to the brow.
LEFT_UPPER_LID_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133]
RIGHT_UPPER_LID_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362]

# Brow used only to measure lid height so color stops well below the eyebrow.
LEFT_BROW_INDICES = [70, 63, 105, 66, 107]
RIGHT_BROW_INDICES = [300, 293, 334, 296, 336]

FACE_LEFT_IDX = 234
FACE_RIGHT_IDX = 454


def get_region_points(landmarks, indices):
    face = landmarks[0]
    return [(face[i].x, face[i].y) for i in indices]


def landmark_xy(landmarks, index: int, width: int, height: int) -> tuple[int, int]:
    point = landmarks[0][index]
    return int(point.x * width), int(point.y * height)


def face_width_px(landmarks, width: int, height: int) -> int:
    left = landmark_xy(landmarks, FACE_LEFT_IDX, width, height)
    right = landmark_xy(landmarks, FACE_RIGHT_IDX, width, height)
    return max(1, int(((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5))