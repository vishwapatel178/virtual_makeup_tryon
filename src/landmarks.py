LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_CHEEK_INDICES = [345, 346, 347, 330, 266, 425, 411, 352]
RIGHT_CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]

LIP_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

def get_region_points(landmarks, indices):
    face = landmarks[0]
    return [(face[i].x, face[i].y) for i in indices]