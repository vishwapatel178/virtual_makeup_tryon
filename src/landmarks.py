LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]

CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]

LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_CHEEK_INDICES = [116, 117, 118, 101, 36, 205, 187, 123]
LEFT_CHEEK_INDICES = [345, 346, 347, 330, 266, 425, 411, 352]

def get_region_points(landmarks, indices):
    face = landmarks[0]
    return [(face[i].x, face[i].y) for i in indices]