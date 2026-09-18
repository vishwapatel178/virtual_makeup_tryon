import cv2
import numpy as np

def create_lip_mask(image_shape, lip_points):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pixel_points = np.array([(int(x * w), int(y * h)) for x, y in lip_points])
    cv2.fillPoly(mask, [pixel_points], 255)

    return mask

def apply_lip_color(image, mask, color_bgr, opacity=0.5):
    colored_layer = np.zeros_like(image)
    colored_layer[:] = color_bgr

    colored_layer = cv2.bitwise_and(colored_layer, colored_layer, mask=mask)
    result = image.copy()

    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    result = (result * (1 - mask_3ch * opacity) + colored_layer * (mask_3ch * opacity)).astype(np.uint8)

    return result