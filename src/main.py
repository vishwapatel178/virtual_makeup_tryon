import cv2
from detection import detect_face
from landmarks import get_region_points, LIP_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES
from masking import create_lip_mask, apply_lip_color

image_path = input("Enter the path to your test image: ")
image = cv2.imread(image_path)

landmarks = detect_face(image_path)

if not landmarks:
    print("No face found.")
else:
    print(f"Face detected! Found {len(landmarks[0])} landmark points.")

    # Lipstick
    print("\n-- Lipstick color --")
    lip_r = int(input("Enter Red value (0-255): "))
    lip_g = int(input("Enter Green value (0-255): "))
    lip_b = int(input("Enter Blue value (0-255): "))

    lip_points = get_region_points(landmarks, LIP_INDICES)
    lip_mask = create_lip_mask(image.shape, lip_points)
    result = apply_lip_color(image, lip_mask, (lip_b, lip_g, lip_r), opacity=0.7)

    # Blush
    print("\n-- Blush color --")
    cheek_r = int(input("Enter Red value (0-255): "))
    cheek_g = int(input("Enter Green value (0-255): "))
    cheek_b = int(input("Enter Blue value (0-255): "))

    left_cheek_points = get_region_points(landmarks, LEFT_CHEEK_INDICES)
    right_cheek_points = get_region_points(landmarks, RIGHT_CHEEK_INDICES)

    left_mask = create_lip_mask(image.shape, left_cheek_points)
    right_mask = create_lip_mask(image.shape, right_cheek_points)

    result = apply_lip_color(result, left_mask, (cheek_b, cheek_g, cheek_r), opacity=0.5)
    result = apply_lip_color(result, right_mask, (cheek_b, cheek_g, cheek_r), opacity=0.5)

    cv2.imwrite("makeup_result.jpg", result)
    print("\nSaved result to makeup_result.jpg")