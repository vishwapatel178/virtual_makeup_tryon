import cv2
from detection import detect_face_from_frame
from landmarks import get_region_points, LIP_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES
from masking import create_lip_mask, apply_lip_color

# Fixed test colors for now (BGR order) — palette/slider comes in Step 4
LIPSTICK_COLOR = (60, 30, 180)   # a red lipstick
BLUSH_COLOR = (150, 140, 230)    # a natural pink

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks_list = detect_face_from_frame(frame)

    if landmarks_list:
        # Lips
        lip_points = get_region_points(landmarks_list, LIP_INDICES)
        lip_mask = create_lip_mask(frame.shape, lip_points)
        frame = apply_lip_color(frame, lip_mask, LIPSTICK_COLOR, opacity=0.5)

        # Left cheek
        left_points = get_region_points(landmarks_list, LEFT_CHEEK_INDICES)
        left_mask = create_lip_mask(frame.shape, left_points)
        frame = apply_lip_color(frame, left_mask, BLUSH_COLOR, opacity=0.3)

        # Right cheek
        right_points = get_region_points(landmarks_list, RIGHT_CHEEK_INDICES)
        right_mask = create_lip_mask(frame.shape, right_points)
        frame = apply_lip_color(frame, right_mask, BLUSH_COLOR, opacity=0.3)

    cv2.imshow("Live Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()