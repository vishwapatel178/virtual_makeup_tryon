
import cv2
from detection import detect_face_from_frame
from landmarks import (
    get_region_points,
    LIP_INDICES,
    LEFT_CHEEK_INDICES,
    RIGHT_CHEEK_INDICES
)
from masking import create_lip_mask, apply_lip_color

# Fixed test colors (BGR order)
LIPSTICK_COLOR = (60, 30, 180)
BLUSH_COLOR = (150, 140, 230)

WINDOW_NAME = "Live Virtual Try-On"

cap = cv2.VideoCapture(0)

try:
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            landmarks_list = detect_face_from_frame(frame)

            if landmarks_list:
                # Lips
                lip_points = get_region_points(
                    landmarks_list, LIP_INDICES
                )
                lip_mask = create_lip_mask(frame.shape, lip_points)
                frame = apply_lip_color(
                    frame, lip_mask, LIPSTICK_COLOR, opacity=0.5
                )

                # Left cheek
                left_points = get_region_points(
                    landmarks_list, LEFT_CHEEK_INDICES
                )
                left_mask = create_lip_mask(frame.shape, left_points)
                frame = apply_lip_color(
                    frame, left_mask, BLUSH_COLOR, opacity=0.3
                )

                # Right cheek
                right_points = get_region_points(
                    landmarks_list, RIGHT_CHEEK_INDICES
                )
                right_mask = create_lip_mask(frame.shape, right_points)
                frame = apply_lip_color(
                    frame, right_mask, BLUSH_COLOR, opacity=0.3
                )

            cv2.imshow(WINDOW_NAME, frame)

            # Exit when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Exit when the window's X button is clicked
            if cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
