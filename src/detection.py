import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_face(image_path, model_path="face_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options)
    detector = vision.FaceLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    return result.face_landmarks  # empty list if no face found


# --- Live video version ---
base_options_live = python.BaseOptions(model_asset_path="face_landmarker.task")
options_live = vision.FaceLandmarkerOptions(base_options=base_options_live)
detector_live = vision.FaceLandmarker.create_from_options(options_live)

def detect_face_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector_live.detect(mp_image)
    return result.face_landmarks