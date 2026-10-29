
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Get the model file from the same folder as detection.py
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


def detect_face(image_path, model_path=MODEL_PATH):
    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(image)

    detector.close()
    return result.face_landmarks


# --- Live video version ---
base_options_live = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)
options_live = vision.FaceLandmarkerOptions(
    base_options=base_options_live
)
detector_live = vision.FaceLandmarker.create_from_options(
    options_live
)


def detect_face_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector_live.detect(mp_image)
    return result.face_landmarks
