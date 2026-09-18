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