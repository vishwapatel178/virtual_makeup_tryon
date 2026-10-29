"""Face landmark detection for still images and live video."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


class FaceDetector:
    """Reusable MediaPipe FaceLandmarker (IMAGE or VIDEO running mode)."""

    def __init__(self, model_path: str | Path | None = None, video: bool = False):
        model_path = str(model_path or MODEL_PATH)
        self._video = video
        running_mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.perf_counter()
        self._last_ts = -1

    def detect_bgr(self, frame_bgr: np.ndarray):
        """Return face_landmarks for a BGR OpenCV frame (or empty list)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        if self._video:
            timestamp_ms = int((time.perf_counter() - self._start) * 1000)
            if timestamp_ms <= self._last_ts:
                timestamp_ms = self._last_ts + 1
            self._last_ts = timestamp_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        return result.face_landmarks

    def close(self) -> None:
        self._landmarker.close()


def detect_face(image_path: str, model_path: str = "face_landmarker.task"):
    """Backward-compatible helper for a file path (used by early CLI work)."""
    resolved = Path(model_path)
    if not resolved.is_file():
        resolved = Path(__file__).resolve().parent / "face_landmarker.task"
    detector = FaceDetector(model_path=resolved, video=False)
    try:
        image = mp.Image.create_from_file(str(image_path))
        result = detector._landmarker.detect(image)
        return result.face_landmarks
    finally:
        detector.close()