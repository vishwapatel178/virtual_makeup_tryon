"""Still-image pipeline used for tests and offline checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detection import FaceDetector
from makeup import MakeupSettings, ProductState, apply_makeup
from palette import SWATCHES, rgb_to_bgr


def process_image(
    image_path: str | Path,
    output_path: str | Path,
    lipstick_index: int = 0,
    blush_index: int = 0,
    eyeshadow_index: int = 0,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = FaceDetector(video=False)
    try:
        landmarks = detector.detect_bgr(image)
    finally:
        detector.close()

    if not landmarks:
        raise RuntimeError("No face found in the image.")

    settings = MakeupSettings(
        lipstick=ProductState(True, rgb_to_bgr(SWATCHES["lipstick"][lipstick_index][1]), 0.65),
        blush=ProductState(True, rgb_to_bgr(SWATCHES["blush"][blush_index][1]), 0.40),
        eyeshadow=ProductState(True, rgb_to_bgr(SWATCHES["eyeshadow"][eyeshadow_index][1]), 0.45),
    )
    result = apply_makeup(image, landmarks, settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply virtual makeup to a still image.")
    parser.add_argument("image", help="Path to a face photo")
    parser.add_argument("-o", "--output", default="makeup_result.jpg")
    args = parser.parse_args()
    out = process_image(args.image, args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
