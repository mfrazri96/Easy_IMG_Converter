from typing import Optional

try:
    import cv2
except Exception:
    cv2 = None

from easy_img_converter.services.output_naming import build_enhance_output_path


def validate_enhance_ready(model_path: str) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-contrib-python.")

    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError("cv2.dnn_superres not found. Use opencv-contrib-python.")

    if not model_path:
        raise RuntimeError("Model path is empty.")

    import os

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")


def build_upsampler(model_path: str, model_name: str, scale: int):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name.lower(), int(scale))
    return sr


def is_blurry(bgr_img, threshold: float):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var


def process_enhance(
    input_file: str,
    output_dir: str,
    upsampler,
    model_name: str,
    scale: int,
    auto_check_blur: bool,
    blur_threshold: float,
) -> str:
    image = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {input_file}")

    if auto_check_blur:
        blurry, _score = is_blurry(image, blur_threshold)
        if not blurry:
            return "skipped"

    result = upsampler.upsample(image)
    output_path = build_enhance_output_path(input_file, output_dir, model_name, scale)
    ok = cv2.imwrite(output_path, result)
    if not ok:
        raise ValueError(f"Failed to save enhanced image: {output_path}")
    return "done"
