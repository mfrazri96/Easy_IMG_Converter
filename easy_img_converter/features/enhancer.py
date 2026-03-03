import os

try:
    import cv2
except Exception:
    cv2 = None

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except Exception:
    RealESRGANer = None
    RRDBNet = None

from easy_img_converter.services.output_naming import build_enhance_output_path


MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "arch": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        ),
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "arch": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        ),
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "arch": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        ),
    },
}


def validate_enhance_ready(weights_path: str, model_name: str) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-python.")

    if RealESRGANer is None or RRDBNet is None:
        raise RuntimeError("Real-ESRGAN dependencies missing. Install realesrgan and basicsr.")

    if model_name not in MODEL_CONFIGS:
        raise RuntimeError(f"Unsupported Real-ESRGAN model: {model_name}")

    if not weights_path:
        raise RuntimeError("Weights path is empty.")

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")


def build_upsampler(weights_path: str, model_name: str, tile: int = 400, use_half: bool = False):
    config = MODEL_CONFIGS[model_name]
    return RealESRGANer(
        scale=config["scale"],
        model_path=weights_path,
        model=config["arch"](),
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
    )


def process_enhance(input_file: str, output_dir: str, upsampler, model_name: str, outscale: int) -> str:
    image = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {input_file}")

    output, _ = upsampler.enhance(image, outscale=outscale)
    output_path = build_enhance_output_path(input_file, output_dir, model_name, outscale)

    ok = cv2.imwrite(output_path, output)
    if not ok:
        raise ValueError(f"Failed to save enhanced image: {output_path}")

    return "done"
