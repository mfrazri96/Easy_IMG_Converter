from PIL import Image
from pathlib import Path

from easy_img_converter.services.output_naming import safe_output_path


def prepare_image_for_format(image, save_format: str):
    if save_format in {"JPEG", "BMP"} and image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    if save_format == "JPEG" and image.mode != "RGB":
        return image.convert("RGB")

    if save_format == "ICO" and image.mode != "RGBA":
        return image.convert("RGBA")

    return image


def process_convert(input_file: str, output_dir: str, save_format: str, extension: str, quality: int) -> None:
    with Image.open(input_file) as img:
        converted_img = prepare_image_for_format(img, save_format)
        output_path = safe_output_path(output_dir, Path(input_file).stem, extension)

        save_kwargs = {}
        if save_format in {"JPEG", "WEBP"}:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        if save_format == "PNG":
            save_kwargs["optimize"] = True

        converted_img.save(output_path, save_format, **save_kwargs)
