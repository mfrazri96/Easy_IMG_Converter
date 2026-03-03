import tempfile
import unittest
from pathlib import Path

from PIL import Image

from easy_img_converter.features.converter import process_convert
from easy_img_converter.services.output_naming import build_enhance_output_path, safe_output_path


class CoreSmokeTests(unittest.TestCase):
    def test_safe_output_path_adds_suffix_when_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "sample.png"
            first.write_bytes(b"dummy")
            path = safe_output_path(tmp, "sample", ".png")
            self.assertTrue(path.endswith("sample_1.png"))

    def test_build_enhance_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = build_enhance_output_path("photo.jpg", tmp, "EDSR", 2)
            self.assertTrue(output.endswith("photo_enhanced_edsr_x2.png"))

    def test_process_convert_png_to_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.png"
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()

            img = Image.new("RGBA", (40, 40), (255, 0, 0, 120))
            img.save(src, "PNG")

            process_convert(
                input_file=str(src),
                output_dir=str(out_dir),
                save_format="JPEG",
                extension=".jpg",
                quality=90,
            )

            outputs = list(out_dir.glob("*.jpg"))
            self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
