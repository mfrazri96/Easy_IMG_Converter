from pathlib import Path


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{int(size_bytes)} B"


def safe_output_path(output_dir: str, stem: str, extension: str) -> str:
    candidate = Path(output_dir) / f"{stem}{extension}"
    counter = 1
    while candidate.exists():
        candidate = Path(output_dir) / f"{stem}_{counter}{extension}"
        counter += 1
    return str(candidate)


def build_enhance_output_path(input_file: str, output_dir: str, model_name: str, scale: int) -> str:
    stem = f"{Path(input_file).stem}_enhanced_{model_name.lower()}_x{scale}"
    return safe_output_path(output_dir, stem, ".png")
