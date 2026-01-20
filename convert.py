from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("/path/to/input")
OUTPUT_DIR = Path("/path/to/output")


def convert(tiff_path: Path):
    try:
        out_path = OUTPUT_DIR / tiff_path.relative_to(INPUT_DIR).with_suffix(".jp2")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(tiff_path) as img:
            img.save(out_path, "JPEG2000", quality_mode="lossless")
        print(f"OK: {tiff_path.name}")
    except Exception as e:
        print(f"FAIL: {tiff_path.name} - {e}")


if __name__ == "__main__":
    files = list(INPUT_DIR.rglob("*.tif")) + list(INPUT_DIR.rglob("*.tiff"))
    print(f"Converting {len(files)} files...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(convert, files))
    print("Done")
