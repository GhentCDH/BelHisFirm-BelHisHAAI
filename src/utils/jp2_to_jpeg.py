import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

NUM_THREADS = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

JP2_EXTENSIONS = {".jp2"}


def convert_jp2_to_jpeg(input_path: Path, output_path: Path):
    try:
        with Image.open(input_path) as img:
            if hasattr(img, "n_frames") and img.n_frames > 1:
                logger.warning(
                    f"Multi-frame JP2 detected ({img.n_frames} frames): {input_path}. "
                    "Converting first frame only."
                )

            if img.mode in {"RGBA", "LA"}:
                # JPEG does not support alpha; flatten to white background.
                background = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.split()[-1]
                base = img.convert("RGB")
                background.paste(base, mask=alpha)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, "JPEG", quality=95)
            logger.info(f"Converted: {input_path} -> {output_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def convert_directory(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    jp2_files = [
        f for f in input_dir.rglob("*") if f.suffix.lower() in JP2_EXTENSIONS and f.is_file()
    ]

    if not jp2_files:
        logger.warning(f"No JP2 files found in {input_dir}")
        return 0, 0

    logger.info(f"Found {len(jp2_files)} JP2 file(s) to convert")

    conversion_pairs = []
    for jp2_path in jp2_files:
        relative_path = jp2_path.relative_to(input_dir)
        jpeg_path = output_dir / relative_path.with_suffix(".jpeg")
        conversion_pairs.append((jp2_path, jpeg_path))

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(
            executor.map(lambda p: convert_jp2_to_jpeg(p[0], p[1]), conversion_pairs)
        )

    success_count = sum(results)
    fail_count = len(results) - success_count

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert JP2 images to JPEG format.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing JP2 files to convert",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for JPEG files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input_dir.is_dir():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    success, failed = convert_directory(args.input_dir, args.output_dir)
    logger.info(f"Conversion complete: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
