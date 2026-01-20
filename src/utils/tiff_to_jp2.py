import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

NUM_THREADS = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = {".tif", ".tiff"}


def convert_tiff_to_jp2(input_path: Path, output_path: Path):
    try:
        with Image.open(input_path) as img:
            if hasattr(img, "n_frames") and img.n_frames > 1:
                logger.warning(
                    f"Multi-page TIFF detected ({img.n_frames} pages): {input_path}. "
                    "Converting first page only."
                )

            if img.mode == "P":
                img = img.convert("RGB")
            elif img.mode == "RGBA":
                if img.split()[3].getextrema() == (255, 255):
                    img = img.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, "JPEG2000", quality_mode="lossless")
            logger.info(f"Converted: {input_path} -> {output_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def convert_directory(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    tiff_files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in TIFF_EXTENSIONS and f.is_file()
    ]

    if not tiff_files:
        logger.warning(f"No TIFF files found in {input_dir}")
        return 0, 0

    logger.info(f"Found {len(tiff_files)} TIFF file(s) to convert")

    # Build list of (input, output) pairs
    conversion_pairs = []
    for tiff_path in tiff_files:
        relative_path = tiff_path.relative_to(input_dir)
        jp2_path = output_dir / relative_path.with_suffix(".jp2")
        conversion_pairs.append((tiff_path, jp2_path))

    # Process in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(lambda p: convert_tiff_to_jp2(p[0], p[1]), conversion_pairs))

    success_count = sum(results)
    fail_count = len(results) - success_count

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert TIFF images to lossless JPEG2000 format.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing TIFF files to convert",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for JP2 files",
    )
    parser.add_argument(
        "-v", "--verbose",
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
