"""TIFF and JPEG2000 image validation with error logging."""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

NUM_THREADS = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Magic bytes for file format detection
TIFF_LE_MAGIC = b"\x49\x49\x2A\x00"  # Little-endian TIFF
TIFF_BE_MAGIC = b"\x4D\x4D\x00\x2A"  # Big-endian TIFF
JP2_MAGIC = b"\x00\x00\x00\x0C\x6A\x50\x20\x20"  # JPEG2000 JP2 container
J2K_MAGIC = b"\xFF\x4F\xFF\x51"  # JPEG2000 codestream

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jp2", ".j2k", ".jpf", ".jpx"}


class ImageType(Enum):
    TIFF = "tiff"
    JPEG2000 = "jpeg2000"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    path: Path
    valid: bool
    image_type: ImageType
    error: str | None = None


def detect_image_type(data: bytes) -> ImageType:
    """Detect image type from magic bytes."""
    if data[:4] in (TIFF_LE_MAGIC, TIFF_BE_MAGIC):
        return ImageType.TIFF
    if data[:8] == JP2_MAGIC or data[:4] == J2K_MAGIC:
        return ImageType.JPEG2000
    return ImageType.UNKNOWN


def validate_image(path: Path) -> ValidationResult:
    """Full validation by attempting to decode the image."""
    try:
        with open(path, "rb") as f:
            data = f.read(1024)

        if len(data) == 0:
            return ValidationResult(path, False, ImageType.UNKNOWN, "Empty file")

        image_type = detect_image_type(data)

        if image_type == ImageType.UNKNOWN:
            return ValidationResult(path, False, ImageType.UNKNOWN, "Unknown image format")

        from PIL import Image
        with Image.open(path) as img:
            img.load()

        return ValidationResult(path, True, image_type)

    except PermissionError:
        return ValidationResult(path, False, ImageType.UNKNOWN, "Permission denied")
    except Exception as e:
        return ValidationResult(path, False, ImageType.UNKNOWN, str(e))


def validate_directory(directory: Path) -> list[ValidationResult]:
    """Validate all TIFF and JPEG2000 files in a directory."""
    directory = Path(directory).resolve()
    files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning(f"No supported image files found in {directory}")
        return []

    logger.info(f"Validating {len(files)} file(s)...")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(validate_image, files))

    return results


def write_error_log(results: list[ValidationResult], output_path: Path) -> int:
    """Write invalid files to log. Returns count of invalid files."""
    invalid = [r for r in results if not r.valid]

    with open(output_path, "w") as f:
        f.write(f"# Validation errors - {datetime.now().isoformat()}\n")
        f.write(f"# Total files checked: {len(results)}\n")
        f.write(f"# Invalid files: {len(invalid)}\n\n")

        for result in invalid:
            f.write(f"{result.path}\t{result.error}\n")

    return len(invalid)


def main():
    parser = argparse.ArgumentParser(
        description="Validate TIFF and JPEG2000 images.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to validate",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        logger.error(f"Directory does not exist: {args.input_dir}")
        return 1

    results = validate_directory(args.input_dir)

    if not results:
        return 0

    log_path = args.input_dir / "validation_errors.log"
    invalid_count = write_error_log(results, log_path)
    valid_count = len(results) - invalid_count

    logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
    logger.info(f"Error log written to: {log_path}")

    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
