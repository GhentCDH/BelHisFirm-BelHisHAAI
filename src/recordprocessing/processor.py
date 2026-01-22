import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings

INPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - jpeg")
OUTPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - jpeg/processed")

# Regex pattern for valid section headers: starts with one or more digits followed by a dot
SECTION_HEADER_PATTERN = re.compile(r"^\d+\.—")


def apply_otsu_threshold(image: Image.Image) -> Image.Image:
    """Apply Otsu's automatic thresholding to a grayscale PIL Image."""
    img_array = np.array(image)
    _, thresholded = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresholded)


def is_valid_section_header(text: str) -> bool:
    """Check if text starts with numbers followed by a dot and a dash (e.g., '1.-', '123.-')."""
    return bool(SECTION_HEADER_PATTERN.match(text.strip().replace(" ", "").replace("\n", "")))


def display_valid_section_headers(
    image: Image.Image,
    section_headers: list[dict],
    output_path: str | None = None,
) -> Image.Image:
    """
    Display only valid section headers on the image.

    Args:
        image: PIL Image to annotate
        section_headers: List of dicts with 'bbox', 'text', and 'valid' keys
        output_path: Optional path to save the annotated image

    Returns:
        Annotated PIL Image
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for header in section_headers:
        if not header["valid"]:
            continue

        bbox = header["bbox"]
        text = header["text"]

        # Draw bounding box
        draw.rectangle(bbox, outline="darkred", width=3)

        # Draw label
        label_y = bbox[1] - 20
        if label_y < 0:
            label_y = bbox[1] + 5

        label_text = f"SectionHeader: {text[:30]}..." if len(text) > 30 else f"SectionHeader: {text}"
        text_bbox = draw.textbbox((bbox[0], label_y), label_text, font=font)
        draw.rectangle(text_bbox, fill="darkred")
        draw.text((bbox[0], label_y), label_text, fill="white", font=font)

    if output_path:
        annotated.save(output_path)

    return annotated


def process_images():
    """Process images to detect and validate section headers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize predictors
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor(FoundationPredictor())

    # Collect all image files
    files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + list(INPUT_DIR.glob("*.tif"))
    print(f"Found {len(files)} images")

    for file_path in files:
        print(f"Processing: {file_path.name}")

        # Load and preprocess image
        image = Image.open(file_path).convert("L")
        processed_image = apply_otsu_threshold(image)

        # Get layout predictions
        layout_predictions = layout_predictor([processed_image])
        predictions = layout_predictions[0].bboxes

        # Filter for SectionHeaders only
        section_header_bboxes = [
            pred for pred in predictions if pred.label == "SectionHeader"
        ]

        if not section_header_bboxes:
            print(f"  No SectionHeaders found in {file_path.name}")
            continue

        # OCR each section header region
        section_headers = []
        for pred in section_header_bboxes:
            bbox = [int(c) for c in pred.bbox]

            # Crop the section header region
            cropped = processed_image.crop(bbox)

            # Run OCR on the cropped region
            ocr_results = recognition_predictor([cropped], det_predictor=detection_predictor)

            # Extract text from OCR results
            text = ""
            if ocr_results and ocr_results[0].text_lines:
                text = " ".join(line.text for line in ocr_results[0].text_lines)

            # Validate the section header
            valid = is_valid_section_header(text)

            section_headers.append({
                "bbox": bbox,
                "text": text,
                "valid": valid,
            })

            status = "VALID" if valid else "INVALID"
            print(f"  SectionHeader: '{text[:50]}...' -> {status}" if len(text) > 50 else f"  SectionHeader: '{text}' -> {status}")

        # Count valid headers
        valid_count = sum(1 for h in section_headers if h["valid"])
        if valid_count == 0:
            print(f"  No valid SectionHeaders in {file_path.name}")
            continue

        # Save annotated image with only valid section headers
        output_path = OUTPUT_DIR / file_path.with_suffix(".jpg").name
        display_valid_section_headers(image, section_headers, str(output_path))
        print(f"  Saved: {output_path.name} ({valid_count} valid headers)")

    print("Done")


if __name__ == "__main__":
    process_images()
