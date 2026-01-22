import re
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings

INPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/test")
OUTPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/test/processed")

# Regex pattern for valid section headers: starts with one or more digits followed by a dot
SECTION_HEADER_PATTERN = re.compile(r"^\d+\.—")


def apply_otsu_threshold(image: Image.Image) -> Image.Image:
    """Apply Otsu's automatic thresholding to a grayscale PIL Image."""
    img_array = np.array(image)
    _, thresholded = cv.threshold(img_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return Image.fromarray(thresholded)


def is_valid_section_header(text: str) -> bool:
    """Check if text starts with numbers followed by a dot and a dash (e.g., '1.-', '123.-')."""
    return bool(SECTION_HEADER_PATTERN.match(text.strip().replace(" ", "").replace("\n", "")))


def is_section_header_candidate(pred) -> bool:
    """
    Check if a prediction should be treated as a SectionHeader.

    Returns True if:
    - The label is already SectionHeader, OR
    - The top label confidence is < 90% AND SectionHeader is the second choice in top_k
    """
    if pred.label == "SectionHeader":
        return True

    if not hasattr(pred, "top_k") or not pred.top_k:
        return False

    # Get sorted labels by confidence
    sorted_labels = sorted(pred.top_k.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_labels) < 2:
        return False

    first_label, first_conf = sorted_labels[0]
    second_label, _ = sorted_labels[1]

    if first_conf < 0.90 and second_label == "SectionHeader":
        print("Overridden:", first_label, first_conf)
        return True

    return False


def display_all_bboxes(
    image: Image.Image,
    bboxes: list,
    output_path: str | None = None,
) -> Image.Image:
    """
    Display all bounding boxes on the image (debug mode).

    Args:
        image: PIL Image to annotate
        bboxes: List of bbox prediction objects with 'bbox' and 'label' attributes
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

    # Color mapping for different labels
    colors = {
        "SectionHeader": "darkred",
        "Text": "blue",
        "Picture": "green",
        "Table": "orange",
        "Caption": "purple",
        "Footnote": "brown",
        "Formula": "cyan",
        "PageHeader": "magenta",
        "PageFooter": "gray",
    }

    for pred in bboxes:
        bbox = [int(c) for c in pred.bbox]
        label = pred.label

        # Check if this is a SectionHeader candidate (overridden from another label)
        is_candidate = is_section_header_candidate(pred)
        if is_candidate and label != "SectionHeader":
            label = f"{label}→SectionHeader"
            color = "darkred"
        else:
            color = colors.get(label, "black")

        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=2)

        # Draw label
        label_y = bbox[1] - 20
        if label_y < 0:
            label_y = bbox[1] + 5

        text_bbox = draw.textbbox((bbox[0], label_y), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], label_y), label, fill="white", font=font)

    if output_path:
        annotated.save(output_path)

    return annotated


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


def process_images(debug: bool = False):
    """Process images to detect and validate section headers.

    Args:
        debug: If True, skip OCR and show all bounding boxes instead.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize predictors
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    if not debug:
        detection_predictor = DetectionPredictor()
        recognition_predictor = RecognitionPredictor(FoundationPredictor())

    # Collect all image files
    files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.jp2"))
    print(f"Found {len(files)} images")

    for file_path in files:
        print(f"Processing: {file_path.name}")

        # Load and preprocess image
        image = Image.open(file_path)

        # Get layout predictions
        layout_predictions = layout_predictor([image])
        predictions = layout_predictions[0].bboxes
        print(predictions)

        # Debug mode: show all bounding boxes and skip OCR
        if debug:
            print(f"  Found {len(predictions)} bounding boxes")
            output_path = OUTPUT_DIR / f"{file_path.stem}.jpg"
            display_all_bboxes(image, predictions, str(output_path))
            print(f"  Saved: {output_path.name}")
            continue

        # Filter for SectionHeaders (including candidates based on top_k)
        section_header_bboxes = [
            pred for pred in predictions if is_section_header_candidate(pred)
        ]

        if not section_header_bboxes:
            print(f"  No SectionHeaders found in {file_path.name}")
            continue

        # OCR each section header region
        section_headers = []
        for pred in section_header_bboxes:
            bbox = [int(c) for c in pred.bbox]
            print(pred)

            # Crop the section header region
            cropped = image.crop(bbox)

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
        output_path = OUTPUT_DIR / f"{file_path.stem}.jpg"
        display_valid_section_headers(image, section_headers, str(output_path))
        print(f"  Saved: {output_path.name} ({valid_count} valid headers)")

    print("Done")


if __name__ == "__main__":
    DEBUG = True # Set to True to skip OCR and show all bounding boxes
    process_images(debug=DEBUG)