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


def is_sus_table(pred, image_width: int, image_height: int, area_threshold: float = 0.4, confidence_threshold: float = 0.85) -> bool:

    sus_asci = """
        ⠀⠀⠀⠀⢀⣴⣶⠿⠟⠻⠿⢷⣦⣄⠀⠀⠀
        ⠀⠀⠀⠀⣾⠏⠀⠀⣠⣤⣤⣤⣬⣿⣷⣄⡀
        ⠀⢀⣀⣸⡿⠀⠀⣼⡟⠁⠀⠀⠀⠀⠀⠙⣷
        ⢸⡟⠉⣽⡇⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⢀⣿
        ⣾⠇⠀⣿⡇⠀⠀⠘⠿⢶⣶⣤⣤⣶⡶⣿⠋
        ⣿⠂⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠃ sus
        ⣿⡆⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀
        ⢿⡇⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀
        ⠘⠻⠷⢿⡇⠀⠀⠀⣴⣶⣶⠶⠖⠀⢸⡟⠀
        ⠀⠀⠀⢸⣇⠀⠀⠀⣿⡇⣿⡄⠀⢀⣿⠇⠀
        ⠀⠀⠀⠘⣿⣤⣤⣴⡿⠃⠙⠛⠛⠛⠋⠀⠀

            """



    """
    Check if a Table detection is suspicious (likely a false positive).

    Returns True if:
    - Label is "Table"
    - Confidence is below threshold
    - Bbox covers more than area_threshold of the image
    """
    if pred.label != "Table":
        return False

    if pred.confidence >= confidence_threshold:
        return False

    # Calculate bbox area as fraction of image
    bbox = pred.bbox
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_area = bbox_width * bbox_height
    image_area = image_width * image_height
    area_fraction = bbox_area / image_area

    print(sus_asci)

    return area_fraction > area_threshold


def find_spine_position(image_array: np.ndarray, search_margin: int = 50) -> int | None:
    """
    Find the spine/split position in an image by looking for a dark vertical line in the middle.

    Args:
        image_array: Grayscale numpy array of the image
        search_margin: Number of pixels around the center to search

    Returns:
        X coordinate of the spine position, or None if no clear spine found
    """
    if len(image_array.shape) != 2:
        gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    else:
        gray = image_array

    h, w = gray.shape
    half_w = w // 2

    # Extract a vertical strip around the center
    left_bound = max(0, half_w - search_margin)
    right_bound = min(w, half_w + search_margin)
    center_strip = gray[:, left_bound:right_bound]

    # Threshold to find dark pixels (spine is usually dark)
    thresholded = cv.threshold(center_strip, 180, 255, cv.THRESH_BINARY_INV)[1]

    # Sum vertically to find the column with most dark pixels
    vertical_sum = np.sum(thresholded, axis=0)

    # Check if there's a significant dark line (spine)
    max_darkness = np.max(vertical_sum)
    mean_darkness = np.mean(vertical_sum)

    # Only split if there's a clear dark line (max is significantly above mean)
    if max_darkness > mean_darkness * 1.5:
        spine_offset = np.argmax(vertical_sum)
        return left_bound + spine_offset

    return None


def redetect_region(image: Image.Image, bbox: list, layout_predictor) -> list:
    """
    Re-run layout detection on a cropped region and map coordinates back to original image.
    If a spine (dark vertical line) is detected in the middle, splits the region first.

    Args:
        image: Original PIL Image
        bbox: Bounding box [x1, y1, x2, y2] of the region to re-detect
        layout_predictor: The layout predictor instance

    Returns:
        List of predictions with coordinates mapped back to original image
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Crop the region
    cropped = image.crop((x1, y1, x2, y2))
    cropped_array = np.array(cropped)

    # Check if there's a spine to split on
    spine_pos = find_spine_position(cropped_array)

    if spine_pos is not None:
        print(f"  Spine detected at x={spine_pos}, splitting region into two halves")
        # Split into left and right halves
        h, w = cropped_array.shape[:2]
        left_half = cropped.crop((0, 0, spine_pos, h))
        right_half = cropped.crop((spine_pos, 0, w, h))

        # Process both halves
        regions = [
            (left_half, 0),           # left half, no x offset within crop
            (right_half, spine_pos),  # right half, offset by spine position
        ]
    else:
        # No spine found, process as single region
        regions = [(cropped, 0)]

    # Run layout detection and map coordinates back
    mapped_predictions = []

    # Define MappedPrediction class once outside the loop
    class MappedPrediction:
        pass

    for region_image, region_x_offset in regions:
        predictions = layout_predictor([region_image])[0].bboxes

        for pred in predictions:
            # Offset the bbox coordinates (region offset + original crop offset)
            new_bbox = [
                pred.bbox[0] + x1 + region_x_offset,
                pred.bbox[1] + y1,
                pred.bbox[2] + x1 + region_x_offset,
                pred.bbox[3] + y1,
            ]
            # Offset the polygon coordinates
            new_polygon = [[p[0] + x1 + region_x_offset, p[1] + y1] for p in pred.polygon]

            mapped = MappedPrediction()
            mapped.bbox = new_bbox
            mapped.polygon = new_polygon
            mapped.confidence = pred.confidence
            mapped.label = pred.label
            mapped.position = pred.position
            mapped.top_k = pred.top_k if hasattr(pred, "top_k") else {}

            mapped_predictions.append(mapped)

    return mapped_predictions


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
    """Process images to detect and validate section headers.

    Args:
        debug: If True, skip OCR and show all bounding boxes instead.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize predictors
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
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
        predictions = list(layout_predictions[0].bboxes)

        # Check for suspicious Table detections and re-detect those regions
        image_width, image_height = image.size
        final_predictions = []
        for pred in predictions:
            if is_sus_table(pred, image_width, image_height):
                print(f"  Suspicious Table detected (conf={pred.confidence:.2f}), re-running detection on region...")
                new_preds = redetect_region(image, pred.bbox, layout_predictor)
                print(f"  Re-detection found {len(new_preds)} boxes")
                final_predictions.extend(new_preds)
            else:
                final_predictions.append(pred)

        predictions = final_predictions

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
    process_images()