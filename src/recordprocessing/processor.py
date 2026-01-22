import json
import re
from dataclasses import dataclass, field, asdict
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
OUTPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/test/result")


@dataclass
class BoundingRegion:
    """Represents a bounding region on a page for a record."""
    page_index: int
    column: str  # "left", "right", or "single"
    y_start: int
    y_end: int
    x_start: int
    x_end: int


@dataclass
class RecordLocation:
    """Represents a record's location across potentially multiple pages."""
    record_number: int
    record_title: str
    regions: list[BoundingRegion] = field(default_factory=list)


# Regex pattern for extracting record number from title
RECORD_NUMBER_PATTERN = re.compile(r"^(\d+)\.\s*[—–-]")

# Regex pattern for valid section headers: starts with one or more digits followed by a dot and dash
# Accepts optional whitespace and various dash types (em dash, en dash, hyphen)
SECTION_HEADER_PATTERN = re.compile(r"^\d+\.\s*[—–-]")


def apply_otsu_threshold(image: Image.Image) -> Image.Image:
    """Apply Otsu's automatic thresholding to a grayscale PIL Image."""
    img_array = np.array(image)
    _, thresholded = cv.threshold(img_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return Image.fromarray(thresholded)


def is_valid_section_header(text: str) -> bool:
    """Check if text starts with numbers followed by a dot and a dash, and contains at least one comma."""
    cleaned = text.strip().replace("\n", "")
    # Must match pattern AND contain at least one comma
    return bool(SECTION_HEADER_PATTERN.match(cleaned)) and "," in text


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

        # Batch process both halves together
        region_images = [left_half, right_half]
        region_x_offsets = [0, spine_pos]
    else:
        # No spine found, process as single region
        region_images = [cropped]
        region_x_offsets = [0]

    # Batch layout detection for all regions
    batch_predictions = layout_predictor(region_images)

    # Map coordinates back
    mapped_predictions = []

    class MappedPrediction:
        pass

    for idx, (predictions, region_x_offset) in enumerate(zip(batch_predictions, region_x_offsets)):
        for pred in predictions.bboxes:
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


def extract_record_number(title_text: str) -> int | None:
    """
    Extract numeric ID from a title string.

    Args:
        title_text: The title text (e.g., "4919. - L'Union...")

    Returns:
        The record number as an integer, or None if not found
    """
    match = RECORD_NUMBER_PATTERN.match(title_text.strip())
    if match:
        return int(match.group(1))
    return None


def assign_column(bbox: list, spine_position: int | None, image_width: int) -> str:
    """
    Determine if a bounding box is in the left, right, or single column.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        spine_position: X coordinate of the spine, or None if no spine
        image_width: Width of the image

    Returns:
        "left", "right", or "single"
    """
    if spine_position is None:
        return "single"

    # Use the center x of the bbox to determine column
    bbox_center_x = (bbox[0] + bbox[2]) / 2

    if bbox_center_x < spine_position:
        return "left"
    else:
        return "right"


def sort_headers_by_reading_order(
    headers: list[dict],
    spine_position: int | None,
    image_width: int
) -> list[dict]:
    """
    Sort headers by reading order: left column (by y), then right column (by y).

    Args:
        headers: List of header dicts with 'bbox' and 'text' keys
        spine_position: X coordinate of the spine, or None if no spine
        image_width: Width of the image

    Returns:
        Sorted list of headers with 'column' key added
    """
    # Add column info to each header
    for header in headers:
        header["column"] = assign_column(header["bbox"], spine_position, image_width)

    if spine_position is None:
        # Single column - sort by y position only
        return sorted(headers, key=lambda h: h["bbox"][1])

    # Separate into left and right columns
    left_headers = [h for h in headers if h["column"] == "left"]
    right_headers = [h for h in headers if h["column"] == "right"]

    # Sort each by y position
    left_headers.sort(key=lambda h: h["bbox"][1])
    right_headers.sort(key=lambda h: h["bbox"][1])

    # Reading order: left column first, then right column
    return left_headers + right_headers


def calculate_record_regions(
    sorted_headers: list[dict],
    page_idx: int,
    spine_position: int | None,
    image_width: int,
    image_height: int
) -> dict[int, BoundingRegion]:
    """
    Calculate bounding region for each record on a page.

    Each record starts at its header's y_start and ends at the next header's y_start
    (or column/page end).

    Args:
        sorted_headers: Headers sorted by reading order
        page_idx: Index of the current page
        spine_position: X coordinate of the spine, or None
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        Dict mapping record_number to its BoundingRegion on this page
    """
    regions = {}

    if not sorted_headers:
        return regions

    for i, header in enumerate(sorted_headers):
        record_num = extract_record_number(header["text"])
        if record_num is None:
            continue

        column = header["column"]
        y_start = int(header["bbox"][1])

        # Determine x boundaries based on column
        if spine_position is None:
            x_start = 0
            x_end = image_width
        elif column == "left":
            x_start = 0
            x_end = spine_position
        else:  # right
            x_start = spine_position
            x_end = image_width

        # Find y_end: next header in same column, or column bottom
        y_end = image_height  # Default to bottom of page/column

        for j in range(i + 1, len(sorted_headers)):
            next_header = sorted_headers[j]
            if next_header["column"] == column:
                # Next header in same column - this record ends there
                y_end = int(next_header["bbox"][1])
                break

        regions[record_num] = BoundingRegion(
            page_index=page_idx,
            column=column,
            y_start=y_start,
            y_end=y_end,
            x_start=x_start,
            x_end=x_end
        )

    return regions


def handle_column_continuation(
    regions: dict[int, BoundingRegion],
    sorted_headers: list[dict],
    spine_position: int | None,
    image_height: int,
    image_width: int
) -> dict[int, list[BoundingRegion]]:
    """
    Extend records that continue from left column to right column.

    If a record is the last one in the left column, it continues into the right
    column until the first header in the right column.

    Args:
        regions: Dict mapping record_number to BoundingRegion
        sorted_headers: Headers sorted by reading order
        spine_position: X coordinate of the spine, or None
        image_height: Height of the image
        image_width: Width of the image

    Returns:
        Dict mapping record_number to list of BoundingRegions (may have multiple for continuation)
    """
    result = {num: [region] for num, region in regions.items()}

    if spine_position is None:
        return result

    # Find the last record in the left column
    left_headers = [h for h in sorted_headers if h["column"] == "left"]
    right_headers = [h for h in sorted_headers if h["column"] == "right"]

    if not left_headers:
        return result

    last_left_header = left_headers[-1]
    last_left_record_num = extract_record_number(last_left_header["text"])

    if last_left_record_num is None:
        return result

    # Check if this record's region ends at the column bottom (image_height)
    if last_left_record_num in result:
        left_region = result[last_left_record_num][0]
        if left_region.y_end >= image_height - 10:  # Near bottom
            # This record continues into the right column
            if right_headers:
                # Continues until the first right column header
                first_right_header = right_headers[0]
                continuation_y_end = int(first_right_header["bbox"][1])
            else:
                # No headers in right column - entire right column is continuation
                continuation_y_end = image_height

            continuation_region = BoundingRegion(
                page_index=left_region.page_index,
                column="right",
                y_start=0,  # Start from top of right column
                y_end=continuation_y_end,
                x_start=spine_position,
                x_end=image_width
            )
            result[last_left_record_num].append(continuation_region)

    return result


def create_masked_image(
    image: Image.Image,
    regions: list[BoundingRegion],
    mask_color: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create an image with irrelevant areas filled with white.

    Args:
        image: Original PIL Image
        regions: List of BoundingRegions to keep visible
        mask_color: Color to fill masked areas (default white)

    Returns:
        New image with only the specified regions visible
    """
    # Create a white image
    masked = Image.new("RGB", image.size, mask_color)

    # Copy only the regions we want to keep
    for region in regions:
        # Crop the region from the original image
        bbox = (region.x_start, region.y_start, region.x_end, region.y_end)
        cropped = image.crop(bbox)

        # Paste onto the masked image
        masked.paste(cropped, (region.x_start, region.y_start))

    return masked


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to a safe filename slug.

    Args:
        text: The text to convert
        max_length: Maximum length of the output

    Returns:
        Safe filename string
    """
    # Remove the record number prefix (e.g., "4919. - ")
    text = re.sub(r"^\d+\.\s*[—–-]\s*", "", text)

    # Replace problematic characters with underscores
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)

    # Truncate and clean up
    text = text[:max_length].strip("_")

    return text


def export_records(
    all_records: dict[int, RecordLocation],
    image_files: list[Path],
    output_dir: Path
) -> None:
    """
    Export each record to its own folder with masked images and metadata.

    Args:
        all_records: Dict mapping record_number to RecordLocation
        image_files: List of image file paths (ordered by page)
        output_dir: Directory to save output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all images once
    images = []
    for file_path in image_files:
        img = Image.open(file_path).convert("RGB")
        images.append(img)

    index_data = []

    for record_num, record_loc in sorted(all_records.items()):
        # Create folder name
        title_slug = sanitize_filename(record_loc.record_title)
        folder_name = f"{record_num:05d}_{title_slug}"
        record_dir = output_dir / folder_name
        record_dir.mkdir(parents=True, exist_ok=True)

        # Group regions by page
        regions_by_page: dict[int, list[BoundingRegion]] = {}
        for region in record_loc.regions:
            if region.page_index not in regions_by_page:
                regions_by_page[region.page_index] = []
            regions_by_page[region.page_index].append(region)

        # Create masked images for each page
        saved_pages = []
        for page_idx in sorted(regions_by_page.keys()):
            if page_idx >= len(images):
                continue

            page_regions = regions_by_page[page_idx]
            masked_image = create_masked_image(images[page_idx], page_regions)

            # Save the masked image
            page_filename = f"page_{page_idx:03d}.jpg"
            masked_image.save(record_dir / page_filename, quality=95)
            saved_pages.append(page_filename)

        print(f"  Exported record {record_num}: {title_slug} ({len(saved_pages)} pages)")
        
    print(f"Exported {len(all_records)} records to {output_dir}")


def extract_records_pipeline(input_dir: Path, output_dir: Path) -> None:
    """
    Main orchestration function for extracting records from historical documents.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save extracted records
    """
    # Initialize predictors
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor(FoundationPredictor())

    # Collect and sort image files
    files = (
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.jp2"))
    )
    files = sorted(files, key=lambda f: f.name)
    print(f"Found {len(files)} images")

    if not files:
        print("No images found")
        return

    # Process each page to collect headers and regions
    all_records: dict[int, RecordLocation] = {}
    last_record_num: int | None = None

    for page_idx, file_path in enumerate(files):
        print(f"Processing page {page_idx}: {file_path.name}")

        # Load image
        image = Image.open(file_path).convert("RGB")
        image_width, image_height = image.size
        image_array = np.array(image)

        # Detect spine position
        spine_position = find_spine_position(image_array)
        if spine_position:
            print(f"  Spine detected at x={spine_position}")

        # Run layout detection
        layout_predictions = layout_predictor([image])
        predictions = list(layout_predictions[0].bboxes)

        # Handle suspicious tables
        final_predictions = []
        for pred in predictions:
            if is_sus_table(pred, image_width, image_height):
                print(f"  Suspicious Table detected (conf={pred.confidence:.2f}), re-running detection...")
                new_preds = redetect_region(image, pred.bbox, layout_predictor)
                print(f"  Re-detection found {len(new_preds)} boxes")
                final_predictions.extend(new_preds)
            else:
                final_predictions.append(pred)
        predictions = final_predictions

        # Filter for section header candidates
        section_header_bboxes = [
            pred for pred in predictions if is_section_header_candidate(pred)
        ]

        # OCR section headers
        valid_headers = []
        padding = 15
        for pred in section_header_bboxes:
            bbox = [int(c) for c in pred.bbox]

            # Crop with padding
            padded_bbox = [
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image_width, bbox[2] + padding),
                min(image_height, bbox[3] + padding),
            ]
            cropped = image.crop(padded_bbox)

            # Run OCR
            ocr_results = recognition_predictor([cropped], det_predictor=detection_predictor)
            text = ""
            if ocr_results and ocr_results[0].text_lines:
                text = " ".join(line.text for line in ocr_results[0].text_lines)

            # Validate section header
            if is_valid_section_header(text):
                valid_headers.append({
                    "bbox": bbox,
                    "text": text.strip().replace("\n", " "),
                    "valid": True
                })
                print(f"  Valid header: {text[:60]}...")

        # Sort headers by reading order
        sorted_headers = sort_headers_by_reading_order(valid_headers, spine_position, image_width)

        # Calculate record regions
        regions = calculate_record_regions(
            sorted_headers, page_idx, spine_position, image_width, image_height
        )

        # Handle column continuation
        expanded_regions = handle_column_continuation(
            regions, sorted_headers, spine_position, image_height, image_width
        )

        # Handle case where no headers on this page - entire page belongs to last record
        if not sorted_headers and last_record_num is not None:
            # Create a region covering the entire page
            if spine_position:
                # Two-column layout - left and right regions
                left_region = BoundingRegion(
                    page_index=page_idx,
                    column="left",
                    y_start=0,
                    y_end=image_height,
                    x_start=0,
                    x_end=spine_position
                )
                right_region = BoundingRegion(
                    page_index=page_idx,
                    column="right",
                    y_start=0,
                    y_end=image_height,
                    x_start=spine_position,
                    x_end=image_width
                )
                all_records[last_record_num].regions.extend([left_region, right_region])
            else:
                # Single column
                full_region = BoundingRegion(
                    page_index=page_idx,
                    column="single",
                    y_start=0,
                    y_end=image_height,
                    x_start=0,
                    x_end=image_width
                )
                all_records[last_record_num].regions.append(full_region)

        # Add regions to all_records
        for record_num, region_list in expanded_regions.items():
            if record_num not in all_records:
                # Find the title for this record
                title = ""
                for h in sorted_headers:
                    if extract_record_number(h["text"]) == record_num:
                        title = h["text"]
                        break

                all_records[record_num] = RecordLocation(
                    record_number=record_num,
                    record_title=title,
                    regions=[]
                )

            all_records[record_num].regions.extend(region_list)
            last_record_num = record_num

        # Handle continuation from last record of previous page into this page
        # if there are headers on this page starting in the right column
        if sorted_headers and last_record_num is not None:
            first_header = sorted_headers[0]
            if first_header["column"] == "right" and spine_position:
                # The left column of this page belongs to the last record
                left_continuation = BoundingRegion(
                    page_index=page_idx,
                    column="left",
                    y_start=0,
                    y_end=image_height,
                    x_start=0,
                    x_end=spine_position
                )
                # Only add if we haven't already added a left region for this record on this page
                existing_pages = {r.page_index for r in all_records[last_record_num].regions if r.column == "left"}
                if page_idx not in existing_pages:
                    all_records[last_record_num].regions.append(left_continuation)

        # Update last_record_num to the last record on this page
        if expanded_regions:
            last_record_num = max(expanded_regions.keys())

    # Export all records
    export_records(all_records, files, output_dir)


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
        padding = 15  # Pixels of padding around the crop for better OCR
        for pred in section_header_bboxes:
            bbox = [int(c) for c in pred.bbox]

            # Crop the section header region with padding
            padded_bbox = [
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.width, bbox[2] + padding),
                min(image.height, bbox[3] + padding),
            ]
            cropped = image.crop(padded_bbox)

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
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        # Run the record extraction pipeline
        input_path = Path(sys.argv[2]) if len(sys.argv) > 2 else INPUT_DIR
        output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else OUTPUT_DIR / "records"
        extract_records_pipeline(input_path, output_path)
    else:
        # Default: run the debug/visualization pipeline
        process_images()