from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw
from surya.detection import DetectionPredictor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class OCRProcessor:
    def __init__(self):
        self.detection_predictor = DetectionPredictor()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", # 8B
            dtype="auto",
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct") # 8B

        # Spine detection parameters
        self.spine_vertical_margin = 200
        self.spine_margin = 300

    def find_spine_position(self, image: Image.Image) -> int | None:
        """Find the vertical spine/split position in a two-column image."""
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape
        half_w = w // 2

        top = min(self.spine_vertical_margin, h // 2)
        bottom = max(h - self.spine_vertical_margin, h // 2)
        cropped = gray[top:bottom, :]

        # Extract a vertical strip around the center
        left_bound = max(0, half_w - self.spine_margin)
        right_bound = min(w, half_w + self.spine_margin)
        center_strip = cropped[:, left_bound:right_bound]

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

    def is_bbox_in_excluded_region(self, text_bbox: list, excluded_regions: list, overlap_threshold: float = 0.5) -> bool:
        """Check if a text bbox falls within an excluded region.
        
        Args:
            text_bbox: [x1, y1, x2, y2] of the text line
            excluded_regions: List of dicts with 'bbox' key
            overlap_threshold: Minimum overlap ratio to exclude the text
        
        Returns:
            True if the text bbox should be excluded
        """
        tx1, ty1, tx2, ty2 = text_bbox
        text_area = (tx2 - tx1) * (ty2 - ty1)
        
        if text_area == 0:
            return False
        
        for region in excluded_regions:
            rx1, ry1, rx2, ry2 = region["bbox"]
            
            # Calculate intersection
            ix1 = max(tx1, rx1)
            iy1 = max(ty1, ry1)
            ix2 = min(tx2, rx2)
            iy2 = min(ty2, ry2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection_area / text_area
                
                if overlap_ratio >= overlap_threshold:
                    return True
        
        return False

    def filter_excluded_bboxes(self, bboxes: list, excluded_regions: list) -> list:
        """Filter out text bboxes that fall within excluded regions."""
        if not excluded_regions:
            return bboxes
        
        filtered = []
        excluded_count = 0
        
        for bbox in bboxes:
            bbox_coords = bbox.bbox if hasattr(bbox, 'bbox') else bbox
            if not self.is_bbox_in_excluded_region(bbox_coords, excluded_regions):
                filtered.append(bbox)
            else:
                excluded_count += 1
        
        if excluded_count > 0:
            print(f"[DEBUG] Filtered out {excluded_count} text lines in excluded regions")
        
        return filtered

    def extend_bboxes_to_column_edges(self, bboxes: list, image: Image.Image) -> list:
        """Extend bounding boxes to the edge of their respective columns."""
        if not bboxes:
            return bboxes

        spine_pos = self.find_spine_position(image)
        img_width = image.width

        if spine_pos is None:
            # No spine detected - single column, extend all to image width
            max_x2 = max(bbox.bbox[2] for bbox in bboxes)
            extended = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.bbox
                extended.append(type('ExtendedBBox', (), {
                    'bbox': [x1, y1, max_x2, y2],
                    'confidence': getattr(bbox, 'confidence', 1.0),
                    'column': 'single'
                })())
            return extended

        # Split bboxes into left, right, and spanning columns based on physical position
        left_bboxes = []
        right_bboxes = []
        spanning_bboxes = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.bbox
            # Check if bbox physically spans both columns (crosses the spine)
            if x1 < spine_pos and x2 > spine_pos:
                spanning_bboxes.append(bbox)
            elif x2 <= spine_pos:
                # Entirely on left side
                left_bboxes.append(bbox)
            else:
                # Entirely on right side (x1 >= spine_pos)
                right_bboxes.append(bbox)

        # Find max x2 for left column (but cap at spine)
        left_max_x2 = spine_pos - 10  # Small margin from spine
        if left_bboxes:
            left_max_x2 = min(max(bbox.bbox[2] for bbox in left_bboxes), spine_pos - 10)

        # Find max x2 for right column (extend to image edge)
        right_max_x2 = img_width
        if right_bboxes:
            right_max_x2 = max(bbox.bbox[2] for bbox in right_bboxes)

        # Extend all bboxes and tag with column assignment
        extended = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.bbox

            # Check if bbox spans both columns - don't extend these (e.g., titles)
            if x1 < spine_pos and x2 > spine_pos:
                new_x2 = x2  # Keep original
                column = 'spanning'
            elif x2 <= spine_pos:
                # Left column
                new_x2 = left_max_x2
                column = 'left'
            else:
                # Right column
                new_x2 = right_max_x2
                column = 'right'

            extended.append(type('ExtendedBBox', (), {
                'bbox': [x1, y1, new_x2, y2],
                'confidence': getattr(bbox, 'confidence', 1.0),
                'column': column
            })())

        print(f"[DEBUG] Spine at {spine_pos}, left: {len(left_bboxes)} (max_x2={left_max_x2}), right: {len(right_bboxes)} (max_x2={right_max_x2}), spanning: {len(spanning_bboxes)}")
        return extended

    def calculate_vertical_overlap(self, bbox1, bbox2) -> float:
        """Calculate vertical overlap ratio between two bboxes.

        Returns the ratio of vertical intersection to the smaller bbox's height.
        This helps detect when two bboxes are on the same text line.
        """
        _, y1_1, _, y2_1 = bbox1.bbox
        _, y1_2, _, y2_2 = bbox2.bbox

        # Calculate vertical intersection
        y_intersect_start = max(y1_1, y1_2)
        y_intersect_end = min(y2_1, y2_2)

        if y_intersect_end <= y_intersect_start:
            return 0.0

        vertical_intersection = y_intersect_end - y_intersect_start

        # Use the smaller height as reference
        height1 = y2_1 - y1_1
        height2 = y2_2 - y1_2
        min_height = min(height1, height2)

        if min_height == 0:
            return 0.0

        return vertical_intersection / min_height

    def calculate_horizontal_overlap(self, bbox1, bbox2) -> float:
        """Calculate horizontal overlap ratio between two bboxes.

        Returns the ratio of horizontal intersection to the smaller bbox's width.
        """
        x1_1, _, x2_1, _ = bbox1.bbox
        x1_2, _, x2_2, _ = bbox2.bbox

        # Calculate horizontal intersection
        x_intersect_start = max(x1_1, x1_2)
        x_intersect_end = min(x2_1, x2_2)

        if x_intersect_end <= x_intersect_start:
            return 0.0

        horizontal_intersection = x_intersect_end - x_intersect_start

        # Use the smaller width as reference
        width1 = x2_1 - x1_1
        width2 = x2_2 - x1_2
        min_width = min(width1, width2)

        if min_width == 0:
            return 0.0

        return horizontal_intersection / min_width

    def are_same_line(self, bbox1, bbox2, vertical_threshold: float = 0.7, horizontal_threshold: float = 0.3) -> bool:
        """Check if two bboxes are detecting the same text line.

        Two bboxes are considered the same line if:
        1. They have significant vertical overlap (same row)
        2. They have some horizontal overlap (overlapping text)
        """
        vertical_overlap = self.calculate_vertical_overlap(bbox1, bbox2)
        horizontal_overlap = self.calculate_horizontal_overlap(bbox1, bbox2)

        return vertical_overlap >= vertical_threshold and horizontal_overlap >= horizontal_threshold

    def remove_overlapping_bboxes(self, bboxes: list, vertical_threshold: float = 0.7, horizontal_threshold: float = 0.3) -> list:
        """Remove bounding boxes that detect the same text line.

        Uses vertical and horizontal overlap to determine if two bboxes
        are detecting the same line, which works better than IoU for text
        lines that may have different widths after column extension.
        """
        if not bboxes:
            return bboxes

        # Sort by area (largest first) and confidence (higher first)
        # Keeping larger bboxes ensures we don't lose text
        sorted_bboxes = sorted(
            bboxes,
            key=lambda b: (
                (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]),
                getattr(b, 'confidence', 0.0)
            ),
            reverse=True
        )

        kept = []
        for bbox in sorted_bboxes:
            should_keep = True
            for kept_bbox in kept:
                if self.are_same_line(bbox, kept_bbox, vertical_threshold, horizontal_threshold):
                    should_keep = False
                    print(f"[DEBUG] Removing duplicate line bbox (v_overlap={self.calculate_vertical_overlap(bbox, kept_bbox):.2f}, h_overlap={self.calculate_horizontal_overlap(bbox, kept_bbox):.2f})")
                    break
            if should_keep:
                kept.append(bbox)

        removed_count = len(bboxes) - len(kept)
        if removed_count > 0:
            print(f"[DEBUG] Removed {removed_count} duplicate line bboxes")

        return kept

    def detect_lines(self, image: Image.Image) -> list:
        """Detect text lines in an image."""
        predictions = self.detection_predictor([image])
        return predictions[0].bboxes

    def ocr_cropped_line(self, cropped_image: Image.Image) -> str:
        """Run OCR on a cropped text line using Qwen."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": "Transcribe the text. Ignore multiple . in a row."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""

    def process_image(self, image_path: Path, padding: int = 5) -> dict:
        """Detect lines and OCR each one from a file path. Returns dict with 'lines' and 'spine_position'."""
        image = Image.open(image_path)
        return self.process_pil_image(image, padding)

    def process_pil_image(self, image: Image.Image, padding: int = 5, debug_name: str = None, save_debug_image: bool = False, excluded_regions: list = None) -> dict:
        """Detect lines and OCR each one from a PIL Image. Returns dict with 'lines' and 'spine_position'.
        
        Args:
            image: PIL Image to process
            padding: Padding to add around text lines when cropping
            debug_name: Optional name for debug output files
            save_debug_image: Whether to save debug images
            excluded_regions: List of regions to exclude from OCR (e.g., tables, figures).
                              Each region is a dict with 'bbox' key containing [x1, y1, x2, y2].
        """
        # Work with a copy to avoid any reference issues
        image = image.copy()
        if image.mode != "RGB":
            image = image.convert("RGB")

        if debug_name:
            print(f"[DEBUG OCR] Processing: {debug_name} (size: {image.size})")
            if save_debug_image:
                debug_path = debug_name.replace(".jpg", "_ocr_input.jpg")
                image.save(debug_path)
                print(f"[DEBUG OCR] Saved OCR input image to: {debug_path}")

        bboxes = self.detect_lines(image)
        print(f"Detected {len(bboxes)} text lines")

        # Filter out text lines in excluded regions (tables, figures, etc.)
        if excluded_regions:
            bboxes = self.filter_excluded_bboxes(bboxes, excluded_regions)
            print(f"After filtering excluded regions: {len(bboxes)} text lines")

        # Detect spine position and save it for later use
        spine_position = self.find_spine_position(image)

        # Extend bboxes to column edges for complete line capture
        bboxes = self.extend_bboxes_to_column_edges(bboxes, image)

        # Remove fully overlapping bboxes
        bboxes = self.remove_overlapping_bboxes(bboxes)

        if debug_name and save_debug_image:
            # Save image with bounding boxes drawn
            bbox_image = image.copy()
            draw = ImageDraw.Draw(bbox_image)
            # Draw excluded regions in blue
            if excluded_regions:
                for region in excluded_regions:
                    rx1, ry1, rx2, ry2 = region["bbox"]
                    draw.rectangle([rx1, ry1, rx2, ry2], outline="blue", width=4)
                    label = region.get("label", "excluded")
                    draw.text((rx1, ry1 - 20), f"SKIP: {label}", fill="blue")
            # Draw text bboxes in red
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox.bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 15), str(i + 1), fill="red")
            bbox_debug_path = debug_name.replace(".jpg", "_ocr_bboxes.jpg")
            bbox_image.save(bbox_debug_path)
            print(f"[DEBUG OCR] Saved bounding boxes image to: {bbox_debug_path}")

        lines = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.bbox
            column = getattr(bbox, 'column', 'unknown')

            # Save original bbox coordinates (before padding)
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1), int(y1), int(x2), int(y2)

            # Add padding for cropping context (but don't save these coordinates)
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image.width, x2 + padding)
            crop_y2 = min(image.height, y2 + padding)

            cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            text = self.ocr_cropped_line(cropped)

            lines.append({
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                "text": text,
                "column": column,
            })
            print(f"Line {i + 1} ({column}): {text}")

        return {
            "lines": lines,
            "spine_position": int(spine_position) if spine_position is not None else None
        }

    def visualize_detected_lines(self, image_path: Path, output_path: Path = None):
        """Detect text lines and visualize them with bounding boxes."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        bboxes = self.detect_lines(image)

        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if output_path:
            image.save(output_path)
            print(f"Saved visualization to {output_path}")
        else:
            image.show()

        print(f"Detected {len(bboxes)} text lines")
        return image


if __name__ == "__main__":
    sample_image = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/EHC_B665_O_2025_1909_III_0015.jp2")

    ocr = OCRProcessor()

    # Visualize detected lines
    lines_output = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/detected_lines_visualization.jpg")
    ocr.visualize_detected_lines(sample_image, lines_output)

    # Process and OCR all detected lines
    ocr_result = ocr.process_image(sample_image)
    print("\n--- OCR Results ---")
    print(f"Spine position: {ocr_result['spine_position']}")
    for i, line in enumerate(ocr_result['lines']):
        print(f"{i + 1}. {line['text']}")
    with open("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/ocr_results.txt", "w") as f:
        for line in ocr_result['lines']:
            f.write(f"{line['text']}\n")
