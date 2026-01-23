from importlib.resources import files
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List
import os

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings

from logging import getLogger
logger = getLogger(__name__)

@dataclass
class Record:
    record_id : str | int
    record_title : str
    images : List[Image.Image]
    start_header_bbox : list[float]
    start_header_bbox_meta : dict
    start_header_bbox_page : int
    end_header_bbox : list[float]
    end_header_bbox_meta : dict
    end_header_bbox_page : int
   
class RecordProcessor:
    
    def __init__(self):
        self.image_files = []
        self.headers = []

        self.padding = 15  # Pixels of padding around the crop for better OCR
        self.sus_table_confidence_threshold = 0.85 
        self.sus_table_area_threshold = 0.4

        self.record = None

        self.layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
        self.detection_predictor = DetectionPredictor()
        self.recognition_predictor = RecognitionPredictor(FoundationPredictor())

    def create_new_record(self, image, record_id: int, record_title: str = ""):
        self.record = Record(
            images=[image],
            record_id=record_id,
            record_title=record_title,
            start_header_bbox=[],
            start_header_bbox_meta={},
            start_header_bbox_page=0,
            end_header_bbox=[],
            end_header_bbox_meta={},
            end_header_bbox_page=0,
        )

    def collect_image_files(self, folder_path):
        logger.info(f"Collecting image files from {folder_path}...")
        image_files = sorted(list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.jpeg")) + list(Path(folder_path).glob("*.tif")) + list(Path(folder_path).glob("*.jp2")))
        return image_files 
    
    # This function detectes if a Table prediction spans more than half of a page, which could mean that it overrides headers

    def is_sus_table(self, pred, image_width: int, image_height: int, confidence_threshold: float = 0.85) -> bool:
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

        if area_fraction <= self.sus_table_area_threshold:
            logging.info(f"Table detection check passed!")
            return False
        else:
            logging.info(f"Sus ඞ Table detected: conf={pred.confidence:.2f}, area_fraction={area_fraction:.2f}")
            return True
    
    def is_valid_section_header(self, text: str):
        cleaned = text.strip().replace("\n", "")
        SECTION_HEADER_PATTERN = re.compile(r"^\d+\.\s*[—–-]")
        return bool(SECTION_HEADER_PATTERN.match(cleaned)) and "," in text
    
    def find_spine_position(self, image_array: np.ndarray, search_margin: int = 50) -> int | None:
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

    def redetect_region(self, image: Image.Image, bbox: list, original_prediction) -> list:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped)
        
        spine_pos = self.find_spine_position(cropped_array)
        
        if spine_pos is None:
            # No spine found - keep original prediction
            logger.warning("No spine detected, keeping original prediction!")
            return [original_prediction]
        
        # Spine found - split and redetect
        logger.info(f"  Spine detected at x={spine_pos}, splitting region into two halves")
        h, w = cropped_array.shape[:2]
        left_half = cropped.crop((0, 0, spine_pos, h))
        right_half = cropped.crop((spine_pos, 0, w, h))
        
        # Batch layout detection for split regions
        batch_predictions = self.layout_predictor([left_half, right_half])
        
        # Map coordinates back
        mapped_predictions = []
        
        for predictions, region_x_offset in zip(batch_predictions, [0, spine_pos]):
            for pred in predictions.bboxes:
                mapped = type('MappedPrediction', (), {})()
                mapped.bbox = [
                    pred.bbox[0] + x1 + region_x_offset,
                    pred.bbox[1] + y1,
                    pred.bbox[2] + x1 + region_x_offset,
                    pred.bbox[3] + y1,
                ]
                mapped.polygon = [[p[0] + x1 + region_x_offset, p[1] + y1] for p in pred.polygon]
                mapped.confidence = pred.confidence
                mapped.label = pred.label
                mapped.position = pred.position
                mapped.top_k = pred.top_k if hasattr(pred, "top_k") else {}
                mapped_predictions.append(mapped)
        
        return mapped_predictions

    def is_record_header_candidate(self, pred) -> bool:
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
            logger.info(f"Overridden: {first_label} with confidence {first_conf}")
            return True

        return False

    def detect_record_headers(self, image):

        # Initialize predictors

        layout_predictions = self.layout_predictor([image])
        predictions = list(layout_predictions[0].bboxes)
        if not predictions:
            logger.info(f"No layout predictions..")
            return
        else:
            image_width, image_height = image.size

            verified_predictions = [] # validated predictions with re-detections 

            for pred in predictions:
                if self.is_sus_table(pred, image_width, image_height):
                    new_preds = self.redetect_region(image, pred.bbox, pred)
                    verified_predictions.extend(new_preds)
                else:
                    verified_predictions.append(pred)
            
            record_header_predictions = [pred for pred in verified_predictions if self.is_record_header_candidate(pred)]
            if not record_header_predictions:
                logger.info(f"No record headers found in layout predictions...")
                return None
            
            headers_on_page = []
            for pred in record_header_predictions:
                bbox = [int(c) for c in pred.bbox]
                # Crop the section header region with padding
                padded_bbox = [
                    max(0, bbox[0] - self.padding),
                    max(0, bbox[1] - self.padding),
                    min(image.width, bbox[2] + self.padding),
                    min(image.height, bbox[3] + self.padding),
                ]
                cropped = image.crop(padded_bbox)

                # Run OCR on the cropped region
                ocr_results = self.recognition_predictor([cropped], det_predictor=self.detection_predictor)

                # Extract text from OCR results
                text = ""
                if ocr_results and ocr_results[0].text_lines:
                    text = " ".join(line.text for line in ocr_results[0].text_lines)

                # Validate the section header
                valid = self.is_valid_section_header(text)

                status = "VALID" if valid else "INVALID"
                logger.info(f'{status} recordheader found at {text[:50]}...' if len(text) > 50 else f"{status} recordheader found at {text}")

                if valid:
                    headers_on_page.append({
                        "bbox": bbox,
                        "text": text})

            return headers_on_page if headers_on_page else None
    

    def which_half_is_bbox_on(self, bbox: list, image):
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        image_array = np.array(image)
        halfline = self.find_spine_position(image_array=image_array)

        if halfline is None:
            logger.warning("No spine detected, cannot determine bbox side")
            return {"side": "UNKNOWN", "halfline": None}

        # Check if bbox spans across the halfline
        if x1 < halfline < x2:
            meta = {"side": "MIDDLE", "halfline": halfline}
            return meta
        elif bbox_center_x < halfline:
            meta = {"side": "LEFT", "halfline": halfline}
            return meta
        else:
            meta = {"side": "RIGHT", "halfline": halfline}
            return meta
    

    def generate_record(self):
        output_folder = self.output_folder

        os.makedirs(output_folder, exist_ok=True)
        folder_name = f"record_{int(self.record.record_id):03d}_{self.record.record_title.replace(' ', '_')}"
        record_folder = output_folder / folder_name
        os.makedirs(record_folder, exist_ok=True)
        for idx, image in enumerate(self.record.images):
            image_filename = record_folder / f"page_{idx+1:03d}.jpg"
            image.save(image_filename)


    def process_record(self, record_path: Path, output_folder: Path):
        self.output_folder = output_folder
        
        from pathlib import Path
        folder_path = Path(record_path)
        images = self.collect_image_files(folder_path)

        id = 0
        for idx, image_path in enumerate(images):
            print(f"Processing image: {image_path.name}")
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image {image_path.name}: {e}")
                return
            if idx == 0:
                self.create_new_record(image=image, record_id=id, record_title="TITLE_PAGES")
                id += 1
            else:
                headers_on_page = self.detect_record_headers(image)

                if headers_on_page:
                    for header in headers_on_page:

                        self.record.end_header_bbox = header["bbox"]
                        self.record.end_header_bbox_meta = self.which_half_is_bbox_on(header["bbox"], image)
                        self.record.images.append(image)

                        self.generate_record()
                        
                        text = header["text"]
                        parts = re.split(r'[-–—−]+', text, maxsplit=1)
                        title = parts[1].strip() if len(parts) > 1 else text.strip()
                        self.create_new_record(image=image, record_id=id, record_title=title)
                        self.record.start_header_bbox = header["bbox"]
                        self.record.start_header_bbox_meta = self.which_half_is_bbox_on(header["bbox"], image)

                        id += 1

                else:
                    logger.info(f"No record headers detected on page {image_path.name}...")
                    self.record.images.append(image)

if __name__ == "__main__":
    processor = RecordProcessor()
    processor.process_record(Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing"), Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing - Sort"))