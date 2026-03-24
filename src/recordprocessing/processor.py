import json
import re
import gc
from pathlib import Path
import os

import torch
import pytesseract as tesseract

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO

from recordprocessing.data import ConfigParameter
from recordprocessing.pipeline import IOManager, PDFExporter, ImageProcessor

from src.recordprocessing.data import MappedPrediction
from src.recordprocessing.data import Record

from src.utils import ComponentProcessor

try:
    from .OCR import OCRProcessor
except ImportError:
    from OCR import OCRProcessor

from logging import getLogger
logger = getLogger(__name__)

class RecordProcessor:

    def __init__(self, skip_ocr: bool = False):

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr)

        self.padding = 50
        self.sus_table_confidence_threshold = 0.85
        self.sus_table_area_threshold = 0.4
        self.spine_vertical_margin = 200
        self.spine_margin = 300
        self.skip_ocr = skip_ocr

        self.record = None
        
        # Labels to exclude from OCR output (add more labels here as needed)
        self.ocr_excluded_labels = {"Table", "Picture", "Figure", "Form", "Handwriting", "Formula"}

        """ TEMPORARY HARDCODE """
        self.yolo_model = YOLO("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/model/best.pt")
        
        # Clear cache after model initialization
        self._clear_gpu_memory()
        
        if not skip_ocr:
            self.ocr_processor = OCRProcessor()
        else:
            self.ocr_processor = None

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory to prevent OOM errors."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear metal performance shader cache on Mac devices
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def create_new_record(self, image: Image.Image, record_id: int, record_title: str = "", internal_record_number: str = "") -> None:
        self.record = Record(
            images=[image],
            record_id=record_id,
            record_title=record_title,
            internal_record_number=internal_record_number,
            start_header_bbox=[],
            start_header_bbox_meta={},
            start_header_bbox_page=0,
            end_header_bbox=[],
            end_header_bbox_meta={},
            end_header_bbox_page=0,
        )


    # This function detects if a Table prediction spans more than half of a page, which could mean that it overrides headers
    def is_sus_table(self, prediction: MappedPrediction, image_width: int, image_height: int) -> bool:
        if prediction.label != "Table":
            return False

        if prediction.confidence >= self.sus_table_confidence_threshold:
            return False

        bbox = prediction.bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        area_fraction = bbox_area / image_area

        if area_fraction <= self.sus_table_area_threshold:
            logger.info(f"Table detection check passed!")
            return False
        else:
            logger.info(f"Sus table detected: conf={prediction.confidence:.2f}, area_fraction={area_fraction:.2f}")
            return True
    
    def is_valid_section_header(self, text: str) -> bool:
        cleaned = text.strip().replace("\n", "")
        section_header_pattern = re.compile(r"^\d+\.\s*[—–-]")
        return bool(section_header_pattern.match(cleaned)) and "," in text


    def redetect_region(self, image: Image.Image, bbox: list, original_prediction: MappedPrediction) -> list:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped)
        
        spine_pos = ImageProcessor.find_spine_position(self.config, cropped_array)
        
        if spine_pos is None:
            # No spine found - keep original prediction
            logger.warning("No spine detected, keeping original prediction!")
            return [original_prediction]
        
        # Spine found - split and redetect
        logger.info(f"  Spine detected at x={spine_pos}, splitting region into two halves")
        h, w = cropped_array.shape[:2]
        left_half = cropped.crop((0, 0, spine_pos, h))
        right_half = cropped.crop((spine_pos, 0, w, h))

        images = [left_half, right_half]
        offsets = [0, spine_pos]

        mapped_predictions = []

        for image, region_x_offset in zip(images, offsets):
            results = self.yolo_model.predict(image, stream=True)

            for result in results:
                mapped_prediction = ComponentProcessor.process_result(result, x1 + region_x_offset, y1)

                mapped_predictions.append(mapped_prediction)
        
        return mapped_predictions

    def is_record_header_candidate(self, prediction: MappedPrediction) -> bool:
        if prediction.label == "title":
            return True
        else:
            return False

    def detect_record_headers(self, image: Image.Image) -> list | None:

        results = self.yolo_model.predict(image, stream=True)

        if not results:
            logger.info(f"No layout predictions..")
            return None

        image_width, image_height = image.size
        verified_predictions = []

        for result in results:
            mapped_predictions = ComponentProcessor.process_result(result)

            for mapped_prediction in mapped_predictions:
                if self.is_sus_table(mapped_prediction, image_width, image_height):
                    new_predictions = self.redetect_region(image, mapped_prediction.bbox, mapped_prediction)
                    verified_predictions.extend(new_predictions)
                else:
                    verified_predictions.append(mapped_prediction)

        record_header_predictions = [prediction for prediction in verified_predictions if self.is_record_header_candidate(prediction)]
        if not record_header_predictions:
            logger.info(f"No record headers found in layout predictions...")
            return None

        headers_on_page = []
        for prediction in record_header_predictions:
            bbox = [int(c) for c in prediction.bbox]
            padded_bbox = (
                max(0, bbox[0] - self.padding),
                max(0, bbox[1] - self.padding),
                min(image.width, bbox[2] + self.padding),
                min(image.height, bbox[3] + self.padding),
            )
            cropped = image.crop(padded_bbox)


            # Convert to grayscale
            cropped = cropped.convert("L")

            # Increase contrast
            enhancer = ImageEnhance.Contrast(cropped)
            cropped = enhancer.enhance(2)

            # Sharpen the image
            cropped = cropped.filter(ImageFilter.SHARPEN)

            text = tesseract.image_to_string(cropped, config="--psm 6", lang="fra")
            print(f"\n||{text}||\n")

            valid = self.is_valid_section_header(text)

            status = "VALID" if valid else "INVALID"
            logger.info(f'{status} recordheader found at {text[:50]}...' if len(text) > 50 else f"{status} recordheader found at {text}")

            if valid:
                headers_on_page.append({
                    "bbox": bbox,
                    "text": text})

        # Clear GPU memory after processing headers
        self._clear_gpu_memory()
        
        return headers_on_page if headers_on_page else None
    

    def get_excluded_regions(self, image: Image.Image) -> list:
        """Get regions to exclude from OCR based on layout detection.

        Returns list of dicts with 'bbox' and 'label' keys.
        Configure excluded labels via self.ocr_excluded_labels.
        """

        excluded_regions = []

        results = self.yolo_model.predict(image, stream=True)
        for result in results:
            mapped_predictions = ComponentProcessor.process_result(result)

            for mapped_prediction in mapped_predictions:
                if mapped_prediction.label in self.ocr_excluded_labels:
                    excluded_regions.append({
                        "bbox": [int(c) for c in mapped_prediction.bbox],
                        "label": mapped_prediction.label,
                        "confidence": mapped_prediction.confidence
                    })
                    logger.debug(f"Excluding region: {mapped_prediction.label} at {mapped_prediction.bbox}")

        return excluded_regions

    def generate_record(self) -> None:
        output_folder = self.output_folder

        os.makedirs(output_folder, exist_ok=True)

        folder_name = IOManager.generate_folder_name(self.record)
        record_folder = output_folder / folder_name
        
        # Clear existing page files to prevent stale files from previous runs being included
        if record_folder.exists():
            for old_file in record_folder.glob("page_*.jpg"):
                old_file.unlink()
        os.makedirs(record_folder, exist_ok=True)

        # Run OCR on all images and collect results
        ocr_data = []
        skip_ocr_for_record = self.skip_ocr or self.record.record_title == "TITLE_PAGES"

        for idx, image in enumerate(self.record.images):
            image_filename = record_folder / f"page_{idx+1:03d}.jpg"
            image.save(image_filename)

            if skip_ocr_for_record:
                if self.skip_ocr:
                    logger.info(f"Skipping OCR (--no-ocr): {image_filename.name}")
                else:
                    logger.info(f"Skipping OCR for TITLE_PAGES: {image_filename.name}")
                ocr_data.append({"lines": [], "spine_position": None})
            else:
                # Run OCR on this page
                logger.info(f"Running OCR on page {idx + 1}/{len(self.record.images)}: {image_filename.name} (size: {image.size})")
                
                # Get excluded regions (tables, figures, etc.) from layout detection
                excluded_regions = self.get_excluded_regions(image)
                if excluded_regions:
                    logger.info(f"Excluding {len(excluded_regions)} regions from OCR: {[r['label'] for r in excluded_regions]}")
                
                page_ocr = self.ocr_processor.process_pil_image(
                    image, 
                    excluded_regions=excluded_regions,
                    debug_name=str(image_filename), 
                    save_debug_image=False
                )
                ocr_data.append(page_ocr)
                
                # Clear GPU memory after each page to prevent OOM
                self._clear_gpu_memory()

        # Save OCR data as JSON
        ocr_json_path = record_folder / "ocr_data.json"
        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "record_id": self.record.record_id,
                "record_title": self.record.record_title,
                "pages": ocr_data
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR data saved: {ocr_json_path}")

        # Generate PDF from saved images (with or without OCR text layer)
        if skip_ocr_for_record:
            logger.info("Generating PDF without OCR text layer...")
        else:
            logger.info("Generating searchable PDF with OCR text layer...")

        PDFExporter.generate_pdf_from_images(record_folder, ocr_data)

        # Update CSV index
        IOManager.update_records_csv(self.record, record_folder, output_folder)

    def process_record(self, record_path: Path, output_folder: Path) -> None:
        self.output_folder = output_folder
        images = IOManager.collect_image_files(record_path)

        id = 0
        record_page_idx = -1
        for idx, image_path in enumerate(images):
            logger.info(f"Processing image: {image_path.name}")
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image {image_path.name}: {e}")
                continue
            if idx == 0:
                self.create_new_record(image=image, record_id=id, record_title="TITLE_PAGES", internal_record_number="")
                record_page_idx = idx
                id += 1
            else:
                headers_on_page = self.detect_record_headers(image)

                if headers_on_page:
                    for header in headers_on_page:
                        header_meta = ImageProcessor.which_half_is_bbox_on(self.config, header["bbox"], image)

                        self.record.end_header_bbox = header["bbox"]
                        self.record.end_header_bbox_meta = header_meta
                        self.record.end_header_bbox_page = idx
                        if idx != record_page_idx:
                            masked_end = ImageProcessor.mask_image(image, header["bbox"], header_meta, "below")
                            self.record.images.append(masked_end)
                        else:
                            # Same page as record start — apply "below" mask on top of existing "above" mask
                            self.record.images[-1] = ImageProcessor.mask_image(self.record.images[-1], header["bbox"], header_meta, "below")

                        self.generate_record()

                        text = header["text"]
                        logger.info(f"Record header: {text}")
                        parts = re.split(r'[-–—−]+', text, maxsplit=1)
                        internal_number = parts[0].strip() if len(parts) > 0 else ""
                        title = parts[1].strip() if len(parts) > 1 else text.strip()
                        masked_start = ImageProcessor.mask_image(image, header["bbox"], header_meta, "above")
                        self.create_new_record(image=masked_start, record_id=id, record_title=title, internal_record_number=internal_number)
                        self.record.start_header_bbox = header["bbox"]
                        self.record.start_header_bbox_meta = header_meta
                        self.record.start_header_bbox_page = idx

                        record_page_idx = idx
                        id += 1

                else:
                    logger.info(f"No record headers detected on page {image_path.name}...")
                    if self.record:
                        self.record.images.append(image)

        if self.record:
            self.generate_record()
        
        # Final cleanup
        self._clear_gpu_memory()
        logger.info("Processing complete, GPU memory cleared.")

if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process historical records with OCR")
    parser.add_argument("input_folder", type=Path, nargs="?",
                        default=Path("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/images"),
                        help="Path to folder containing input images")
    parser.add_argument("output_folder", type=Path, nargs="?",
                        default=Path("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/output"),
                        help="Path to output folder for processed records")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR processing (only extract and save images)")
    args = parser.parse_args()

    processor = RecordProcessor(skip_ocr=args.no_ocr)
    processor.process_record(args.input_folder, args.output_folder)