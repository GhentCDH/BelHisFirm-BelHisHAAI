import os
import re
import json

from pathlib import Path
from PIL import Image

from recordprocessing.OCR import OCRProcessor
from recordprocessing.data import ConfigParameter
from recordprocessing.pipeline import VisionAnalyzer, ImageProcessor
from src.recordprocessing.data import Record

from src.recordprocessing.pipeline.io_manager import IOManager
from src.recordprocessing.utils.gpu_controller import GPUController

from logging import getLogger
logger = getLogger(__name__)

class RecordManager:


    def __init__(self, config: ConfigParameter, yolo_model_file_path: str):

        self.config = config
        self.vision_analyzer = VisionAnalyzer(config, yolo_model_file_path)
        self.image_processor = ImageProcessor(config)
        self.ocr_processor = None

        if not config.skip_ocr:
            self.ocr_processor = OCRProcessor()


    def build_records(self, image_paths: list[Path]) -> list[Record]:
        """ Builds a list of records from a list of paths, the records are split by headers within the images.

            Args: image_paths (list[Path]): List of paths to images.
            Returns: List of newly built records.
        """
        logger.info("Building records...")

        records = []

        if not image_paths:
            logger.warning("No images provided to build records.")
            return records

        current_record = None
        record_id = 0
        record_start_page_idx = -1  # Tracks which page the current record started on

        for idx, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Failed to open image {image_path.name}: {e}")
                continue

            if idx == 0:
                current_record = self.create_new_record(image, record_id, "TITLE_PAGES")
                record_start_page_idx = idx
                record_id += 1
                continue

            headers_on_page = self.vision_analyzer.detect_record_headers(image)

            if not headers_on_page:
                if current_record:
                    current_record.images.append(image)
                continue

            for header in headers_on_page:
                bbox = header["bbox"]
                header_meta = self.image_processor.which_half_is_bbox_on(bbox, image)

                if current_record:
                    if idx != record_start_page_idx:
                        ending_image = self.image_processor.mask_image(image, bbox, header_meta, "below")
                        current_record.images.append(ending_image)
                    else:
                        current_record.images[-1] = self.image_processor.mask_image(
                            current_record.images[-1], bbox, header_meta, "below"
                        )

                    current_record.end_header_bbox = bbox
                    current_record.end_header_bbox_meta = header_meta
                    current_record.end_header_bbox_page = idx
                    records.append(current_record)

                # Parse title and internal number
                text = header["text"]
                parts = re.split(r'[-–—−]+', text, maxsplit=1)
                internal_number = parts[0].strip() if len(parts) > 0 else ""
                title = parts[1].strip() if len(parts) > 1 else text.strip()

                # New record starts with the above-masked image
                masked_start = self.image_processor.mask_image(image, bbox, header_meta, "above")
                current_record = self.create_new_record(masked_start, record_id, title, internal_number)
                current_record.start_header_bbox = bbox
                current_record.start_header_bbox_meta = header_meta
                current_record.start_header_bbox_page = idx

                record_start_page_idx = idx
                record_id += 1

        if current_record:
            records.append(current_record)

        GPUController.clear_gpu_memory()
        logger.info(f"Finished building records. (Count: {len(records)})")
        return records

    def run_ocr(self, record: Record) -> list:
        """ Runs OCR via QWEN on all pages within a record.

            Args: record (Record): The record to run OCR on.

            Returns: List with parse strings for every page.
        """

        logger.info(f"Running OCR on record. (ID = {record.record_id}, Title = {record.record_title})")

        ocr_data = []

        # Run OCR on all images in the record and collect string results
        for idx, image in enumerate(record.images):
            # Run OCR on this page
            logger.info(f"Running OCR on record page. (Page {idx + 1}/{len(record.images)})")

            # Get excluded regions (tables, figures, etc.) from layout detection
            excluded_regions = self.vision_analyzer.get_excluded_regions(image)

            if excluded_regions:
                logger.info(f"Excluding {len(excluded_regions)} regions from OCR: {[r['label'] for r in excluded_regions]}")

            page_ocr = self.ocr_processor.process_pil_image(image, excluded_regions=excluded_regions)
            ocr_data.append(page_ocr)

            # Clear GPU memory after each page to prevent OOM
            GPUController.clear_gpu_memory()

        logger.info(f"Finished OCR on record. (ID = {record.record_id}, Title = {record.record_title})")

        return ocr_data

    @staticmethod
    def create_new_record(image: Image.Image, record_id: int, record_title: str = "", internal_record_number: str = "") -> Record:
        """ Creates a new Record object with initial metadata and images.

            Args: image (Image.Image): The first image of the record.
            Args: record_id (int): Unique identifier for the record.
            Args: record_title (str): Title of the record.
            Args: internal_record_number (str): Internal numbering for the record.

            Returns: New record object.
        """

        return Record(
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