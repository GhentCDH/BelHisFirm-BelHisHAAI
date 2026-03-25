import os
import json

from pathlib import Path
from PIL import Image

from recordprocessing.OCR import OCRProcessor
from recordprocessing.data import ConfigParameter
from recordprocessing.pipeline import VisionAnalyzer
from src.recordprocessing.data import Record

from src.recordprocessing.pipeline.io_manager import IOManager
from src.recordprocessing.pipeline.pdf_exporter import PDFExporter
from src.recordprocessing.utils.gpu_controller import GPUController

from logging import getLogger
logger = getLogger(__name__)

class RecordManager:


    def __init__(self, config: ConfigParameter, yolo_model_file_path: str):

        self.config = config
        self.vision_analyzer = VisionAnalyzer(config, yolo_model_file_path)
        self.ocr_processor = None

        if not config.skip_ocr:
            self.ocr_processor = OCRProcessor()


    @staticmethod
    def build_records(images: list[Image.Image]) -> list[Record]:

        records = []

        # Start records at title page
        current_record = RecordManager.create_new_record(images[0], 0, "TITLE_PAGES")

        record_id = 0
        record_page_idx = 0

        for image in images:
            pass


        return records

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

    def generate_record(self, record: Record, output_folder: Path) -> None:
        """ Saves record images, runs OCR if enabled, and exports results as JSON and PDF.

            Returns: None
        """

        os.makedirs(output_folder, exist_ok=True)

        folder_name = IOManager.generate_folder_name(record)
        record_folder = output_folder / folder_name

        # Clear existing page files to prevent stale files from previous runs being included
        if record_folder.exists():
            for old_file in record_folder.glob("page_*.jpg"):
                old_file.unlink()
        os.makedirs(record_folder, exist_ok=True)

        # Run OCR on all images and collect results
        ocr_data = []
        skip_ocr_for_record = self.config.skip_ocr or record.record_title == "TITLE_PAGES"

        for idx, image in enumerate(record.images):
            image_filename = record_folder / f"page_{idx + 1:03d}.jpg"
            image.save(image_filename)

            if skip_ocr_for_record:
                if self.config.skip_ocr:
                    logger.info(f"Skipping OCR (--no-ocr): {image_filename.name}")
                else:
                    logger.info(f"Skipping OCR for TITLE_PAGES: {image_filename.name}")
                ocr_data.append({"lines": [], "spine_position": None})
            else:
                # Run OCR on this page
                logger.info(
                    f"Running OCR on page {idx + 1}/{len(record.images)}: {image_filename.name} (size: {image.size})")

                # Get excluded regions (tables, figures, etc.) from layout detection
                excluded_regions = self.vision_analyzer.get_excluded_regions(image)
                if excluded_regions:
                    logger.info(
                        f"Excluding {len(excluded_regions)} regions from OCR: {[r['label'] for r in excluded_regions]}")

                page_ocr = self.ocr_processor.process_pil_image(
                    image,
                    excluded_regions=excluded_regions,
                    debug_name=str(image_filename),
                    save_debug_image=False
                )
                ocr_data.append(page_ocr)

                # Clear GPU memory after each page to prevent OOM
                GPUController.clear_gpu_memory()

        # Save OCR data as JSON
        ocr_json_path = record_folder / "ocr_data.json"
        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "record_id": record.record_id,
                "record_title": record.record_title,
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
        IOManager.update_records_csv(record, record_folder, output_folder)