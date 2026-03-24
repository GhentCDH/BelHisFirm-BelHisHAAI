import json
import re
import os

from PIL import Image
from pathlib import Path

from recordprocessing.pipeline import VisionAnalyzer
from src.recordprocessing.data import ConfigParameter
from src.recordprocessing.data import Record

from src.recordprocessing.pipeline import IOManager, PDFExporter, ImageProcessor
from src.recordprocessing.utils import GPUController

try:
    from .OCR import OCRProcessor
except ImportError:
    from OCR import OCRProcessor

from logging import getLogger
logger = getLogger(__name__)

class RecordProcessor:

    def __init__(self, skip_ocr: bool = False):

        # File path to YOLO model
        model_file_path: str = "/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/model/best.pt"

        # Labels to exclude from OCR output (add more labels here as needed)
        ocr_excluded_labels = {"Table", "Picture", "Figure", "Form", "Handwriting", "Formula"}

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr, ocr_excluded_labels)

        self.record = None

        # Clear cache after model initialization
        GPUController.clear_gpu_memory()

        self.image_processor = ImageProcessor(self.config)
        self.vision_analyzer = VisionAnalyzer(self.config, model_file_path)

        self.ocr_processor = None

        if not skip_ocr:
            self.ocr_processor = OCRProcessor()


    def create_new_record(self, image: Image.Image, record_id: int, record_title: str = "", internal_record_number: str = "") -> None:
        """Creates a new Record object with initial metadata and images.

            Args: image (Image.Image): The first image of the record.
            Args: record_id (int): Unique identifier for the record.
            Args: record_title (str): Title of the record.
            Args: internal_record_number (str): Internal numbering for the record.

            Returns: None
        """

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

    def generate_record(self) -> None:
        """ Saves record images, runs OCR if enabled, and exports results as JSON and PDF.

            Returns: None
        """

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
        skip_ocr_for_record = self.config.skip_ocr or self.record.record_title == "TITLE_PAGES"

        for idx, image in enumerate(self.record.images):
            image_filename = record_folder / f"page_{idx+1:03d}.jpg"
            image.save(image_filename)

            if skip_ocr_for_record:
                if self.config.skip_ocr:
                    logger.info(f"Skipping OCR (--no-ocr): {image_filename.name}")
                else:
                    logger.info(f"Skipping OCR for TITLE_PAGES: {image_filename.name}")
                ocr_data.append({"lines": [], "spine_position": None})
            else:
                # Run OCR on this page
                logger.info(f"Running OCR on page {idx + 1}/{len(self.record.images)}: {image_filename.name} (size: {image.size})")
                
                # Get excluded regions (tables, figures, etc.) from layout detection
                excluded_regions = self.vision_analyzer.get_excluded_regions(image)
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
                GPUController.clear_gpu_memory()

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
        """ Processes all images in a folder into structured records with headers detection, OCR, and PDF export.

            Args: record_path (Path): Path to the folder containing input images.
            Args: output_folder (Path): Path to the folder where output files should be saved.

            Returns: None
        """

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
                headers_on_page = self.vision_analyzer.detect_record_headers(image)

                if headers_on_page:
                    for header in headers_on_page:
                        header_meta = self.image_processor.which_half_is_bbox_on(header["bbox"], image)

                        self.record.end_header_bbox = header["bbox"]
                        self.record.end_header_bbox_meta = header_meta
                        self.record.end_header_bbox_page = idx
                        if idx != record_page_idx:
                            masked_end = self.image_processor.mask_image(image, header["bbox"], header_meta, "below")
                            self.record.images.append(masked_end)
                        else:
                            # Same page as record start — apply "below" mask on top of existing "above" mask
                            self.record.images[-1] = self.image_processor.mask_image(self.record.images[-1], header["bbox"], header_meta, "below")

                        self.generate_record()

                        text = header["text"]
                        logger.info(f"Record header: {text}")
                        parts = re.split(r'[-–—−]+', text, maxsplit=1)
                        internal_number = parts[0].strip() if len(parts) > 0 else ""
                        title = parts[1].strip() if len(parts) > 1 else text.strip()
                        masked_start = self.image_processor.mask_image(image, header["bbox"], header_meta, "above")
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
        GPUController.clear_gpu_memory()
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