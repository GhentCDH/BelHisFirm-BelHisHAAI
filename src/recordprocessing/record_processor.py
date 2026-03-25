import re

from PIL import Image
from pathlib import Path

from src.recordprocessing.data import ConfigParameter

from src.recordprocessing.pipeline import IOManager, ImageProcessor, VisionAnalyzer
from src.recordprocessing.utils import GPUController
from src.recordprocessing.pipeline.record_manager import RecordManager

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

        self.current_record = None

        # Clear cache after model initialization
        GPUController.clear_gpu_memory()

        self.image_processor = ImageProcessor(self.config)
        self.vision_analyzer = VisionAnalyzer(self.config, model_file_path)
        self.record_manager = RecordManager(self.config, model_file_path)


    def process_record(self, input_path: Path, output_folder: Path) -> None:
        """ Processes all images in a folder into structured records with headers detection, OCR, and PDF export.

            Args: record_path (Path): Path to the folder containing input images.
            Args: output_folder (Path): Path to the folder where output files should be saved.

            Returns: None
        """




        images = IOManager.collect_image_files(input_path)

        id = 0
        record_page_idx = -1

        """ LOOP THROUGH ALL INPUT IMAGES"""
        for idx, image_path in enumerate(images):

            logger.info(f"Processing image: {image_path.name}")

            """ MAKE SURE IMAGE CAN BE OPENED """
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image {image_path.name}: {e}")
                continue

            if idx == 0:
                self.current_record = RecordManager.create_new_record(image=image, record_id=id, record_title="TITLE_PAGES", internal_record_number="")
                record_page_idx = idx
                id += 1
            else:
                headers_on_page = self.vision_analyzer.detect_record_headers(image)

                """ CHECK IF HEADERS ARE ON THE PAGE """
                if not headers_on_page:
                    logger.info(f"No record headers detected on page {image_path.name}...")
                    if self.current_record:
                        self.current_record.images.append(image)
                else:
                    for header in headers_on_page:
                        header_meta = self.image_processor.which_half_is_bbox_on(header["bbox"], image)

                        self.current_record.end_header_bbox = header["bbox"]
                        self.current_record.end_header_bbox_meta = header_meta
                        self.current_record.end_header_bbox_page = idx

                        if idx != record_page_idx:
                            masked_end = self.image_processor.mask_image(image, header["bbox"], header_meta, "below")
                            self.current_record.images.append(masked_end)
                        else:
                            # Same page as record start — apply "below" mask on top of existing "above" mask
                            self.current_record.images[-1] = self.image_processor.mask_image(self.current_record.images[-1], header["bbox"], header_meta, "below")

                        self.record_manager.generate_record(self.current_record, output_folder)

                        text = header["text"]
                        logger.info(f"Record header: {text}")
                        parts = re.split(r'[-–—−]+', text, maxsplit=1)
                        internal_number = parts[0].strip() if len(parts) > 0 else ""
                        title = parts[1].strip() if len(parts) > 1 else text.strip()
                        masked_start = self.image_processor.mask_image(image, header["bbox"], header_meta, "above")
                        self.current_record = RecordManager.create_new_record(image=masked_start, record_id=id, record_title=title, internal_record_number=internal_number)
                        self.current_record.start_header_bbox = header["bbox"]
                        self.current_record.start_header_bbox_meta = header_meta
                        self.current_record.start_header_bbox_page = idx

                        record_page_idx = idx
                        id += 1

        """ GENERATE FINAL RECORD"""
        if self.current_record:
            self.record_manager.generate_record(self.current_record, output_folder)
        
        # Final cleanup
        GPUController.clear_gpu_memory()
        logger.info("Processing complete, GPU memory cleared.")