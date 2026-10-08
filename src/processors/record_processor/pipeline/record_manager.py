import re

from collections.abc import Iterator
from pathlib import Path
from PIL import Image

from src.processors.record_processor import OCRProcessor
from src.processors.record_processor.data import ConfigParameter, TableConfig
from src.processors.record_processor.pipeline import VisionAnalyzer, ImageProcessor
from src.processors.record_processor.pipeline.tables import TablePipeline
from src.processors.record_processor.data import Record

from src.processors.record_processor.utils import GPUController, GLMOCREngine

from logging import getLogger
logger = getLogger(__name__)

class RecordManager:


    def __init__(self, config: ConfigParameter, yolo_model_file_path: str, table_config: TableConfig):

        self.config = config
        self.table_config = table_config

        # Shared across header detection, line OCR, and table transcription -
        # loaded once here rather than once per consumer. A fine-tuned checkpoint replaces the base model for all three.
        self.glm_ocr = GLMOCREngine(table_config.glm_checkpoint) if table_config.glm_checkpoint else GLMOCREngine()

        self.vision_analyzer = VisionAnalyzer(config, yolo_model_file_path, self.glm_ocr)
        self.image_processor = ImageProcessor(config)
        self.ocr_processor = None

        if not config.skip_ocr:
            self.ocr_processor = OCRProcessor(self.glm_ocr)

        self.table_pipeline = None
        if not table_config.skip_tables:
            self.table_pipeline = TablePipeline(table_config, self.glm_ocr)


    def build_records(self, image_paths: list[Path]) -> list[Record]:
        """ Builds a list of records from a list of paths, the records are split by headers within the images.
            Keeps every page image in memory, use iter_records for more than a handful of pages.

            Args: image_paths (list[Path]): List of paths to images.
            Returns: List of newly built records.
        """
        return list(self.iter_records(image_paths))

    def iter_records(self, image_paths: list[Path]) -> Iterator[Record]:
        """ Splits the images into records by the headers found on them, and yields each record as soon
            as it is complete. A page image is 50+ MB in memory, so the caller should save a record and
            let go of it before asking for the next one.

            Args: image_paths (list[Path]): List of paths to images.
            Returns: Iterator over the newly built records.
        """
        logger.info("Building records...")

        record_count = 0

        if not image_paths:
            logger.warning("No images provided to build records.")
            return

        current_record = None
        record_id = 0
        record_start_page_idx = -1  # Tracks which page the current record started on
        last_page_idx = 0

        for idx, image_path in enumerate(image_paths):
            image = self.open_page(image_path)
            if image is None:
                continue

            last_page_idx = idx

            # Detected once per page, record pages take over the regions that are left after masking
            regions = self.vision_analyzer.detect_layout(image)

            if idx == 0:
                current_record = self.create_new_record(image, record_id, "TITLE_PAGES")
                current_record.layout.append(regions)
                record_start_page_idx = idx
                record_id += 1
                continue

            # Layout and header reading stay together per page, a page is too large to keep around
            # or to open a second time for a separate reading pass
            candidates = self.vision_analyzer.find_header_candidates(image, regions)
            headers_on_page = self.vision_analyzer.transcribe_record_headers(image, candidates)

            if not headers_on_page:
                if current_record:
                    current_record.images.append(image)
                    current_record.layout.append(regions)
                continue

            # Detection order is arbitrary, records have to be split in reading order:
            # down the left column (or the full width), then down the right column
            headers_in_reading_order = sorted(
                ((header, self.image_processor.which_half_is_bbox_on(header["bbox"], image)) for header in headers_on_page),
                key=lambda item: (item[1]["side"] == "RIGHT", item[0]["bbox"][1]),
            )

            for header, header_meta in headers_in_reading_order:
                bbox = header["bbox"]

                if current_record:
                    if idx != record_start_page_idx:
                        ending_image = self.image_processor.mask_image(image, bbox, header_meta, "below")
                        current_record.images.append(ending_image)
                        current_record.layout.append(
                            self.image_processor.mask_regions(regions, image.size, bbox, header_meta, "below")
                        )
                    else:
                        current_record.images[-1] = self.image_processor.mask_image(
                            current_record.images[-1], bbox, header_meta, "below"
                        )
                        current_record.layout[-1] = self.image_processor.mask_regions(
                            current_record.layout[-1], image.size, bbox, header_meta, "below"
                        )

                    current_record.end_header_bbox = bbox
                    current_record.end_header_bbox_meta = header_meta
                    current_record.end_header_bbox_page = idx
                    record_count += 1
                    yield current_record

                # Parse title and internal number, GLM-OCR puts line breaks and "---" rules in the text
                lines = [line.strip() for line in header["text"].splitlines()]
                text = " ".join(line for line in lines if line.strip("-—–− "))
                parts = re.split(r'[-–—−]+', text, maxsplit=1)
                internal_number = parts[0].strip() if len(parts) > 0 else ""
                title = parts[1].strip() if len(parts) > 1 else text.strip()

                # New record starts with the above-masked image
                masked_start = self.image_processor.mask_image(image, bbox, header_meta, "above")
                current_record = self.create_new_record(masked_start, record_id, title, internal_number)
                current_record.layout.append(self.image_processor.mask_regions(regions, image.size, bbox, header_meta, "above"))
                current_record.start_header_bbox = bbox
                current_record.start_header_bbox_meta = header_meta
                current_record.start_header_bbox_page = idx

                record_start_page_idx = idx
                record_id += 1

        if current_record:
            # No header ends the last record, it runs until the last page
            current_record.end_header_bbox_page = last_page_idx
            record_count += 1
            yield current_record

        GPUController.clear_gpu_memory()
        logger.info(f"Finished building records. (Count: {record_count})")

    def detect_layout(self, record: Record) -> list[list[dict]]:
        """ Runs layout detection on every page of a record.

            Args: record (Record): The record to detect the layout on.

            Returns: List with the layout regions ('bbox', 'label', 'confidence') for every page.
        """

        logger.info(f"Detecting layout on record. (ID = {record.record_id}, Pages = {len(record.images)})")

        layout = []
        for image in record.images:
            layout.append(self.vision_analyzer.detect_layout(image))

        return layout

    def run_ocr(self, record: Record, layout: list[list[dict]]) -> list:
        """ Transcribes every text block on every page of a record with GLM-OCR.

            Args: record (Record): The record to run OCR on.
            Args: layout (list[list[dict]]): Layout regions for every page, from detect_layout.

            Returns: List with the OCR result (blocks and spine position) for every page.
        """

        logger.info(f"Running OCR on record. (ID = {record.record_id}, Title = {record.record_title})")

        ocr_data = []

        for idx, image in enumerate(record.images):
            logger.info(f"Running OCR on record page. (Page {idx + 1}/{len(record.images)})")

            regions = layout[idx]
            text_regions = [r for r in regions if r["label"] not in self.config.ocr_excluded_labels]
            logger.info(f"OCR on {len(text_regions)} text blocks, skipping {len(regions) - len(text_regions)} non-text regions")

            ocr_data.append(self.ocr_processor.process_text_blocks(image, text_regions))

            GPUController.clear_gpu_memory()

        logger.info(f"Finished OCR on record. (ID = {record.record_id}, Title = {record.record_title})")

        return ocr_data

    @staticmethod
    def open_page(image_path: Path) -> Image.Image | None:
        """ Opens a page image as RGB.

            Args: image_path (Path): Path to the page image.

            Returns: The image, or None when it could not be opened.
        """

        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image {image_path}, the page is left out: {type(e).__name__}: {e}", exc_info=True)
            return None

        return image

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