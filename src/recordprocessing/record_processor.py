from pathlib import Path

from src.recordprocessing.data import ConfigParameter

from src.recordprocessing.pipeline import IOManager, RecordManager
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
        model_file_path: str = "model/best.pt"

        # Labels to exclude from OCR output (add more labels here as needed)
        ocr_excluded_labels = {"Table", "Picture", "Figure", "Form", "Handwriting", "Formula"}

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr, ocr_excluded_labels)

        self.record_manager = RecordManager(self.config, model_file_path)

    def run(self, input_path: Path, output_path: Path):

        image_paths = IOManager.collect_image_files(input_path)

        records = self.record_manager.build_records(image_paths)

        for record in records:

            # Generate folder path arguments
            folder_name = IOManager.generate_folder_name(record)
            record_folder = output_path / folder_name

            # Save record to output folder
            IOManager.save_record_images(record, record_folder)

            # Parse collection of text from all these pages
            ocr_data = self.record_manager.run_ocr(record)

            # Write record to the JSON file
            IOManager.save_record_to_json(record, record_folder, ocr_data)

            # Export to PDF
            IOManager.generate_pdf_from_record(record_folder, ocr_data)

            # Write record to the CSV file
            IOManager.update_records_csv(record, record_folder, output_path)

        GPUController.clear_gpu_memory()
        logger.info("Pipeline finished, GPU memory cleared.")