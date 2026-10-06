from pathlib import Path

from src.processors.record_processor.data import ConfigParameter, TableConfig

from src.processors.record_processor.pipeline import IOManager, RecordManager
from src.processors.record_processor.utils import GPUController

try:
    from .OCR import OCRProcessor
except ImportError:
    from OCR import OCRProcessor

from logging import getLogger
logger = getLogger(__name__)

class RecordProcessor:

    def __init__(self, skip_ocr: bool = False, skip_tables: bool = False,
                 vllm_base_url: str = "http://localhost:8000/v1", vllm_model: str = "qwen3.6-35b-nvfp4"):

        # File path to YOLO model
        model_file_path: str = "model/best.pt"

        # Labels to exclude from OCR output (add more labels here as needed)
        ocr_excluded_labels = {"Table", "Picture", "Formula", "Caption"}

        # Labels that qualify a detected region as a record-divider header
        header_candidate_labels = {"Section-header", "Title"}

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr, ocr_excluded_labels, header_candidate_labels)
        self.table_config = TableConfig(skip_tables=skip_tables, vllm_base_url=vllm_base_url, vllm_model=vllm_model)

        self.record_manager = RecordManager(self.config, model_file_path, self.table_config)

    def run(self, input_path: Path, output_path: Path):

        image_paths = IOManager.collect_image_files(input_path)

        records = self.record_manager.build_records(image_paths)

        for record in records:

            # Generate folder path arguments
            folder_name = IOManager.generate_folder_name(record)
            record_folder = output_path / folder_name

            # Save record to output folder
            IOManager.save_record_images(record, record_folder)

            # Step 2: detect + crop + transcribe + structure tables
            table_results = self.record_manager.process_tables(record)
            IOManager.save_table_outputs(record_folder, table_results)

            # Step 3: body OCR (tables/captions already excluded via ocr_excluded_labels)
            ocr_data = self.record_manager.run_ocr(record)

            # Write record to the JSON file
            IOManager.save_record_to_json(record, record_folder, ocr_data, table_results)

            # Export to PDF
            IOManager.generate_pdf_from_record(record_folder, ocr_data)

            # Write record to the CSV file
            IOManager.update_records_csv(record, record_folder, output_path, table_results)

        GPUController.clear_gpu_memory()
        logger.info("Pipeline finished, GPU memory cleared.")