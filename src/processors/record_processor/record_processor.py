from pathlib import Path

from src.processors.record_processor.data import ConfigParameter, TableConfig

from src.processors.record_processor.pipeline import IOManager, RecordManager
from src.processors.record_processor.utils import GPUController

from logging import getLogger
logger = getLogger(__name__)

class RecordProcessor:

    def __init__(self, skip_ocr: bool = False, skip_tables: bool = False,
                 yolo_model_path: str = "model/best.pt",
                 vllm_base_url: str = "http://localhost:8000/v1", vllm_model: str = "qwen3.6-35b-nvfp4"):

        # Labels to exclude from OCR output (add more labels here as needed)
        ocr_excluded_labels = {"Table", "Picture", "Formula", "Caption"}

        # Labels that qualify a detected region as a record-divider header
        header_candidate_labels = {"Section-header", "Title"}

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr, ocr_excluded_labels, header_candidate_labels)
        self.table_config = TableConfig(skip_tables=skip_tables, vllm_base_url=vllm_base_url, vllm_model=vllm_model)

        self.record_manager = RecordManager(self.config, yolo_model_path, self.table_config)

    def run(self, input_path: Path, output_path: Path):
        self.run_records(input_path, output_path)
        self.run_tables(output_path)
        self.run_ocr(output_path)

        GPUController.clear_gpu_memory()
        logger.info("Pipeline finished, GPU memory cleared.")

    def run_records(self, input_path: Path, output_path: Path):
        """ Stage 1: split pages into records and write each record's page images and metadata. """

        image_paths = IOManager.collect_image_files(input_path)
        records = self.record_manager.build_records(image_paths)

        for record in records:
            record_folder = output_path / IOManager.generate_folder_name(record)
            IOManager.save_record_images(record, record_folder)
            IOManager.save_record_meta(record, record_folder)

        GPUController.clear_gpu_memory()
        logger.info(f"Records stage finished. (Count: {len(records)})")

    def run_tables(self, output_path: Path):
        """ Stage 2: detect, transcribe and structure tables on every record written by run_records. """

        record_folders = IOManager.find_record_folders(output_path)

        for record_folder in record_folders:
            record = IOManager.load_record(record_folder)
            table_results = self.record_manager.process_tables(record)
            IOManager.save_table_outputs(record_folder, table_results)
            IOManager.save_tables_index(record_folder, table_results)

        GPUController.clear_gpu_memory()
        logger.info(f"Tables stage finished. (Records: {len(record_folders)})")

    def run_ocr(self, output_path: Path):
        """ Stage 3: body OCR, searchable PDF, ocr_data.json and the index CSV row for every record. """

        record_folders = IOManager.find_record_folders(output_path)

        for record_folder in record_folders:
            record = IOManager.load_record(record_folder)
            ocr_data = self.record_manager.run_ocr(record)
            tables_summary = IOManager.load_tables_index(record_folder)

            IOManager.save_record_to_json(record, record_folder, ocr_data, tables_summary)
            IOManager.generate_pdf_from_record(record_folder, ocr_data)
            IOManager.update_records_csv(record, record_folder, output_path, tables_summary)

        GPUController.clear_gpu_memory()
        logger.info(f"OCR stage finished. (Records: {len(record_folders)})")
