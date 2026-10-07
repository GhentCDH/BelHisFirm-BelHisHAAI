from pathlib import Path

from collections.abc import Callable

from src.processors.record_processor.data import ConfigParameter, TableConfig, TABLE_STEPS

from src.processors.record_processor.pipeline import IOManager, RecordManager
from src.processors.record_processor.utils import GPUController

from logging import getLogger
logger = getLogger(__name__)

class RecordProcessor:

    def __init__(self, skip_ocr: bool = False, skip_tables: bool = False,
                 yolo_model_path: str = "model/best.pt",
                 vllm_base_url: str = "http://localhost:8000/v1", vllm_model: str = "qwen3.6-35b-nvfp4",
                 table_steps: list[str] | None = None, glm_checkpoint: str | None = None,
                 structuring_concurrency: int = 8, focus_shareholders: bool = False):

        # Labels to exclude from OCR output (add more labels here as needed)
        ocr_excluded_labels = {"Table", "Picture", "Formula", "Caption"}

        # Labels that qualify a detected region as a record-divider header
        header_candidate_labels = {"Section-header", "Title"}

        self.config = ConfigParameter(50, 0.85, 0.4, 200, 300, skip_ocr, ocr_excluded_labels, header_candidate_labels)
        self.table_config = TableConfig(
            skip_tables=skip_tables, vllm_base_url=vllm_base_url, vllm_model=vllm_model,
            # Always in pipeline order, whatever order they were given in
            steps=[step for step in TABLE_STEPS if table_steps is None or step in table_steps],
            glm_checkpoint=glm_checkpoint, structuring_concurrency=structuring_concurrency,
            focus_shareholders=focus_shareholders,
        )

        self.yolo_model_path = Path(yolo_model_path)
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
        fingerprint = IOManager.weights_fingerprint(self.yolo_model_path)
        record_count = 0

        # Saved one by one as they are completed, all pages of a volume do not fit in memory at once
        for record in self.record_manager.iter_records(image_paths):
            record_count += 1
            record_folder = output_path / IOManager.generate_folder_name(record)
            IOManager.save_record_images(record, record_folder)
            IOManager.save_record_meta(record, record_folder)
            # Already detected while splitting, so the tables and OCR stages do not detect it again
            IOManager.save_layout(record_folder, record.layout, fingerprint)

        GPUController.clear_gpu_memory()
        logger.info(f"Records stage finished. (Count: {record_count})")

    def _layout_for(self, record, record_folder: Path) -> list[list[dict]]:
        """ Layout regions for every page of a record, as saved by the records stage. Only detected here for
            record folders without a usable layout.json (older runs, or other weights). """

        fingerprint = IOManager.weights_fingerprint(self.yolo_model_path)
        layout = IOManager.load_layout(record_folder, fingerprint, len(record.images))

        if layout is None:
            layout = self.record_manager.detect_layout(record)
            IOManager.save_layout(record_folder, layout, fingerprint)

        return layout

    def run_tables(self, output_path: Path, on_step: Callable[[str, str], None] | None = None):
        """ Stage 2: the table sub-stages in TableConfig.steps, each over every record written by run_records.

            Args: on_step: Called with (step, "start" | "done") around each sub-stage.
        """

        if self.table_config.skip_tables:
            return

        record_folders = IOManager.find_record_folders(output_path)
        table_pipeline = self.record_manager.table_pipeline

        for step in self.table_config.steps:
            if on_step:
                on_step(step, "start")

            if step == "crop":
                found = 0
                for number, record_folder in enumerate(record_folders, start=1):
                    record = IOManager.load_record(record_folder)
                    tables = table_pipeline.crop(record, record_folder, self._layout_for(record, record_folder))
                    logger.info(
                        f"[{number}/{len(record_folders)}] {record_folder.name}: "
                        f"{tables} table(s) on {len(record.images)} page(s)"
                    )
                    found += tables
                logger.info(f"Table detection finished. (Tables found: {found})")

            elif step == "transcribe":
                transcribed = table_pipeline.transcribe(record_folders)
                GPUController.clear_gpu_memory()
                logger.info(f"Table transcription finished. (Tables transcribed: {transcribed})")

            elif step == "structure":
                structured, failed = table_pipeline.structure(record_folders)
                logger.info(f"Table structuring finished. (Tables structured: {structured})")
                if failed:
                    logger.error(f"{failed} table(s) failed structuring, see the FAILED folder of their record.")

            elif step == "parse":
                parsed = table_pipeline.parse(record_folders)
                logger.info(f"Rule parsing finished. (Shareholder registers parsed: {parsed})")

            elif step == "excel":
                exported = table_pipeline.excel(record_folders)
                logger.info(f"Excel export finished. (Tables exported: {exported})")

            if on_step:
                on_step(step, "done")

        GPUController.clear_gpu_memory()
        logger.info(f"Tables stage finished. (Records: {len(record_folders)})")

    def run_ocr(self, output_path: Path):
        """ Stage 3: body OCR, searchable PDF, ocr_data.json and the index CSV row for every record. """

        record_folders = IOManager.find_record_folders(output_path)

        for record_folder in record_folders:
            record = IOManager.load_record(record_folder)
            layout = self._layout_for(record, record_folder)
            ocr_data = self.record_manager.run_ocr(record, layout)
            tables_summary = IOManager.load_tables_index(record_folder)

            IOManager.save_record_to_json(record, record_folder, ocr_data, tables_summary)
            IOManager.generate_pdf_from_record(record_folder, ocr_data)
            IOManager.update_records_csv(record, record_folder, output_path, tables_summary)

        GPUController.clear_gpu_memory()
        logger.info(f"OCR stage finished. (Records: {len(record_folders)})")
