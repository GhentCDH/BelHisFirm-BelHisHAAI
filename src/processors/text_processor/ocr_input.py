""" Finds the records in an input folder and runs the OCR of the record pipeline on the ones that have no
    ocr_data.json yet (same YOLO layout and GLM-OCR models and settings as the OCR stage), without writing anything
    into the folder. The input is one of:
    - the output folder of the record pipeline: one subfolder per record, with page_NNN.jpg and/or ocr_data.json
    - a folder of page images, or a single page image: read as one record
"""

import gc
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.processors.text_processor.entity_extractor import OCR_FILE

logger = logging.getLogger(__name__)

PAGE_PATTERN = re.compile(r"^page_\d{3}$")


@dataclass
class Source:
    """ One record to read: its folder (None for loose images), page images and ocr_data.json structure, if any. """
    name: str
    folder: Path | None = None
    images: list[Path] = field(default_factory=list)
    ocr_data: dict | None = None


def discover(input_path: Path, prefixes: list[str] | None = None) -> list[Source]:
    """ The records of an input folder (or a single image), with their ocr_data.json when there is one. """
    input_path = Path(input_path)
    if input_path.is_file():
        return [Source(input_path.stem, images=[input_path])]

    from src.processors.record_processor.pipeline import IOManager

    folders = sorted({*IOManager.find_record_folders(input_path), *(p.parent for p in input_path.glob(f"*/{OCR_FILE}"))})
    if prefixes:
        folders = [folder for folder in folders if folder.name.startswith(tuple(prefixes))]
    if folders:
        return [_record_source(folder) for folder in folders]

    images = IOManager.collect_image_files(input_path)
    return [Source(input_path.name, images=images)] if images else []


def _record_source(folder: Path) -> Source:
    source = Source(folder.name, folder, sorted(p for p in folder.glob("page_*.jpg") if PAGE_PATTERN.match(p.stem)))
    if (folder / OCR_FILE).exists():
        source.ocr_data = json.loads((folder / OCR_FILE).read_text(encoding="utf-8"))
    return source


def run_ocr(sources: list[Source], weights: Path = Path("model/best.pt"), glm_checkpoint: Path | None = None) -> None:
    """ Fills in ocr_data of these sources: YOLO layout (the saved layout.json of a record folder when it fits the
        weights) and GLM-OCR on every text region, the models are freed again afterwards. """
    from src.processors.record_processor.pipeline import IOManager, RecordManager
    from src.processors.record_processor.record_processor import RecordProcessor
    from src.processors.record_processor.utils import GPUController

    if glm_checkpoint is None and Path("model/glm-ocr").exists():
        glm_checkpoint = Path("model/glm-ocr")
    # Same models as the OCR stage, tables are skipped so no vLLM server is needed
    processor = RecordProcessor(skip_tables=True, yolo_model_path=str(weights),
                                glm_checkpoint=str(glm_checkpoint) if glm_checkpoint else None)
    manager = processor.record_manager
    fingerprint = IOManager.weights_fingerprint(weights)

    for number, source in enumerate(sources, start=1):
        layout = IOManager.load_layout(source.folder, fingerprint, len(source.images)) if source.folder else None
        meta = {}
        if source.folder and (source.folder / "record_meta.json").exists():
            meta = json.loads((source.folder / "record_meta.json").read_text(encoding="utf-8"))

        pages = []
        for idx, path in enumerate(source.images):
            logger.info("[%d/%d] OCR on %s, page %d/%d", number, len(sources), source.name, idx + 1, len(source.images))
            image = RecordManager.open_page(path)
            if image is None:
                continue
            regions = layout[idx] if layout else manager.vision_analyzer.detect_layout(image)
            text_regions = [r for r in regions if r["label"] not in manager.config.ocr_excluded_labels]
            pages.append(manager.ocr_processor.process_text_blocks(image, text_regions))
            GPUController.clear_gpu_memory()

        source.ocr_data = {"record_id": meta.get("record_id"), "record_title": meta.get("record_title") or source.name,
                           "pages": pages}

    del manager, processor
    gc.collect()
    GPUController.clear_gpu_memory()
