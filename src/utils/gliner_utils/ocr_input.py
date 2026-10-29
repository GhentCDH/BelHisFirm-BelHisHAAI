""" Finds the records in an input folder and runs the OCR of the record pipeline on the ones that have no
    ocr_data.json yet (same YOLO layout and GLM-OCR models and settings as the OCR stage), without writing anything
    into the folder. The input is one of:
    - the output folder of the record pipeline: one subfolder per record, with page_NNN.jpg and/or ocr_data.json
    - a folder of page images, or a single page image: read as one record
"""

import gc
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.processors.text_processor.entity_extractor import OCR_FILE, pick_largest

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


def select_pages(sources: list[Source], max_pages: int, seed: int = 42,
                 labels: tuple[str, ...] | None = None) -> dict[str, set[int]]:
    """ At most max_pages pages over all records, a random pick (the same for the same seed and input). Returns per
        record name the page numbers (0-based, as in ocr_data.json) that are taken. Of records with an ocr_data.json
        only the pages with text in these layout labels can be picked, the others are only known after the OCR. """
    def has_text(page: dict) -> bool:
        return any((block.get("text") or "").strip() and (labels is None or block.get("label") in labels)
                   for block in page.get("blocks", []))

    pages = [(source.name, idx) for source in sources
             for idx in (range(len(source.images)) if source.ocr_data is None else
                         [i for i, page in enumerate(source.ocr_data["pages"]) if has_text(page)])]
    picked = random.Random(f"{seed}-pages").sample(pages, min(max_pages, len(pages)))
    selected: dict[str, set[int]] = {source.name: set() for source in sources}
    for name, idx in picked:
        selected[name].add(idx)
    return selected


def keep_pages(source: Source, pages: set[int]) -> None:
    """ Empties the pages of an ocr_data that are not taken, page numbers stay what they are. """
    source.ocr_data = {**source.ocr_data, "pages": [
        page if idx in pages else {**page, "blocks": []} for idx, page in enumerate(source.ocr_data["pages"])]}


def run_ocr(sources: list[Source], weights: Path = Path("model/best.pt"), glm_checkpoint: Path | None = None,
            labels: tuple[str, ...] | None = None, max_per_page: int | None = None,
            split: dict[str, float] | None = None, only_pages: dict[str, set[int]] | None = None) -> None:
    """ Fills in ocr_data of these sources: YOLO layout (the saved layout.json of a record folder when it fits the
        weights) and GLM-OCR on the text regions, the models are freed again afterwards.

        Args: labels: Only the regions with these layout labels, None for all text regions.
        Args: max_per_page: At most this many regions per page, the largest ones divided over the layout labels as
              split says (pick_largest), the others are not read at all. None for all of them.
        Args: only_pages: Per record the pages (0-based) to read, the others are left empty. None for all pages. """
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
            if only_pages is not None and idx not in only_pages[source.name]:
                pages.append({"blocks": [], "spine_position": None})  # Kept, so the page numbers stay what they are
                continue
            logger.info("[%d/%d] OCR on %s, page %d/%d", number, len(sources), source.name, idx + 1, len(source.images))
            image = RecordManager.open_page(path)
            if image is None:
                continue
            regions = layout[idx] if layout else manager.vision_analyzer.detect_layout(image)
            text_regions = [r for r in regions if r["label"] not in manager.config.ocr_excluded_labels
                            and (labels is None or r["label"] in labels)]
            text_regions = pick_largest(text_regions, max_per_page, split)
            pages.append(manager.ocr_processor.process_text_blocks(image, text_regions))
            GPUController.clear_gpu_memory()

        source.ocr_data = {"record_id": meta.get("record_id"), "record_title": meta.get("record_title") or source.name,
                           "pages": pages}

    del manager, processor
    gc.collect()
    GPUController.clear_gpu_memory()
