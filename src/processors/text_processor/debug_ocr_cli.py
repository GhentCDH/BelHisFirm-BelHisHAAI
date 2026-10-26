""" TEMPORARY DEBUG: OCR page image(s) with the existing record pipeline models, then run GLiNER on the text.
    Does no record splitting, tables, PDF or folders: all given pages are treated as one record. Delete when the
    entity stage is wired into belhisapp.

    uv run python -m src.processors.text_processor.debug_ocr_cli --input page_001.jpg --print
    uv run python -m src.processors.text_processor.debug_ocr_cli --input /path/to/record_folder --save-json out.json
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from src.processors.record_processor.pipeline import IOManager, RecordManager
from src.processors.record_processor.record_processor import RecordProcessor
from src.processors.text_processor.cli import add_model_arguments, make_extractor, print_record

logger = logging.getLogger(__name__)


def collect_pages(path: Path) -> list[Path]:
    """ The image itself, or the images of a folder in sorted order. """
    return [path] if path.is_file() else IOManager.collect_image_files(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TEMP DEBUG: OCR page images and extract entities with GLiNER2.5.")
    parser.add_argument("--input", required=True, type=Path, help="A page image, or a folder of page images (one record)")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--glm-checkpoint", type=Path, default=None,
                        help="Fine-tuned GLM-OCR checkpoint (default: model/glm-ocr if present, else the base model)")
    add_model_arguments(parser)
    parser.add_argument("--print", dest="show", action="store_true", help="Print the OCR text and the entities")
    parser.add_argument("--save-json", type=Path, default=None, help="Write the OCR result and entities to this file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    pages = collect_pages(args.input)
    if not pages:
        logger.error("No images found in %s", args.input)
        return 1

    glm_checkpoint = args.glm_checkpoint
    if glm_checkpoint is None and Path("model/glm-ocr").exists():
        glm_checkpoint = Path("model/glm-ocr")

    # Same models as the OCR stage: tables are skipped, so no vLLM server or table pipeline is needed
    processor = RecordProcessor(skip_tables=True, yolo_model_path=str(args.weights),
                                glm_checkpoint=str(glm_checkpoint) if glm_checkpoint else None)
    manager = processor.record_manager

    ocr_pages = []
    for number, page_path in enumerate(pages, start=1):
        image = RecordManager.open_page(page_path)
        if image is None:
            continue
        logger.info("[%d/%d] OCR on %s", number, len(pages), page_path.name)
        regions = manager.vision_analyzer.detect_layout(image)
        text_regions = [r for r in regions if r["label"] not in manager.config.ocr_excluded_labels]
        ocr_pages.append(manager.ocr_processor.process_text_blocks(image, text_regions))

    ocr_data = {"record_id": None, "record_title": args.input.name, "pages": ocr_pages}
    extractor = make_extractor(args)
    result = extractor.extract_ocr_data(ocr_data)

    if args.show:
        print_record(result)
    if args.save_json:
        args.save_json.write_text(json.dumps({**result, "pages": ocr_pages}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    logger.info("Done: %d pages, %d entities", len(ocr_pages), len(result["entities"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
