""" DEBUG command line entry point for the GLiNER entity extraction, temporary until it is wired into belhisapp.

    uv run python -m src.processors.text_processor.cli --output output/ --print
    uv run python -m src.processors.text_processor.cli --output output/ --records 055 019 --threshold 0.4 --dry-run
    uv run python -m src.processors.text_processor.cli --text "Landen, le 25 juin 1909." --print
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import (
    DEFAULT_MODEL, DEFAULT_THRESHOLD, ENTITIES_FILE, OCR_FILE, EntityExtractor,
)

logger = logging.getLogger(__name__)

# ANSI background colors per label, labels not listed here are shown in grey
LABEL_COLORS = {"person": 44, "company": 45, "place": 42, "date": 43, "money": 46, "role": 41}
DEFAULT_COLOR = 100
RESET = "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEBUG: extract entities from record texts with GLiNER2.5.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path, help="Pipeline output folder with the record folders")
    source.add_argument("--text", help="A single string to try, no folder needed")
    parser.add_argument("--records", nargs="+", default=None, help="Only record folders starting with one of these")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GLiNER2 / GLiNER2.5 checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to load the model on")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum confidence")
    parser.add_argument("--print", dest="show", action="store_true", help="Print the entities grouped by label")
    parser.add_argument("--dry-run", action="store_true", help=f"Do not write {ENTITIES_FILE}")
    return parser


def highlight_text(text: str, entities: list[dict]) -> str:
    """ The text with every entity on a colored background, plus a legend. Where entities overlap, the longest one
        sets the color and the overlapping characters are underlined. Plain text when output is not a terminal. """
    labels = sorted({entity["label"] for entity in entities})
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return text

    # Per character: the longest entity covering it and how many entities cover it
    owner: list[dict | None] = [None] * len(text)
    depth = [0] * len(text)
    for entity in sorted(entities, key=lambda e: e["end"] - e["start"]):
        for position in range(entity["start"], min(entity["end"], len(text))):
            owner[position] = entity
            depth[position] += 1

    def style(position: int) -> str:
        entity = owner[position]
        if entity is None:
            return ""
        code = f"{LABEL_COLORS.get(entity['label'], DEFAULT_COLOR)};97"
        return f"\033[{code}{';4' if depth[position] > 1 else ''}m"

    parts, current = [], ""
    for position, character in enumerate(text):
        wanted = style(position)
        if wanted != current:
            parts.append(RESET + wanted if current else wanted)
            current = wanted
        parts.append(character)
    if current:
        parts.append(RESET)

    legend = "  ".join(f"\033[{LABEL_COLORS.get(label, DEFAULT_COLOR)};97m {label} {RESET}" for label in labels)
    return "".join(parts) + f"\n\n{legend}  (underlined = overlapping entities)"


def print_entities(entities: list[dict]) -> None:
    for label in sorted({entity["label"] for entity in entities}):
        print(f"  {label}:")
        for entity in (e for e in entities if e["label"] == label):
            print(f"    {entity['confidence']:.2f}  {entity['text']}")


def find_records(output: Path, prefixes: list[str] | None) -> list[Path]:
    folders = sorted(path.parent for path in output.glob(f"*/{OCR_FILE}"))
    if prefixes:
        folders = [folder for folder in folders if folder.name.startswith(tuple(prefixes))]
    return folders


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    extractor = EntityExtractor(args.model, args.device, args.threshold)

    if args.text is not None:
        entities = extractor.extract_text(args.text)
        if args.show:
            print(highlight_text(args.text, entities))
            print()
        print_entities(entities)
        return 0

    folders = find_records(args.output, args.records)
    if not folders:
        logger.error("No record folders with %s found in %s", OCR_FILE, args.output)
        return 1

    failed = 0
    for folder in folders:
        try:
            result = extractor.extract_record(folder)
            if not args.dry_run:
                extractor.save(folder, result)
            logger.info("%s: %d entities", folder.name, len(result["entities"]))
            if args.show:
                print(f"{folder.name}")
                ocr_data = json.loads((folder / OCR_FILE).read_text(encoding="utf-8"))
                print(highlight_text(extractor.build_record_text(ocr_data)[0], result["entities"]))
                print()
                print_entities(result["entities"])
        except Exception:
            failed += 1
            logger.exception("Entity extraction failed for %s", folder.name)
    logger.info("Done: %d records, %d failed", len(folders), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
