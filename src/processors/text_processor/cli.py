""" DEBUG command line entry point for the GLiNER entity extraction, temporary until it is wired into belhisapp.

    uv run python -m src.processors.text_processor.cli --output output/ --print   # GLiNER2.5 + adapters
    uv run python -m src.processors.text_processor.cli --output output/ --records 055 019 --dry-run
    uv run python -m src.processors.text_processor.cli --output output/ --adapter-dir model/gliner/adapters/v25
    uv run python -m src.processors.text_processor.cli --text "Landen, le 25 juin 1909." --print

    Train the adapters with gliner_train, pre-annotate for Label Studio with labelstudio_cli.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import (
    DEFAULT_ADAPTER_DIR, DEFAULT_MODEL, DOMAIN_OF, DOMAINS, ENTITIES_FILE, OCR_FILE, SCHEMA_PATH, EntityExtractor,
)

logger = logging.getLogger(__name__)

# ANSI background colors per domain, anything else is shown in grey
DOMAIN_COLORS = {"date": 43, "organisation": 45, "legal": 41, "person": 44, "share": 46, "location": 42}
DEFAULT_COLOR = 100
RESET = "\033[0m"


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """ The arguments that decide which GLiNER is used, shared with the other debug scripts. """
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="GLiNER2.5 checkpoint")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR if DEFAULT_ADAPTER_DIR.exists() else None,
                        help="BelHisFirm LoRA adapters: a folder with one subfolder per domain, or a single adapter. "
                             f"Trained with gliner_train on the same checkpoint as --model, default: {DEFAULT_ADAPTER_DIR} when it exists")
    parser.add_argument("--no-adapters", action="store_true", help="Run the plain model without the adapters")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="Labels with descriptions and thresholds")
    parser.add_argument("--device", default="cuda", help="Device to load the model on")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum confidence for every label (default: the per-label thresholds of the schema)")


def make_extractor(args: argparse.Namespace, **kwargs) -> EntityExtractor:
    adapter_dir = None if args.no_adapters else args.adapter_dir
    return EntityExtractor(args.model, args.device, adapter_dir, args.schema, args.threshold, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEBUG: extract entities from record texts with GLiNER.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--output", type=Path, help="Pipeline output folder with the record folders")
    source.add_argument("--text", help="A single string to try, no folder needed")
    parser.add_argument("--records", nargs="+", default=None, help="Only record folders starting with one of these")
    add_model_arguments(parser)
    parser.add_argument("--print", dest="show", action="store_true", help="Print the highlighted text and the entities")
    parser.add_argument("--dry-run", action="store_true", help=f"Do not write {ENTITIES_FILE}")
    return parser


def highlight_text(text: str, entities: list[dict]) -> str:
    """ The text with every entity on a background color of its domain. Where entities overlap, the longest one sets
        the color and the overlapping characters are underlined. Plain text when output is not a terminal. """
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
        color = DOMAIN_COLORS.get(entity.get("domain") or DOMAIN_OF.get(entity["label"]), DEFAULT_COLOR)
        return f"\033[{color};97{';4' if depth[position] > 1 else ''}m"

    parts, current = [], ""
    for position, character in enumerate(text):
        wanted = style(position)
        if wanted != current:
            parts.append(RESET + wanted if current else wanted)
            current = wanted
        parts.append(character)
    if current:
        parts.append(RESET)
    return "".join(parts)


def legend(entities: list[dict]) -> str:
    """ The domains of these entities in their colors, plus what the underline means. """
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return ""
    domains = [domain for domain in DOMAINS if any(e.get("domain") == domain for e in entities)]
    return "  ".join(f"\033[{DOMAIN_COLORS[domain]};97m {domain} {RESET}" for domain in domains) + \
        "  (underlined = overlapping entities)"


def print_entities(entities: list[dict]) -> None:
    for label in sorted({entity["label"] for entity in entities}):
        print(f"  {label}:")
        for entity in (e for e in entities if e["label"] == label):
            print(f"    {entity['confidence']:.2f}  {entity['text']}")


def print_record(result: dict) -> None:
    """ The text blocks of a record with their entities highlighted, then the entities per label. """
    for block in result.get("blocks", []):
        print(highlight_text(block["text"], block["entities"]))
        print()
    print(legend(result["entities"]))
    print()
    print_entities(result["entities"])


def find_records(output: Path, prefixes: list[str] | None) -> list[Path]:
    folders = sorted(path.parent for path in output.glob(f"*/{OCR_FILE}"))
    if prefixes:
        folders = [folder for folder in folders if folder.name.startswith(tuple(prefixes))]
    return folders


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.output is None and args.text is None:
        build_parser().error("one of the arguments --output --text is required")
    extractor = make_extractor(args)

    if args.text is not None:
        entities = extractor.extract_text(args.text)
        if args.show:
            print(highlight_text(args.text, entities))
            print()
            print(legend(entities))
            print()
        print_entities(entities)
        return 0

    folders = find_records(args.output, args.records)
    if not folders:
        logger.error("No record folders with %s found in %s", OCR_FILE, args.output)
        return 1

    loaded, failed = [], 0
    for folder in folders:
        try:
            loaded.append((folder, json.loads((folder / OCR_FILE).read_text(encoding="utf-8"))))
        except Exception:
            failed += 1
            logger.exception("Could not read %s", folder.name)

    # All records at once: with domain adapters each adapter is then only activated once
    ok = [folder for folder, _ in loaded]
    results = [extractor.build_result(record) for record in extractor.extract_blocks([data for _, data in loaded])]

    for folder, result in zip(ok, results):
        if not args.dry_run:
            extractor.save(folder, result)
        logger.info("%s: %d entities", folder.name, len(result["entities"]))
        if args.show:
            print(folder.name)
            print_record(result)

    logger.info("Done: %d records, %d failed", len(folders), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
