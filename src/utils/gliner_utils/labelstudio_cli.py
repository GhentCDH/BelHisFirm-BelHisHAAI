""" Pre-annotates the detections of a folder with GLiNER2.5 (Joint IE, optionally with a LoRA adapter from gliner_train, --adapter) and sends them to Label Studio, every detection (YOLO
    text block with its OCR text) as a task of its own, so the entities can be corrected there and used as training
    data (see gliner_convert). Only the Text and Section-header detections (--labels), at most 10 per page
    (--max-per-page): the largest ones by surface, 20% section headers and 80% texts (--split), and at most
    --max-pages pages, a random pick that is the same every run. Runs the OCR of the record pipeline on what has no
    ocr_data.json yet.

    # Dry run: writes the tasks to labelstudio_ner_tasks.json
    uv run python -m src.utils.gliner_utils.labelstudio_cli --input /path/to/pages --project 120
    # A few first, then everything: tasks that are already in the project are left out
    uv run python -m src.utils.gliner_utils.labelstudio_cli --input /path/to/pages --project 120 --commit --limit 20
    uv run python -m src.utils.gliner_utils.labelstudio_cli --input /path/to/pages --project 120 --commit
    # Import the tasks of an earlier run, without reading the OCR again
    uv run python -m src.utils.gliner_utils.labelstudio_cli --from-tasks labelstudio_ner_tasks.json --project 120 --commit

    The input is a folder of page images (one record), a single page image, or the output folder of the record
    pipeline: one subfolder per record with page_NNN.jpg and/or ocr_data.json. The OCR (YOLO layout + GLM-OCR, the
    models and checkpoints of the OCR stage) runs for the records without ocr_data.json, or for all with --redo-ocr.
    Nothing is written into the input folder. The project needs a `text` Text tag and a `label` Labels tag with the
    labels of the schema (project 120 has them, --print-config prints the config).
    The text around each detection on its page (--context-chars) goes along as `context_before` and `context_after`,
    shown by that config above and below the text, not annotated. LS_URL and LS_TOKEN come from the
    .env file in the repo root (see .env.example, never pass the token on the CLI), without --project
    LS_NER_PROJECT_ID is used.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

from src.processors.text_processor.entity_extractor import SCHEMA_PATH
from src.utils.gliner_utils.blocks import (DEFAULT_CONTEXT_CHARS, DEFAULT_SPLIT, TEXT_LABELS, build_blocks, labelstudio_config,
                                           load_labels, predict_blocks, to_labelstudio_tasks)
from src.utils.gliner_utils.ocr_input import Source, discover, keep_pages, run_ocr, select_pages
from src.utils.labelstudio_preannotate.client import LabelStudioClient

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("labelstudio_ner_tasks.json")
BATCH_SIZE = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send the detections of a pipeline output folder to Label Studio, "
                                                 "pre-annotated with GLiNER2.5, one task per detection.")
    parser.add_argument("--input", type=Path,
                        help="Folder of page images, a page image, or the output folder of the record pipeline")
    parser.add_argument("--from-tasks", type=Path, default=None,
                        help="Tasks file of an earlier run, to import without reading the input and the OCR again")
    parser.add_argument("--labels", nargs="+", default=list(TEXT_LABELS),
                        help="Layout labels of the detections to take, 'all' for every text detection "
                             "(default: %(default)s)")
    parser.add_argument("--max-per-page", type=int, default=10,
                        help="At most this many detections per page, the largest ones by surface in reading order "
                             "(0 = all, default: %(default)s)")
    parser.add_argument("--split", nargs="+", metavar="LABEL=PARTS",
                        default=[f"{label}={parts}" for label, parts in DEFAULT_SPLIT.items()],
                        help="How the detections per page are divided over the layout labels, a label with too few "
                             "leaves its places to the others (default: %(default)s)")
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS,
                        help="Characters of the text before and after a detection on its page, sent along as context "
                             "to read it in (0 = none, default: %(default)s)")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="At most this many pages over the whole input, a random pick that is the same every run "
                             "(0 = all, default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the pick of pages")
    parser.add_argument("--redo-ocr", action="store_true", help="Run the OCR also where there is an ocr_data.json")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--glm-checkpoint", type=Path, default=None,
                        help="Fine-tuned GLM-OCR checkpoint (default: model/glm-ocr if present, else the base model)")
    parser.add_argument("--records", nargs="+", default=None, help="Only record folders starting with one of these")
    parser.add_argument("--project", type=int, default=None, help="Label Studio project id (default: LS_NER_PROJECT_ID)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Write the tasks to this file")
    parser.add_argument("--commit", action="store_true", help="Import the tasks into the Label Studio project")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tasks (not in the project yet, with --commit)")
    parser.add_argument("--no-predictions", action="store_true", help="Plain tasks without entities, no model is loaded")
    parser.add_argument("--print-config", action="store_true", help="Print the labeling config for a new project and exit")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="LoRA adapter trained with gliner_train (a folder, or its final/ checkpoint)")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="IEschema.json, the entity types of the project")
    return parser


def make_client(project: int | None) -> LabelStudioClient:
    from dotenv import load_dotenv

    load_dotenv(".env")
    token = os.environ.get("LS_TOKEN")
    project = project or (int(os.environ["LS_NER_PROJECT_ID"]) if os.environ.get("LS_NER_PROJECT_ID") else None)
    if not token or project is None:
        raise SystemExit("Give --project and set LS_TOKEN (and LS_URL) in the .env file, see .env.example")
    return LabelStudioClient(os.environ.get("LS_URL", "http://localhost:8080"), token, project)


def check_project(client: LabelStudioClient, labels: list[str]) -> None:
    """ Fails when the project cannot be read or cannot take the tasks: no `text`/`label` tags, or without the
        labels of the schema (Label Studio would not show those entities). """
    response = requests.get(f"{client.base_url}/api/projects/{client.project_id}/", headers=client.headers, timeout=30)
    if not response.ok:
        raise SystemExit(f"Cannot read project {client.project_id}: HTTP {response.status_code} {response.text[:200]}")
    project = response.json()
    config = project.get("label_config") or ""
    if not re.search(r'<Text\s+name="text"\s+value="\$text"', config) or not re.search(r'<Labels\s+name="label"\s+toName="text"', config):
        raise SystemExit(f"Project {client.project_id} needs a Text tag `text` (value=$text) and a Labels tag `label` "
                         f"(toName=text), see --print-config")
    missing = [label for label in labels if f'value="{label}"' not in config]
    if missing:
        raise SystemExit(f"Project {client.project_id} has no label for: {', '.join(missing)}")
    logger.info("Project %s '%s': %s tasks, labeling config ok", project["id"], project.get("title"), project.get("task_number"))


def existing_sources(client: LabelStudioClient) -> set[str]:
    """ The `source` of every task already in the project, to not import a detection twice. """
    sources, page = set(), 1
    while True:
        response = requests.get(f"{client.base_url}/api/tasks", headers=client.headers, timeout=60,
                                params={"project": client.project_id, "page": page, "page_size": 1000, "fields": "all"})
        if response.status_code == 404:  # Past the last page
            break
        if not response.ok:
            raise SystemExit(f"Cannot list the tasks of project {client.project_id}: HTTP {response.status_code} {response.text[:200]}")
        data = response.json()
        tasks = data if isinstance(data, list) else data.get("tasks", data.get("results", []))
        sources.update(task["data"]["source"] for task in tasks if "source" in (task.get("data") or {}))
        if len(tasks) < 1000:
            break
        page += 1
    return sources


def block_labels(args: argparse.Namespace) -> tuple[str, ...] | None:
    return None if "all" in args.labels else tuple(args.labels)


def max_per_page(args: argparse.Namespace) -> int | None:
    return args.max_per_page or None


def split(args: argparse.Namespace) -> dict[str, float]:
    return {label: float(parts) for label, _, parts in (item.partition("=") for item in args.split)}


def collect_tasks(args: argparse.Namespace, sources: list[Source]) -> list[dict]:
    """ The tasks of every detection of these records, with the entities of GLiNER unless --no-predictions. """
    version = "none" if args.no_predictions else f"gliner2.5-joint{'+' + args.adapter.name if args.adapter else ''}"
    tasks = []
    for source in sources:
        blocks = build_blocks(source.ocr_data, block_labels(args), max_per_page(args), split(args), args.context_chars)
        blocks = [{**block, "entities": []} for block in blocks] if args.no_predictions \
            else predict_blocks(blocks, args.adapter)
        record = {"record_id": source.ocr_data.get("record_id"), "record_title": source.ocr_data.get("record_title"),
                  "blocks": blocks}
        tasks.extend(to_labelstudio_tasks(record, source.name, version))
    return tasks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.print_config:
        print(labelstudio_config())
        return 0
    if args.input is None and args.from_tasks is None:
        build_parser().error("--input or --from-tasks is required")

    client = None
    if args.commit:
        client = make_client(args.project)
        check_project(client, list(load_labels(args.schema)))

    if args.from_tasks:
        tasks = json.loads(args.from_tasks.read_text(encoding="utf-8"))
    else:
        sources = discover(args.input, args.records)
        if not sources:
            logger.error("No records or page images found in %s", args.input)
            return 1
        selected = select_pages(sources, args.max_pages, args.seed, block_labels(args)) if args.max_pages else None
        to_read = [source for source in sources if source.ocr_data is None or args.redo_ocr]
        if to_read:
            logger.info("Running the OCR on %d of %d records", len(to_read), len(sources))
            run_ocr(to_read, args.weights, args.glm_checkpoint, block_labels(args), max_per_page(args), split(args), selected)
        if selected:
            for source in sources:
                keep_pages(source, selected[source.name])
        tasks = collect_tasks(args, sources)
        logger.info("%d records, %d detections", len(sources), len(tasks))

    if client:
        known = existing_sources(client)
        tasks = [task for task in tasks if task["data"]["source"] not in known]
        logger.info("%d of them are not in project %s yet", len(tasks), client.project_id)
    if args.limit is not None:
        tasks = tasks[:args.limit]

    args.out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d tasks to %s", len(tasks), args.out)

    if client:
        for start in range(0, len(tasks), BATCH_SIZE):
            client.import_tasks(tasks[start:start + BATCH_SIZE])
            logger.info("Imported %d/%d", min(start + BATCH_SIZE, len(tasks)), len(tasks))
        logger.info("Done, the tasks are in project %s", client.project_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
