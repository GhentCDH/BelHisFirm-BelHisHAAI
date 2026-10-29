""" Turns a Label Studio export of corrected entity pre-annotations into GLiNER training data, and merges it with
    the data you already have. This closes the loop: pre-annotate (labelstudio_cli --commit), correct in Label
    Studio, export as JSON, convert, retrain with gliner_train --data-dir.

    uv run python -m src.utils.gliner_utils.gliner_convert --export ls_export.json \\
        --merge model/gliner/data --out model/gliner/data_round2

    The merged train.json goes to --out, the data you merge with is not changed. test.json is
    copied unchanged, so the scores of the rounds stay comparable (new data is never added to it).

    Both export shapes are read: the Label Studio JSON export (data.text, annotations[].result) and the flat one
    of the tutorial (text, label[]).
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import SCHEMA_PATH
from src.utils.gliner_utils.blocks import clean_value, load_labels

logger = logging.getLogger(__name__)


def results_of(task: dict) -> list[dict] | None:
    """ The entity results of the latest finished annotation of an export task, None when it has none. """
    if "annotations" in task:
        done = [a for a in task["annotations"] if not a.get("was_cancelled") and a.get("result")]
        return [r["value"] for r in done[-1]["result"] if r.get("type") == "labels"] if done else None
    if "label" in task:
        return task["label"]
    return None


def text_of(task: dict) -> str:
    return task["data"]["text"] if "data" in task else task["text"]


def convert_export(tasks: list[dict], labels: list[str]) -> list[dict]:
    """ {text, entities} samples with every label of the schema as a key, like the existing training data. """
    samples = []
    for task in tasks:
        results = results_of(task)
        if results is None:
            continue
        entities = {label: [] for label in labels}
        for value in results:
            text = clean_value(value["text"])
            for label in value.get("labels", []):
                if label in entities and text and text not in entities[label]:
                    entities[label].append(text)
        samples.append({"text": text_of(task), "entities": entities})
    return samples


def write(path: Path, samples: list[dict]) -> None:
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label Studio export -> GLiNER training data.")
    parser.add_argument("--export", required=True, type=Path, help="Label Studio JSON export")
    parser.add_argument("--merge", type=Path, default=None,
                        help="Existing data folder (train.json, test.json) to add the new samples to")
    parser.add_argument("--out", required=True, type=Path, help="Folder for the new data, give --merge a different one")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.merge and args.merge.resolve() == args.out.resolve():
        parser.error("--out must be another folder than --merge")

    labels = list(load_labels(args.schema))
    new = convert_export(json.loads(args.export.read_text(encoding="utf-8")), labels)
    train = json.loads((args.merge / "train.json").read_text(encoding="utf-8")) if args.merge else []
    test_texts = {s["text"] for s in json.loads((args.merge / "test.json").read_text(encoding="utf-8"))} \
        if args.merge and (args.merge / "test.json").exists() else set()

    # Every label of the schema as a key, also in the data of earlier rounds (labels added since are empty there)
    train = [{"text": s["text"], "entities": {label: s["entities"].get(label, []) for label in labels}} for s in train]

    known = {s["text"] for s in train}
    # A text that is already in the test set must stay out of the training data
    added = [s for s in new if s["text"] not in known and s["text"] not in test_texts]
    train = train + added
    logger.info("%d corrected samples in the export, %d added (%d already known or in the test set), %d in train",
                len(new), len(added), len(new) - len(added), len(train))

    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "train.json", train)
    if args.merge and (args.merge / "test.json").exists():
        shutil.copy(args.merge / "test.json", args.out / "test.json")
    logger.info("Wrote %s, train with: gliner_train --data-dir %s", args.out, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
