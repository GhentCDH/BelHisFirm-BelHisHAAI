""" Scores GLiNER2.5 entity extraction on a held-out test set: precision, recall and F1 overall and per entity type,
    for the base model and/or a LoRA adapter from gliner_train, run through the same Joint IE extractor as production.

    uv run python -m src.utils.gliner_utils.gliner_eval --adapter model/gliner/adapters/v25_joint/best
    uv run python -m src.utils.gliner_utils.gliner_eval --adapter model/gliner/adapters/v25_joint/best --base
    uv run python -m src.utils.gliner_utils.gliner_eval --base --test model/gliner/data_round2/test.json

    A prediction is correct when its (type, text) is in the gold entities of the same sample, spans are not compared
    as the training data only has the entity texts. With --adapter and --base both are scored on the same samples.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from src.processors.text_processor.entity_extractor import SCHEMA_PATH, build_schema, extract
from src.utils.gliner_utils.blocks import load_labels

logger = logging.getLogger(__name__)

DEFAULT_TEST = Path("model/gliner/data/test.json")


def score(samples: list[dict], schema_path: Path, adapter: Path | None) -> dict[str, Counter]:
    """ Per entity type (and "ALL") the counts tp, fp and fn over the samples. """
    schema = build_schema(schema_path, adapter)
    counts: dict[str, Counter] = {}
    for sample in samples:
        gold = {(label, text) for label, texts in sample["entities"].items() for text in texts}
        predicted = {(e["type"], e["text"]) for e in extract(sample["text"], schema, adapter)["entities"]}
        for label, text in gold | predicted:
            kind = "tp" if (label, text) in gold and (label, text) in predicted else \
                "fn" if (label, text) in gold else "fp"
            for key in (label, "ALL"):
                counts.setdefault(key, Counter())[kind] += 1
    return counts


def prf(count: Counter) -> tuple[float, float, float]:
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def report(results: dict[str, dict[str, Counter]], labels: list[str]) -> None:
    names = list(results)
    print(f"{'type':<26}{'support':>8}  " + "  ".join(f"{name[:22]:>22}" for name in names))
    print(f"{'':<26}{'':>8}  " + "  ".join(f"{'P / R / F1':>22}" for _ in names))
    for label in [*labels, "ALL"]:
        support = next((c[label]["tp"] + c[label]["fn"] for c in results.values() if label in c), 0)
        if not support and label != "ALL" and not any(label in c for c in results.values()):
            continue
        cells = []
        for counts in results.values():
            p, r, f = prf(counts.get(label, Counter()))
            cells.append(f"{p:6.2f} /{r:6.2f} /{f:6.2f}")
        print(f"{label:<26}{support:>8}  " + "  ".join(f"{cell:>22}" for cell in cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score GLiNER2.5 entity extraction on a test set.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test samples, {text, entities}")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter to score (folder with adapter_config.json)")
    parser.add_argument("--base", action="store_true", help="Also (or only, without --adapter) score the base model")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--out", type=Path, default=None, help="Write the scores to this JSON file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    samples = json.loads(args.test.read_text(encoding="utf-8"))
    runs = {}
    if args.base or not args.adapter:
        runs["base"] = None
    if args.adapter:
        runs[args.adapter.parent.name + "/" + args.adapter.name] = args.adapter

    results = {}
    for name, adapter in runs.items():
        logger.info("Scoring %s on %d samples", name, len(samples))
        results[name] = score(samples, args.schema, adapter)
    report(results, list(load_labels(args.schema)))

    if args.out:
        args.out.write_text(json.dumps({name: {label: dict(c) | dict(zip(("precision", "recall", "f1"), prf(c)))
                                               for label, c in counts.items()} for name, counts in results.items()},
                                       indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
