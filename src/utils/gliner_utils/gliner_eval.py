""" Scores GLiNER2.5 entity extraction on a held-out test set: precision, recall and F1 overall and per entity type, for
    the base model and/or the LoRA adapters of gliner_train (a single adapter, or a folder with one adapter per domain).

    uv run python -m src.utils.gliner_utils.gliner_eval --base
    uv run python -m src.utils.gliner_utils.gliner_eval --adapter-dir model/gliner/adapters/v25_domains --base
    uv run python -m src.utils.gliner_utils.gliner_eval --adapter-dir model/gliner/adapters/v25_single --base --threshold 0.3
    uv run python -m src.utils.gliner_utils.gliner_eval --base --joint     # through the Joint IE extractor, with relations

    --v2 adds the GLiNER2 (not 2.5) domain adapters (--v2-dir, default model/gliner/adapters/v2), --v2-base the GLiNER2
    base model, next to --base for GLiNER2.5:
    uv run python -m src.utils.gliner_utils.gliner_eval --adapter-dir model/gliner/adapters/v25_domains --base --v2 --v2-base

    --run scores more models in the same table, as NAME=MODEL or NAME=MODEL@ADAPTER_DIR, e.g. the GLiNER2 (not 2.5) domain
    adapters and their base model:
    uv run python -m src.utils.gliner_utils.gliner_eval --adapter-dir model/gliner/adapters/v25_domains --base \\
        --run v2_base=fastino/gliner2-multi-v1 --run v2_domains=fastino/gliner2-multi-v1@model/gliner/adapters/v2

    Does more training data help? Train adapters on the data before and after a round (gliner_train --data-dir ... --out
    ...), then score both on the same untouched test set, with a bootstrap over the samples that says whether the
    difference is more than noise (--baseline is the run the others are compared with):
    uv run python -m src.utils.gliner_utils.gliner_eval --base --baseline v25_round1 \\
        --adapter-dir model/gliner/adapters/v25_round1 model/gliner/adapters/v25_round2

    As in the tutorial notebook a prediction is correct when its (type, text) is in the gold entities of the same sample,
    compared lowercased and stripped, spans are not compared as the training data only has the entity texts. Counts are
    summed over the samples (micro). With --adapter-dir and --base both are scored on the same samples.
"""

import argparse
import gc
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, SCHEMA_PATH, build_schema, extract
from src.utils.gliner_utils.blocks import clean_value, load_labels

logger = logging.getLogger(__name__)

DEFAULT_TEST = Path("model/gliner/data/test.json")
V2_MODEL = "fastino/gliner2-multi-v1"
V2_DIR = Path("model/gliner/adapters/v2")


def norm(text: str) -> str:
    return " ".join(clean_value(text).lower().split())


def predict_joint(samples: list[dict], schema_path: Path, adapter: Path | None) -> list[set]:
    """ The (type, text) of the entities of the Joint IE extractor, per sample. """
    schema = build_schema(schema_path, adapter)
    return [{(e["type"], norm(e["text"])) for e in extract(s["text"], schema, adapter)["entities"]}
            for s in tqdm(samples, desc="joint", unit="sample")]


def predict_entities(samples: list[dict], extractor) -> list[set]:
    """ The (type, text) of the entities of an EntityExtractor, per sample. """
    found = extractor.extract_texts([s["text"] for s in samples])
    return [{(e["label"], norm(e["text"])) for e in entities} for entities in found]


def score(samples: list[dict], predicted: list[set], scope: set[str]) -> tuple[dict[str, Counter], list[Counter]]:
    """ Per entity type (and "ALL") the counts tp, fp and fn, over the entity types in scope only, and the counts of
        every sample (for the bootstrap). """
    counts: dict[str, Counter] = {}
    per_sample: list[Counter] = []
    for sample, pred in zip(samples, predicted):
        per_sample.append(Counter())
        gold = {(label, norm(text)) for label, texts in sample["entities"].items() if label in scope for text in texts}
        pred = {item for item in pred if item[0] in scope}
        for item in gold | pred:
            kind = "tp" if item in gold and item in pred else "fn" if item in gold else "fp"
            per_sample[-1][kind] += 1
            for key in (item[0], "ALL"):
                counts.setdefault(key, Counter())[kind] += 1
    return counts, per_sample


def prf(count: Counter) -> tuple[float, float, float]:
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap(per_sample: dict[str, list[Counter]], baseline: str | None, rounds: int = 1000, seed: int = 42) -> None:
    """ The 95% interval of the overall F1 of every run, and of its difference with the baseline run, over samples
        drawn with replacement (the same draws for every run, so the difference is paired). """
    names = list(per_sample)
    size = len(next(iter(per_sample.values())))
    rng = random.Random(seed)
    draws = [[rng.randrange(size) for _ in range(size)] for _ in range(rounds)]

    def f1(counts: list[Counter], indexes: list[int]) -> float:
        total = Counter()
        for i in indexes:
            total.update(counts[i])
        return prf(total)[2]

    scores = {name: sorted(f1(per_sample[name], draw) for draw in draws) for name in names}
    lo, hi = int(rounds * 0.025), int(rounds * 0.975) - 1
    print("\nOverall F1, 95% interval over the samples (bootstrap):")
    for name in names:
        print(f"  {name:<24}{f1(per_sample[name], list(range(size))):.3f}  [{scores[name][lo]:.3f}, {scores[name][hi]:.3f}]")
    if baseline in per_sample and len(names) > 1:
        print(f"Difference with {baseline} (paired):")
        for name in names:
            if name == baseline:
                continue
            diffs = sorted(f1(per_sample[name], d) - f1(per_sample[baseline], d) for d in draws)
            verdict = "better" if diffs[lo] > 0 else "worse" if diffs[hi] < 0 else "not clearly different"
            print(f"  {name:<24}{f1(per_sample[name], list(range(size))) - f1(per_sample[baseline], list(range(size))):+.3f}"
                  f"  [{diffs[lo]:+.3f}, {diffs[hi]:+.3f}]  {verdict}")


def report(results: dict[str, dict[str, Counter]], labels: list[str]) -> None:
    names = list(results)
    print(f"\n{'type':<26}{'support':>8}  " + "  ".join(f"{name[:24]:>24}" for name in names))
    print(f"{'':<26}{'':>8}  " + "  ".join(f"{'P / R / F1':>24}" for _ in names))
    for label in [*labels, "ALL"]:
        counts = [c[label] for c in results.values() if label in c]
        if not counts and label != "ALL":
            continue
        support = counts[0]["tp"] + counts[0]["fn"] if counts else 0
        cells = ["{:6.2f} /{:6.2f} /{:6.2f}".format(*prf(c.get(label, Counter()))) if label in c or label == "ALL" else "-"
                 for c in results.values()]
        print(f"{label:<26}{support:>8}  " + "  ".join(f"{cell:>24}" for cell in cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score GLiNER2.5 entity extraction on a test set.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test samples, {text, entities}")
    parser.add_argument("--adapter-dir", type=Path, nargs="+", default=[],
                        help="A single LoRA adapter, or a folder with one per domain (see gliner_train), several to compare")
    parser.add_argument("--baseline", default=None,
                        help="Name of the run the others are compared with in the bootstrap (default: the first one)")
    parser.add_argument("--base", action="store_true", help="Also (or only, without --adapter-dir) score the base model")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum confidence for every type (default: 0.5, or the `threshold` of the schema)")
    parser.add_argument("--v2", action="store_true", help=f"Also score the GLiNER2 domain adapters ({V2_MODEL}, --v2-dir)")
    parser.add_argument("--v2-base", action="store_true", help=f"Also score the GLiNER2 base model ({V2_MODEL})")
    parser.add_argument("--v2-dir", type=Path, default=V2_DIR, help="Folder with the GLiNER2 domain adapters (default: %(default)s)")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=MODEL[@ADAPTER_DIR]",
                        help="Also score another checkpoint, with the adapters in ADAPTER_DIR if given (repeatable)")
    parser.add_argument("--joint", action="store_true",
                        help="Use the Joint IE extractor (with the relations of the schema, a single adapter only)")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--out", type=Path, default=None, help="Write the scores to this JSON file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    samples = json.loads(args.test.read_text(encoding="utf-8"))
    labels = list(load_labels(args.schema))
    runs = {}  # name -> (model, adapter folder)
    if args.base or not (args.adapter_dir or args.v2 or args.v2_base or args.run):
        runs["base"] = (DEFAULT_MODEL, None)
    for folder in args.adapter_dir:
        runs[folder.name] = (DEFAULT_MODEL, folder)
    if args.v2_base:
        runs["v2_base"] = (V2_MODEL, None)
    if args.v2:
        runs["v2_domains"] = (V2_MODEL, args.v2_dir)
    for spec in args.run:
        name, _, rest = spec.partition("=")
        model, _, folder = rest.partition("@")
        runs[name] = (model, Path(folder) if folder else None)

    results = {}
    per_sample = {}
    for name, (model_name, adapter_dir) in runs.items():
        print(f"Scoring {name} on {len(samples)} samples ...", flush=True)
        if args.joint:
            from src.processors.text_processor.entity_extractor import find_adapter
            predicted = predict_joint(samples, args.schema, find_adapter(adapter_dir) if adapter_dir else None)
            scope = set(labels)
        else:
            from src.utils.gliner_utils.entities import EntityExtractor
            extractor = EntityExtractor(model_name, adapter_dir=adapter_dir, schema_path=args.schema, threshold=args.threshold)
            scope = {label for _, passed in extractor._passes() for label in passed}
            print(f"Adapters: {', '.join(extractor.adapters) or 'none'}", flush=True)
            predicted = predict_entities(samples, extractor)
            del extractor
            gc.collect()
        results[name], per_sample[name] = score(samples, predicted, scope)
    report(results, labels)
    bootstrap(per_sample, args.baseline or next(iter(results)))

    if args.out:
        args.out.write_text(json.dumps({name: {label: dict(c) | dict(zip(("precision", "recall", "f1"), prf(c)))
                                               for label, c in counts.items()} for name, counts in results.items()},
                                       indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
