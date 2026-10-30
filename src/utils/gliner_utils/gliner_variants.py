""" Trains and scores every variant of the GLiNER entity extraction, to see what works best:
    - model:     GLiNER2 (v2, fastino/gliner2-multi-v1) and GLiNER2.5 (v25, fastino/gliner2.5-multi-v1)
    - structure: one LoRA adapter per domain (domain) or one for all entity types (single)
    - data:      the original data (model/gliner/data/train.json), the new data (the Label Studio export, corrected
                 entity annotations of project 120) or both combined
    That makes 12 trained variants, scored next to the two base models (14 runs) on the same untouched test set, with the
    bootstrap interval of the overall F1 and per domain and per entity type scores.

    uv run python -m src.utils.gliner_utils.gliner_variants --dry-run                 # the plan, nothing is trained
    uv run python -m src.utils.gliner_utils.gliner_variants                           # train what is missing, then score
    uv run python -m src.utils.gliner_utils.gliner_variants --models v25 --structures domain --data original combined
    uv run python -m src.utils.gliner_utils.gliner_variants --no-train                # only score the adapters that exist
    uv run python -m src.utils.gliner_utils.gliner_variants --reference v25_domain_original

    Adapters that already exist are not trained again (--retrain), and the scores of a variant are kept until its adapters
    change (--rescore). Every variant is trained by gliner_train in a process of its own, with its log in <out>/logs, so
    a failing variant does not stop the others. The adapters go to --adapters-dir/<model>_<structure>_<data>, the data
    sets, scores and summary (summary.md, per_label_f1.csv, results.json) to --out.

    The new data is the export without the texts that are in the original data or the test set, so the test set stays
    untouched for every variant, and it has the same cleaning (clean_value) as gliner_convert.
"""

import argparse
import csv
import gc
import glob
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, SCHEMA_PATH, find_adapter
from src.utils.gliner_utils.blocks import load_domains, load_labels
from src.utils.gliner_utils.gliner_convert import convert_export
from src.utils.gliner_utils.gliner_eval import predict_entities, prf, score

logger = logging.getLogger(__name__)

MODELS = {"v2": "fastino/gliner2-multi-v1", "v25": DEFAULT_MODEL}
STRUCTURES = ("domain", "single")
DATA = ("original", "new", "combined")
ORIGINAL_DIR = Path("model/gliner/data")
DEFAULT_ADAPTERS_DIR = Path("model/gliner/variants")
DEFAULT_OUT = Path("output/gliner_variants")
BOOTSTRAP_ROUNDS = 1000
# The lines of gliner_train that are shown while it runs, the rest only goes to the log
SHOWN = re.compile(r"Adapter:|Training the adapter|Epoch \d+/\d+|Validation|New best|Early stopping|Saved |Adapter done|"
                   r"dropped|ERROR|Error|Traceback|OutOfMemory")
BAR = re.compile(r"^(Training|Evaluating):\s+\d+%")


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass(frozen=True)
class Variant:
    model: str
    structure: str
    data: str

    @property
    def name(self) -> str:
        return f"{self.model}_{self.structure}_{self.data}"


def all_labels(schema: Path) -> list[str]:
    return list(load_labels(schema))


def fill_labels(samples: list[dict], labels: list[str]) -> list[dict]:
    """ Every label of the schema as a key (empty where there is none), like the training data of gliner_convert. """
    return [{"text": s["text"], "entities": {label: s["entities"].get(label, []) for label in labels}} for s in samples]


def build_datasets(export: Path | None, original_dir: Path, labels: list[str], out_dir: Path) -> dict[str, int]:
    """ Writes <out_dir>/data/<original|new|combined>/train.json (+ the test.json of the original data) and returns the
        number of samples of each. The new data leaves out the texts of the original data and of the test set. """
    original = fill_labels(json.loads((original_dir / "train.json").read_text(encoding="utf-8")), labels)
    test = json.loads((original_dir / "test.json").read_text(encoding="utf-8"))
    new: list[dict] = []
    if export is not None:
        known = {s["text"] for s in original} | {s["text"] for s in test}
        for sample in fill_labels(convert_export(json.loads(export.read_text(encoding="utf-8")), labels), labels):
            if sample["text"] not in known:
                known.add(sample["text"])
                new.append(sample)
    sets = {"original": original, "new": new, "combined": original + new}
    for name, samples in sets.items():
        folder = out_dir / "data" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "train.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy(original_dir / "test.json", folder / "test.json")
    return {name: len(samples) for name, samples in sets.items()}


def has_adapters(folder: Path, structure: str, domains: list[str]) -> bool:
    try:
        if structure == "single":
            find_adapter(folder)
        else:
            for domain in domains:
                find_adapter(folder / domain)
    except FileNotFoundError:
        return False
    return True


def train(variant: Variant, folder: Path, data_dir: Path, log_file: Path, args: argparse.Namespace) -> bool:
    """ Trains the adapters of a variant with gliner_train in a process of its own. Returns whether it worked. """
    command = [sys.executable, "-m", "src.utils.gliner_utils.gliner_train", "--model", MODELS[variant.model],
               "--data-dir", str(data_dir), "--out", str(folder), "--epochs", str(args.epochs),
               "--val-split", str(args.val_split), "--patience", str(args.patience), "--seed", str(args.seed),
               "--batch-size", str(args.batch_size), "--effective-batch", str(args.effective_batch)]
    if variant.structure == "single":
        command.append("--single")
    return run_logged(command, log_file)


def run_logged(command: list[str], log_file: Path) -> bool:
    """ Runs a command, writes all of its output to log_file and shows the progress: the key lines (epochs, validation
        losses, saved adapters, errors) and a live bar of the training steps. Returns whether it exited with 0. """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started, last_bar = time.time(), 0.0
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        bar_shown = False
        for line in process.stdout:  # universal newlines: tqdm's carriage returns end a line too
            line = line.rstrip()
            log.write(line + "\n")
            if BAR.match(line.strip()):
                if time.time() - last_bar >= 1:
                    last_bar = time.time()
                    print(f"\r    {line.strip()[:110]}  [{hms(time.time() - started)}]", end="", flush=True)
                    bar_shown = True
            elif SHOWN.search(line):
                print(("\r" + " " * 130 + "\r" if bar_shown else "") + "    " + re.sub(r"^\w+ ", "", line)[:200], flush=True)
                bar_shown = False
        print("\r" + " " * 130 + "\r", end="")
    return process.wait() == 0


def evaluate(model_name: str, adapter_dir: Path | None, samples: list[dict], schema: Path) -> dict:
    """ Counts per entity type (and ALL) and per sample of one model on the test samples. """
    from src.utils.gliner_utils.entities import EntityExtractor

    extractor = EntityExtractor(model_name, adapter_dir=adapter_dir, schema_path=schema)
    scope = {label for _, passed in extractor._passes() for label in passed}
    adapters = list(extractor.adapters)
    predicted = predict_entities(samples, extractor)
    del extractor
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    counts, per_sample = score(samples, predicted, scope)
    return {"adapters": adapters, "labels": {label: dict(c) for label, c in counts.items()},
            "per_sample": [[c["tp"], c["fp"], c["fn"]] for c in per_sample]}


def f1_of(per_sample: list[list[int]], indexes) -> float:
    tp = sum(per_sample[i][0] for i in indexes)
    fp = sum(per_sample[i][1] for i in indexes)
    fn = sum(per_sample[i][2] for i in indexes)
    return prf({"tp": tp, "fp": fp, "fn": fn})[2]


def intervals(results: dict[str, dict], reference: str | None) -> dict[str, dict]:
    """ The 95% bootstrap interval of the overall F1 of every run, and its difference with the reference run (the same
        draws for every run, so the difference is paired). """
    size = len(next(iter(results.values()))["per_sample"])
    rng = random.Random(42)
    draws = [[rng.randrange(size) for _ in range(size)] for _ in range(BOOTSTRAP_ROUNDS)]
    lo, hi = int(BOOTSTRAP_ROUNDS * 0.025), int(BOOTSTRAP_ROUNDS * 0.975) - 1
    everything = list(range(size))
    out = {}
    for name, result in results.items():
        ps = result["per_sample"]
        scores = sorted(f1_of(ps, d) for d in draws)
        entry = {"f1": f1_of(ps, everything), "low": scores[lo], "high": scores[hi]}
        if reference in results and name != reference:
            base = results[reference]["per_sample"]
            diffs = sorted(f1_of(ps, d) - f1_of(base, d) for d in draws)
            entry |= {"delta": entry["f1"] - f1_of(base, everything), "delta_low": diffs[lo], "delta_high": diffs[hi],
                      "verdict": "better" if diffs[lo] > 0 else "worse" if diffs[hi] < 0 else "same"}
        out[name] = entry
    return out


def domain_f1(result: dict, labels: list[str]) -> float | None:
    total = {"tp": 0, "fp": 0, "fn": 0}
    for label in labels:
        for key, value in result["labels"].get(label, {}).items():
            total[key] += value
    return prf(total)[2] if any(total.values()) else None


def summarize(results: dict[str, dict], schema: Path, reference: str | None, out_dir: Path) -> str:
    """ Writes summary.md, per_label_f1.csv and results.json to out_dir and returns the summary table. """
    domains = load_domains(schema)
    labels = all_labels(schema)
    ci = intervals(results, reference)
    order = sorted(results, key=lambda name: -ci[name]["f1"])

    header = ["run", "P", "R", "F1", "95% interval", *domains]
    if reference in results:
        header += [f"vs {reference}"]
    rows = []
    for name in order:
        p, r, f = prf(Counter(results[name]["labels"].get("ALL", {})))
        row = [name, f"{p:.2f}", f"{r:.2f}", f"{f:.3f}", f"[{ci[name]['low']:.3f}, {ci[name]['high']:.3f}]"]
        row += [("-" if (v := domain_f1(results[name], labs)) is None else f"{v:.2f}") for labs in domains.values()]
        if reference in results:
            e = ci[name]
            row.append("(reference)" if name == reference else
                       f"{e['delta']:+.3f} [{e['delta_low']:+.3f}, {e['delta_high']:+.3f}] {e['verdict']}")
        rows.append(row)
    table = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + \
            "\n".join("| " + " | ".join(row) + " |" for row in rows)
    n_samples = len(next(iter(results.values()))["per_sample"])
    text = (f"# GLiNER variants, entity extraction on {n_samples} test samples\n\n{table}\n\n"
            "F1 per domain is the micro F1 over its entity types, the interval a bootstrap over the test samples "
            "(1000 draws), the difference with the reference is paired and called better/worse when its interval "
            "excludes 0. A prediction is correct when its (type, text) is in the gold entities of the sample.\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"intervals": ci, "results": results}, indent=1), encoding="utf-8")
    with (out_dir / "per_label_f1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "support", *order])
        for label in [*labels, "ALL"]:
            first = next((r["labels"][label] for r in results.values() if label in r["labels"]), None)
            writer.writerow([label, (first.get("tp", 0) + first.get("fn", 0)) if first else 0,
                             *[f"{prf(Counter(results[n]['labels'].get(label, {})))[2]:.3f}" for n in order]])
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and score every GLiNER variant (model x structure x data).")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS), help="GLiNER2 (v2), GLiNER2.5 (v25)")
    parser.add_argument("--structures", nargs="+", choices=STRUCTURES, default=list(STRUCTURES),
                        help="One adapter per domain, or one for everything")
    parser.add_argument("--data", nargs="+", choices=DATA, default=list(DATA), help="Training data of the variants")
    parser.add_argument("--export", type=Path, default=None,
                        help="Label Studio JSON export with the new data (default: the newest datasets/Gliner/project-120-*.json)")
    parser.add_argument("--original-dir", type=Path, default=ORIGINAL_DIR, help="Folder with the original train.json and test.json")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR, help="Where the adapters of the variants go")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Data sets, logs, scores and summary")
    parser.add_argument("--reference", default=None, help="Run the others are compared with, e.g. v25_domain_original")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan, train and score nothing")
    parser.add_argument("--no-train", action="store_true", help="Do not train, score the adapters that exist")
    parser.add_argument("--retrain", action="store_true", help="Train again where the adapters exist")
    parser.add_argument("--rescore", action="store_true", help="Score again where there are scores of the variant")
    parser.add_argument("--no-base", action="store_true", help="Leave the base models out of the scoring")
    # Passed on to gliner_train, the same for every variant
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    labels, domains = all_labels(args.schema), list(load_domains(args.schema))
    export = args.export
    if export is None and any(data != "original" for data in args.data):
        found = sorted(glob.glob("datasets/Gliner/project-120-*.json"))
        export = Path(found[-1]) if found else None
    if export is None and any(data != "original" for data in args.data):
        build_parser().error("The new data needs the Label Studio export: give --export")
    sizes = build_datasets(export, args.original_dir, labels, args.out)
    logger.info("Samples: %s (export: %s)", sizes, export)

    variants = [Variant(m, s, d) for d in args.data for m in args.models for s in args.structures]
    test = json.loads((args.original_dir / "test.json").read_text(encoding="utf-8"))
    status = {v: has_adapters(args.adapters_dir / v.name, v.structure, domains) for v in variants}

    print(f"\n{'variant':<26}{'train samples':>14}  adapters")
    for v in variants:
        todo = "trained" if status[v] and not args.retrain else "no train" if args.no_train else "to train"
        print(f"{v.name:<26}{sizes[v.data]:>14}  {todo}")
    print(f"+ base models: {'none' if args.no_base else ', '.join(MODELS[m] for m in args.models)}; test samples: {len(test)}")
    if args.dry_run:
        return 0

    # Train, one process each so the GPU memory is free again afterwards
    retrained = set()
    to_train = [v for v in variants if not args.no_train and (args.retrain or not status[v])]
    began, durations = time.time(), []
    for number, v in enumerate(to_train, start=1):
        folder = args.adapters_dir / v.name
        if folder.exists() and args.retrain:
            shutil.rmtree(folder)
        eta = f", about {hms(sum(durations) / len(durations) * (len(to_train) - number + 1))} left" if durations else ""
        print(f"\n[{number}/{len(to_train)}] Training {v.name} on {sizes[v.data]} samples "
              f"(elapsed {hms(time.time() - began)}{eta})", flush=True)
        started = time.time()
        if train(v, folder, args.out / "data" / v.data, args.out / "logs" / f"{v.name}.log", args):
            print(f"    done in {hms(time.time() - started)}", flush=True)
            retrained.add(v.name)
        else:
            print(f"    FAILED, see {args.out / 'logs' / f'{v.name}.log'}", flush=True)
        durations.append(time.time() - started)
        status[v] = has_adapters(folder, v.structure, domains)

    # Score, every run on the same test samples
    runs = {} if args.no_base else {f"{m}_base": (MODELS[m], None) for m in args.models}
    runs |= {v.name: (MODELS[v.model], args.adapters_dir / v.name) for v in variants if status[v]}
    results = {}
    cache = args.out / "results"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"\nScoring {len(runs)} runs on {len(test)} test samples", flush=True)
    for number, (name, (model_name, folder)) in enumerate(runs.items(), start=1):
        cached = cache / f"{name}.json"
        if cached.exists() and not args.rescore and name not in retrained:
            results[name] = json.loads(cached.read_text(encoding="utf-8"))
            print(f"[{number}/{len(runs)}] {name}: scores kept, F1 {f1_of(results[name]['per_sample'], range(len(test))):.3f}", flush=True)
            continue
        print(f"[{number}/{len(runs)}] Scoring {name} ...", end=" ", flush=True)
        started = time.time()
        results[name] = evaluate(model_name, folder, test, args.schema)
        cached.write_text(json.dumps(results[name]), encoding="utf-8")
        print(f"F1 {f1_of(results[name]['per_sample'], range(len(test))):.3f} ({hms(time.time() - started)})", flush=True)
    if not results:
        logger.error("Nothing to score")
        return 1

    print("\n" + summarize(results, args.schema, args.reference, args.out))
    print(f"\nWritten to {args.out}: summary.md, per_label_f1.csv, results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
