"""
Inter-Annotator Agreement (IAA) for NER annotations exported from Label Studio.

Usage:
    python iaa_ner.py <export.json> [options]

Options:
    --label <name>      Filter stats to a specific label type
    --show-full         Print tasks with full agreement
    --show-issues       Print tasks with disagreements and what differs
    --annotators A B    Only compare two specific annotator IDs
"""

import json
import argparse
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        record.setdefault("label", [])
    return data


def group_by_task(data: list[dict]) -> dict[int, list[dict]]:
    """Group annotation records by task id."""
    tasks: dict[int, list[dict]] = defaultdict(list)
    for record in data:
        tasks[record["id"]].append(record)
    return dict(tasks)


def get_multi_annotated_tasks(
    data: list[dict],
    annotator_pair: Optional[tuple[int, int]] = None,
) -> dict[int, list[dict]]:
    """Return only tasks that have annotations from at least 2 annotators.

    If annotator_pair is given, only return tasks annotated by both of those
    specific annotators.
    """
    tasks = group_by_task(data)
    result = {}
    for task_id, records in tasks.items():
        if len(records) < 2:
            continue
        if annotator_pair:
            ids = {r["annotator"] for r in records}
            if not all(a in ids for a in annotator_pair):
                continue
            # Keep only the two requested annotators
            records = [r for r in records if r["annotator"] in annotator_pair]
        result[task_id] = records
    return result


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def spans_from_record(record: dict, label_filter: Optional[str] = None,
                      ignore_labels: Optional[set[str]] = None) -> set[tuple]:
    """Return a set of (start, end, label) tuples from a record's label list."""
    result = set()
    for ann in record.get("label", []):
        for lbl in ann["labels"]:
            if label_filter is not None and lbl != label_filter:
                continue
            if ignore_labels and lbl in ignore_labels:
                continue
            result.add((ann["start"], ann["end"], lbl))
    return result


# ---------------------------------------------------------------------------
# Span-level F1 between two annotation sets
# ---------------------------------------------------------------------------

def span_f1(spans_a: set[tuple], spans_b: set[tuple]) -> dict:
    tp = len(spans_a & spans_b)
    fp = len(spans_b - spans_a)   # in b but not a
    fn = len(spans_a - spans_b)   # in a but not b

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Cohen's Kappa at character level
# ---------------------------------------------------------------------------

def char_label_map(record: dict, text_len: int,
                   label_filter: Optional[str] = None,
                   ignore_labels: Optional[set[str]] = None) -> list[str]:
    """Assign a label (or 'O') to every character position."""
    labels = ["O"] * text_len
    for ann in record.get("label", []):
        for lbl in ann["labels"]:
            if label_filter is not None and lbl != label_filter:
                continue
            if ignore_labels and lbl in ignore_labels:
                continue
            for i in range(ann["start"], ann["end"]):
                if i < text_len:
                    labels[i] = lbl
    return labels


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two flat label sequences."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 1.0

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    all_labels = set(labels_a) | set(labels_b)
    p_e = 0.0
    for lbl in all_labels:
        p_a = labels_a.count(lbl) / n
        p_b = labels_b.count(lbl) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Agreement classification per task
# ---------------------------------------------------------------------------

def classify_task(records: list[dict],
                  label_filter: Optional[str] = None,
                  ignore_labels: Optional[set[str]] = None) -> str:
    """Return 'full', 'partial', or 'none' agreement for a task."""
    all_spans = [spans_from_record(r, label_filter, ignore_labels) for r in records]
    reference = all_spans[0]
    if all(s == reference for s in all_spans[1:]):
        return "full"
    if any(len(s & reference) > 0 for s in all_spans[1:]):
        return "partial"
    return "none"


# ---------------------------------------------------------------------------
# Main IAA calculation
# ---------------------------------------------------------------------------

def calculate_iaa(data: list[dict],
                  label_filter: Optional[str] = None,
                  annotator_pair: Optional[tuple[int, int]] = None,
                  ignore_labels: Optional[set[str]] = None) -> dict:
    """
    Calculate inter-annotator agreement metrics across all multi-annotated tasks.

    Returns a dict with:
        - total_tasks: total tasks in the export
        - multi_annotated: tasks with >= 2 annotators
        - full_agreement: count of tasks with identical span sets
        - partial_agreement: count with some overlap
        - no_agreement: count with no overlap at all
        - macro_f1: average span-F1 across tasks (treating ann[0] as ref)
        - macro_precision / macro_recall
        - cohen_kappa: average character-level kappa across tasks
        - per_label: per-label breakdown (only for exact spans)
        - annotators: list of annotator IDs found
    """
    tasks = group_by_task(data)
    multi = get_multi_annotated_tasks(data, annotator_pair)

    full = partial = none = 0
    f1_scores: list[float] = []
    prec_scores: list[float] = []
    rec_scores: list[float] = []
    kappa_scores: list[float] = []

    per_label_tp: dict[str, int] = defaultdict(int)
    per_label_fp: dict[str, int] = defaultdict(int)
    per_label_fn: dict[str, int] = defaultdict(int)

    for task_id, records in multi.items():
        # Use first two records for pairwise metrics
        r1, r2 = records[0], records[1]
        text_len = len(r1["text"])

        spans1 = spans_from_record(r1, label_filter, ignore_labels)
        spans2 = spans_from_record(r2, label_filter, ignore_labels)

        metrics = span_f1(spans1, spans2)
        f1_scores.append(metrics["f1"])
        prec_scores.append(metrics["precision"])
        rec_scores.append(metrics["recall"])

        # Per-label counts (unfiltered)
        if label_filter is None:
            all_labels_in_task = {lbl for s in [spans1, spans2]
                                  for _, _, lbl in s}
            for lbl in all_labels_in_task:
                s1 = {s for s in spans1 if s[2] == lbl}
                s2 = {s for s in spans2 if s[2] == lbl}
                m = span_f1(s1, s2)
                per_label_tp[lbl] += m["tp"]
                per_label_fp[lbl] += m["fp"]
                per_label_fn[lbl] += m["fn"]

        # Cohen's kappa (character level)
        la = char_label_map(r1, text_len, label_filter, ignore_labels)
        lb = char_label_map(r2, text_len, label_filter, ignore_labels)
        kappa_scores.append(cohen_kappa(la, lb))

        cls = classify_task(records, label_filter, ignore_labels)
        if cls == "full":
            full += 1
        elif cls == "partial":
            partial += 1
        else:
            none += 1

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Per-label F1
    per_label: dict[str, dict] = {}
    for lbl in per_label_tp.keys() | per_label_fp.keys() | per_label_fn.keys():
        tp = per_label_tp[lbl]
        fp = per_label_fp[lbl]
        fn = per_label_fn[lbl]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_label[lbl] = {"precision": p, "recall": r, "f1": f,
                          "tp": tp, "fp": fp, "fn": fn}

    return {
        "total_tasks": len(tasks),
        "multi_annotated": len(multi),
        "full_agreement": full,
        "partial_agreement": partial,
        "no_agreement": none,
        "macro_f1": safe_mean(f1_scores),
        "macro_precision": safe_mean(prec_scores),
        "macro_recall": safe_mean(rec_scores),
        "cohen_kappa": safe_mean(kappa_scores),
        "per_label": per_label,
        "annotators": sorted({r["annotator"] for r in data}),
    }


# ---------------------------------------------------------------------------
# Task inspection helpers
# ---------------------------------------------------------------------------

def get_full_agreement_tasks(data: list[dict],
                              label_filter: Optional[str] = None,
                              annotator_pair: Optional[tuple[int, int]] = None,
                              ignore_labels: Optional[set[str]] = None,
                              ) -> list[dict]:
    """Return tasks where all annotators are in complete agreement."""
    multi = get_multi_annotated_tasks(data, annotator_pair)
    result = []
    for task_id, records in multi.items():
        if classify_task(records, label_filter, ignore_labels) == "full":
            result.append({
                "task_id": task_id,
                "annotators": [r["annotator"] for r in records],
                "spans": sorted(spans_from_record(records[0], label_filter, ignore_labels)),
            })
    return result


def get_disagreement_tasks(data: list[dict],
                            label_filter: Optional[str] = None,
                            annotator_pair: Optional[tuple[int, int]] = None,
                            ignore_labels: Optional[set[str]] = None,
                            ) -> list[dict]:
    """
    Return tasks where annotators disagree, with a diff of what's different.

    Each entry contains:
        task_id, text, annotators, only_in_<annotator_id>, shared_spans
    """
    multi = get_multi_annotated_tasks(data, annotator_pair)
    result = []
    for task_id, records in multi.items():
        cls = classify_task(records, label_filter, ignore_labels)
        if cls == "full":
            continue

        entry: dict = {
            "task_id": task_id,
            "agreement_level": cls,
            "text_preview": records[0]["text"][:120].replace("\n", " "),
            "annotators": {},
        }

        all_spans = [spans_from_record(r, label_filter, ignore_labels) for r in records]
        shared = all_spans[0].copy()
        for s in all_spans[1:]:
            shared &= s

        entry["shared_spans"] = sorted(shared)

        for record, spans in zip(records, all_spans):
            ann_id = record["annotator"]
            unique = sorted(spans - shared)
            entry["annotators"][ann_id] = {
                "unique_spans": unique,
                "total_spans": len(spans),
            }

        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_iaa_report(stats: dict, label_filter: Optional[str] = None,
                     ignore_labels: Optional[set[str]] = None) -> None:
    print("\n" + "=" * 60)
    print("  Inter-Annotator Agreement Report")
    if label_filter:
        print(f"  Label filter: {label_filter}")
    if ignore_labels:
        print(f"  Ignored labels: {sorted(ignore_labels)}")
    print("=" * 60)
    print(f"  Annotators found         : {stats['annotators']}")
    print(f"  Total tasks              : {stats['total_tasks']}")
    print(f"  Multi-annotated tasks    : {stats['multi_annotated']}")
    print()
    total_multi = stats["multi_annotated"] or 1
    print(f"  Full agreement           : {stats['full_agreement']:4d}  "
          f"({stats['full_agreement']/total_multi*100:.1f}%)")
    print(f"  Partial agreement        : {stats['partial_agreement']:4d}  "
          f"({stats['partial_agreement']/total_multi*100:.1f}%)")
    print(f"  No agreement             : {stats['no_agreement']:4d}  "
          f"({stats['no_agreement']/total_multi*100:.1f}%)")
    print()
    print(f"  Macro Precision          : {stats['macro_precision']:.4f}")
    print(f"  Macro Recall             : {stats['macro_recall']:.4f}")
    print(f"  Macro F1                 : {stats['macro_f1']:.4f}")
    print(f"  Cohen's Kappa (char)     : {stats['cohen_kappa']:.4f}")

    if stats["per_label"]:
        print()
        print("  Per-label F1 (over multi-annotated tasks):")
        header = f"  {'Label':<28} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}"
        print(header)
        print("  " + "-" * 58)
        for lbl, m in sorted(stats["per_label"].items(),
                              key=lambda x: -x[1]["f1"]):
            print(f"  {lbl:<28} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
    print("=" * 60 + "\n")


def print_full_agreement_tasks(tasks: list[dict], minimal: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Tasks with FULL agreement ({len(tasks)})")
    print(f"{'='*60}")
    for t in tasks:
        if minimal:
            print(t["task_id"])
        else:
            print(f"  Task {t['task_id']}  annotators={t['annotators']}  "
                  f"spans={len(t['spans'])}")
    print()


def print_disagreement_tasks(tasks: list[dict], minimal: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Tasks with DISAGREEMENT ({len(tasks)})")
    print(f"{'='*60}")
    for t in tasks:
        if minimal:
            print(t["task_id"])
            continue
        print(f"\n  Task {t['task_id']}  [{t['agreement_level']} agreement]")
        print(f"  Text: \"{t['text_preview']}\"")
        print(f"  Shared spans ({len(t['shared_spans'])}):")
        for s in t["shared_spans"]:
            print(f"    {s[2]:<26} [{s[0]:>4}:{s[1]:>4}]")
        for ann_id, info in t["annotators"].items():
            unique = info["unique_spans"]
            print(f"  Annotator {ann_id}  (total={info['total_spans']}, "
                  f"unique={len(unique)}):")
            for s in unique:
                print(f"    + {s[2]:<24} [{s[0]:>4}:{s[1]:>4}]  \"{s[2]}\"")
    print()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_issues_to_labelstudio(data: list[dict], output_path: str,
                                  label_filter: Optional[str] = None,
                                  annotator_pair: Optional[tuple[int, int]] = None,
                                  ignore_labels: Optional[set[str]] = None) -> int:
    """
    Write a new Label Studio-compatible JSON file containing only the records
    that belong to disagreement tasks. All original fields are preserved exactly.

    Returns the number of records written.
    """
    issue_task_ids = {
        t["task_id"]
        for t in get_disagreement_tasks(data, label_filter, annotator_pair, ignore_labels)
    }
    seen = set()
    records = []
    for r in data:
        if r["id"] in issue_task_ids and r["id"] not in seen:
            seen.add(r["id"])
            records.append(r)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inter-Annotator Agreement for Label Studio NER exports"
    )
    parser.add_argument("filepath", help="Path to Label Studio JSON export")
    parser.add_argument("--label", default=None,
                        help="Filter to a specific label type")
    parser.add_argument("--show-full", action="store_true",
                        help="Print tasks with full agreement")
    parser.add_argument("--show-issues", action="store_true",
                        help="Print tasks with disagreements and diffs")
    parser.add_argument("--annotators", nargs=2, type=int, metavar=("A", "B"),
                        help="Only compare two specific annotator IDs")
    parser.add_argument("--minimal", action="store_true",
                        help="Only print task IDs (use with --show-full or --show-issues)")
    parser.add_argument("--ignore-label", dest="ignore_labels", action="append",
                        metavar="LABEL", default=[],
                        help="Exclude a label type from all metrics (repeatable)")
    parser.add_argument("--export-issues", metavar="OUTPUT_FILE",
                        help="Write disagreement tasks to a Label Studio-compatible JSON file")
    parser.add_argument("--agreed-ids", action="store_true",
                        help="Print only the task IDs with full agreement")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.filepath)
    pair = tuple(args.annotators) if args.annotators else None
    ignore = set(args.ignore_labels) if args.ignore_labels else None

    stats = calculate_iaa(data, label_filter=args.label, annotator_pair=pair,
                          ignore_labels=ignore)
    print_iaa_report(stats, label_filter=args.label, ignore_labels=ignore)

    if args.agreed_ids:
        tasks = get_full_agreement_tasks(data, label_filter=args.label,
                                         annotator_pair=pair, ignore_labels=ignore)
        for t in tasks:
            print(t["task_id"])

    if args.show_full:
        tasks = get_full_agreement_tasks(data, label_filter=args.label,
                                         annotator_pair=pair, ignore_labels=ignore)
        print_full_agreement_tasks(tasks, minimal=args.minimal)

    if args.show_issues:
        tasks = get_disagreement_tasks(data, label_filter=args.label,
                                        annotator_pair=pair, ignore_labels=ignore)
        print_disagreement_tasks(tasks, minimal=args.minimal)

    if args.export_issues:
        n = export_issues_to_labelstudio(
            data, args.export_issues,
            label_filter=args.label,
            annotator_pair=pair,
            ignore_labels=ignore,
        )
        print(f"Exported {n} records ({n // 2 if n % 2 == 0 else n} tasks) "
              f"to {args.export_issues}")


if __name__ == "__main__":
    main()
