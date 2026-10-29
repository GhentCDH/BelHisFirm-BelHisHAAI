"""Relabels one class of a YOLO dataset as another, e.g. the List-item boxes of DocLayNet as Text, so a model
trained on it starts out with the same classes the fine-tune datasets use (see prepare_dataset.py).

The class indices stay the same (the merged class is just never used), so the model keeps its 11 outputs and
the weights stay compatible with the rest of the pipeline. The label files are rewritten in place, and the
ultralytics label caches next to them are deleted: a relabel doesn't change a file's size, so the cache
would otherwise be reused with the old classes. Running it twice is harmless.

Usage:
    # See what would change:
    uv run python -m src.utils.yolo_training.merge_classes datasets/DocLayNet/YOLO --dry-run

    # List-item -> Text on train, val and test:
    uv run python -m src.utils.yolo_training.merge_classes datasets/DocLayNet/YOLO
"""

import argparse

from collections import Counter
from pathlib import Path

_CLASS_NAMES = ("Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture",
                "Section-header", "Table", "Text", "Title")


def merge_labels(label_file: Path, source: int, target: int, write: bool) -> Counter:
    """ Counts the boxes of LABEL_FILE per class before the merge, and rewrites SOURCE as TARGET if WRITE. """

    counts = Counter()
    lines = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        counts[int(parts[0])] += 1
        if int(parts[0]) == source:
            parts[0] = str(target)
        lines.append(" ".join(parts))
    if write and counts[source]:
        label_file.write_text("\n".join(lines) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="YOLO dataset folder with labels/<split>/*.txt, e.g. datasets/DocLayNet/YOLO")
    parser.add_argument("--source", default="List-item", choices=_CLASS_NAMES, help="Class to relabel (default: %(default)s)")
    parser.add_argument("--target", default="Text", choices=_CLASS_NAMES, help="Class it becomes (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Only count, don't change any file")
    args = parser.parse_args()

    labels_dir = args.dataset / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"{labels_dir} not found")
    if args.source == args.target:
        raise SystemExit("--source and --target are the same class")
    source, target = _CLASS_NAMES.index(args.source), _CLASS_NAMES.index(args.target)

    for split_dir in sorted(p for p in labels_dir.iterdir() if p.is_dir()):
        counts = Counter()
        files = sorted(split_dir.glob("*.txt"))
        for label_file in files:
            counts += merge_labels(label_file, source, target, write=not args.dry_run)
        print(f"{split_dir.name}: {len(files)} file(s), {counts[source]} {args.source} -> {args.target}"
              f" ({args.target}: {counts[target]} -> {counts[target] + counts[source]})")

    if not args.dry_run:
        for cache in labels_dir.glob("*.cache"):
            cache.unlink()
            print(f"Deleted stale {cache}")


if __name__ == "__main__":
    main()
