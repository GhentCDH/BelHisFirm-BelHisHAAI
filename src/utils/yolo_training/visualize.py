"""Test one or more YOLO layout models on a random sample of scanned pages and save every page with its
detected regions drawn on it, to check the models by eye. Detection goes through the pipeline's own
VisionAnalyzer.detect_layout, so the boxes are exactly the ones the pipeline keeps, with its confidence
thresholds (see ConfigParameter.yolo_confidence and yolo_label_confidence).

Every page is also checked for missed text with src/checks/uncovered_ink.py: ink that lies outside every detected
box is outlined in red on the saved image (whether or not the page counts as flagged), with its share of the page.

Output (output/ is gitignored, so the images stay local):
    output/yolo_tests/<name>/<model>/<page>.jpg   one folder per --weights
    output/yolo_tests/<name>/summary.txt          the sampled pages, the models and the region counts and the uncovered ink per page

Usage:
    # The fine-tuned model next to the current one, on 20 random pages:
    uv run python -m src.utils.yolo_training.visualize --pages /path/to/TIF/1880 \\
        --weights runs/finetunenolist28102026/weights/best.pt --weights model/best.pt
"""

import argparse
import random

from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from src.checks.uncovered_ink import Config as InkConfig, analyse, is_flagged
from src.processors.record_processor.pipeline import IOManager, RecordManager

RED = (255, 0, 0)  # the saved arrays are RGB


def model_name(weights: Path) -> str:
    """ The run name for runs/<name>/weights/best.pt, otherwise the file's stem (e.g. 'best' for model/best.pt). """

    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="Folder of scanned pages to sample from")
    parser.add_argument("--weights", type=Path, action="append", help="YOLO weights to test, repeatable (default: model/best.pt)")
    parser.add_argument("--sample", type=int, default=20, help="Number of random pages to test, 0 = all (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for picking the pages, the same seed gives the same pages (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=Path("output/yolo_tests"), help="Directory to save the tests in (default: %(default)s)")
    parser.add_argument("--name", default=None, help="Name of this test under --output (default: <pages folder>_<timestamp>)")
    args = parser.parse_args()

    weights = args.weights or [Path("model/best.pt")]
    for path in weights:
        if not path.exists():
            raise SystemExit(f"YOLO weights not found: {path}")
    names = [model_name(path) for path in weights]
    if len(set(names)) != len(names):
        raise SystemExit(f"--weights must have different run names to get their own folder: {names}")

    page_paths = IOManager.collect_image_files(args.pages)
    if not page_paths:
        raise SystemExit(f"No page images found under {args.pages}")
    if args.sample and args.sample < len(page_paths):
        page_paths = sorted(random.Random(args.seed).sample(page_paths, args.sample))

    test_dir = args.output / (args.name or f"{args.pages.name}_{datetime.now():%Y%m%d_%H%M%S}")
    if test_dir.exists():
        raise SystemExit(f"{test_dir} already exists, pick a new --name")

    # Imported here so --help and argument errors don't wait for torch
    from ultralytics.utils.plotting import Annotator, colors

    from src.processors.record_processor.data import ConfigParameter
    from src.processors.record_processor.pipeline.vision_analyzer import VisionAnalyzer

    config = ConfigParameter(50, 0.85, 0.4, 200, 300, True, set(), set())
    summary = [f"Pages: {args.pages} ({len(page_paths)} sampled, seed {args.seed})",
               f"Thresholds: {config.yolo_confidence}, per label {config.yolo_label_confidence}", ""]

    for name, path in zip(names, weights):
        analyzer = VisionAnalyzer(config, str(path), None)
        class_index = {label: index for index, label in analyzer.yolo_model.names.items()}
        (test_dir / name).mkdir(parents=True)
        totals = Counter()
        flags = 0
        summary.append(f"{name} ({path})")

        for number, page_path in enumerate(page_paths, start=1):
            image = RecordManager.open_page(page_path)
            if image is None:
                continue
            regions = analyzer.detect_layout(image)

            # Drawn with OpenCV on an array, the PIL drawing sizes its labels to the image and covers half the page
            annotator = Annotator(np.array(image), line_width=3, pil=False)
            for region in regions:
                annotator.box_label(region["bbox"], f"{region['label']} {region['confidence']:.2f}", color=colors(class_index[region["label"]]))
            drawn = annotator.result()

            # Ink outside every detected box, always in red
            ink_config = InkConfig()
            scale = ink_config.work_width / image.width
            boxes = np.array([region["bbox"] for region in regions], dtype=float).reshape(-1, 4)
            ink = analyse(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), boxes, scale, ink_config)
            for x, y, w, h in (component["bbox"] for component in ink["comps"]):
                x1, y1, x2, y2 = (int(value / scale) for value in (x, y, x + w, y + h))
                cv2.rectangle(drawn, (x1, y1), (x2, y2), RED, 4)
            for component in ink["comps"]:
                x, y = component["bbox"][:2]
                cv2.putText(drawn, f"missed {component['frac'] * 100:.2f}% ~{component['lines']} lines",
                            (int(x / scale), max(int(y / scale) - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
            flagged = is_flagged(ink["total"], ink["top"], ink_config)
            Image.fromarray(drawn).save(test_dir / name / f"{page_path.stem}.jpg", quality=85)

            counts = Counter(region["label"] for region in regions)
            totals += counts
            line = f"{page_path.name}: " + ", ".join(f"{label} {count}" for label, count in counts.most_common())
            line += f" | uncovered ink {ink['total'] * 100:.2f}% in {len(ink['comps'])} areas" + (" FLAGGED" if flagged else "")
            flags += flagged
            summary.append(f"  {line}")
            print(f"[{name} {number}/{len(page_paths)}] {line}")

        summary += ["  TOTAL: " + ", ".join(f"{label} {count}" for label, count in totals.most_common()),
                    f"  FLAGGED for missed text: {flags}/{len(page_paths)}", ""]

    (test_dir / "summary.txt").write_text("\n".join(summary))
    print(f"Saved to {test_dir}")


if __name__ == "__main__":
    main()
