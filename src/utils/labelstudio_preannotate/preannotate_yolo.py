"""Run the pipeline's YOLO layout model over scanned pages and import each page into Label Studio
as a task, with every detected region as a pre-annotated rectangle for a human to correct into
ground-truth layout.

The tasks use a labeling config with one image and a RectangleLabels control on it, whose
<Label value="..."/> entries must match the class names of the YOLO model:

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="Text"/>
        <Label value="Section-header"/>
        ...
      </RectangleLabels>
    </View>

Setup, in the .env file in the repo root (see .env.example, never pass the token on the CLI):
    LS_URL=https://your-labelstudio-host      # default: http://localhost:8080
    LS_TOKEN=...                               # Label Studio API token
    LS_YOLO_PROJECT_ID=2                        # which project to add tasks to (LS_PROJECT_ID is the OCR project)

Usage (also run by the pre-annotation dashboard, `uv run preannotate`):
    # Dry run - writes labelstudio_tasks.json for inspection
    uv run python -m src.utils.labelstudio_preannotate.preannotate_yolo --pages /path/to/images

    # Once it looks right, actually import:
    uv run python -m src.utils.labelstudio_preannotate.preannotate_yolo --pages /path/to/images --commit
"""

import argparse
import json
import os
import random

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.processors.record_processor.pipeline import IOManager, RecordManager, ResultProcessor
from src.utils.labelstudio_preannotate.client import LabelStudioClient
from src.utils.labelstudio_preannotate.convert_yolo import regions_to_prediction
from src.utils.labelstudio_preannotate.stage_markers import announce_stage

_LS_URL_DEFAULT = "http://localhost:8080"


@dataclass
class PageDetections:
    path: Path
    width: int
    height: int
    regions: list[dict]


def detect_pages(weights: Path, page_paths: list[Path], max_detections: int, confidence: float) -> list[PageDetections]:
    """ Runs YOLO over PAGE_PATHS until MAX_DETECTIONS regions are found (0 = no limit). The page that
        reaches the limit keeps only its most confident regions, up to the limit. """

    # Imported here so --help and argument errors don't wait for torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    pages = []
    total = 0

    for number, page_path in enumerate(page_paths, start=1):
        image = RecordManager.open_page(page_path)
        if image is None:
            continue

        regions = []
        for result in model.predict(image, conf=confidence, verbose=False, stream=True):
            for mapped_prediction in ResultProcessor.process_result(result):
                regions.append({
                    "bbox": [int(c) for c in mapped_prediction.bbox],
                    "label": mapped_prediction.label,
                    "confidence": float(mapped_prediction.confidence),
                })

        if max_detections:
            regions.sort(key=lambda region: region["confidence"], reverse=True)
            regions = regions[:max_detections - total]

        pages.append(PageDetections(page_path, image.width, image.height, regions))
        total += len(regions)
        counts = ", ".join(f"{label} {count}" for label, count in Counter(region["label"] for region in regions).most_common())
        print(f"[{number}/{len(page_paths)}] {page_path.name}: {len(regions)} region(s), total {total}" + (f" ({counts})" if counts else ""))

        if max_detections and total >= max_detections:
            print(f"Reached the limit of {max_detections} detection(s), stopping.")
            break

    return pages


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="Folder of scanned pages to detect regions on")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--max-pages", type=int, default=20, help="Detect on at most this many pages, 0 = all (default: %(default)s)")
    parser.add_argument("--max-detections", type=int, default=0,
                        help="Stop once this many regions are found over all pages, 0 = no limit (default: %(default)s)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Leave out regions below this confidence (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle the pages with this seed before taking --max-pages (default: sorted order)")
    parser.add_argument("--project-id", type=int, default=os.environ.get("LS_YOLO_PROJECT_ID") or None, help="Label Studio project ID to add tasks to (env: LS_YOLO_PROJECT_ID)")
    parser.add_argument("--ls-url", default=os.environ.get("LS_URL") or _LS_URL_DEFAULT, help="Label Studio base URL (env: LS_URL)")
    parser.add_argument("--ls-token", default=os.environ.get("LS_TOKEN"), help="Label Studio API token (env: LS_TOKEN)")
    parser.add_argument("--data-key", default="image", help="Task data key matching the Image tag's value=\"$...\" in the labeling config")
    parser.add_argument("--from-name", default="label", help="RectangleLabels control name in the labeling config")
    parser.add_argument("--to-name", default="image", help="Image control name in the labeling config")
    parser.add_argument("--model-version", default=None, help="Model version tag on each prediction (default: derived from --weights)")
    parser.add_argument("--tasks-json", type=Path, default=Path("labelstudio_tasks.json"), help="Where to write the generated tasks for inspection")
    parser.add_argument("--commit", action="store_true", help="Actually upload images and import into Label Studio. Default: dry-run, only writes --tasks-json")
    args = parser.parse_args()

    if not args.pages.exists():
        raise SystemExit(f"{args.pages} not found")
    if not args.weights.exists():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    if not 0 <= args.confidence <= 1:
        raise SystemExit(f"--confidence must be between 0 and 1: {args.confidence}")
    if args.commit and not args.project_id:
        raise SystemExit("--project-id (or LS_YOLO_PROJECT_ID) is required to --commit")
    if args.commit and not args.ls_token:
        raise SystemExit("--ls-token (or LS_TOKEN) is required to --commit")

    model_version = args.model_version or f"{args.weights.stem}-preanno"

    page_paths = IOManager.collect_image_files(args.pages)
    if not page_paths:
        raise SystemExit(f"No page images found under {args.pages}")
    if args.seed is not None:
        random.Random(args.seed).shuffle(page_paths)
    if args.max_pages:
        page_paths = page_paths[:args.max_pages]

    announce_stage("detect", "start")
    print(f"Detecting layout on {len(page_paths)} page(s) with {args.weights} (confidence >= {args.confidence})...")
    pages = detect_pages(args.weights, page_paths, args.max_detections, args.confidence)
    if not pages:
        raise SystemExit(f"None of the pages under {args.pages} could be opened")
    total = Counter(region["label"] for page in pages for region in page.regions)
    print(f"Detected {total.total()} region(s) on {len(pages)} page(s): " + ", ".join(f"{label} {count}" for label, count in total.most_common()))
    announce_stage("detect", "done")

    page_paths = [page.path for page in pages]
    if args.commit:
        announce_stage("upload", "start")
        client = LabelStudioClient(args.ls_url, args.ls_token, args.project_id)
        print(f"Uploading {len(page_paths)} image(s) to Label Studio project {args.project_id}...")
        server_paths = client.upload_images(page_paths)
        announce_stage("upload", "done")
    else:
        client = None
        server_paths = {p.name: f"<not uploaded — dry run>/{p.name}" for p in page_paths}

    tasks = []
    missing = []
    for page in pages:
        if page.path.name not in server_paths:
            missing.append(page.path.name)
            continue
        prediction = regions_to_prediction(page.regions, page.width, page.height, model_version, args.from_name, args.to_name)
        tasks.append({"data": {args.data_key: server_paths[page.path.name]}, "predictions": [prediction]})

    args.tasks_json.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tasks)} task(s) written to {args.tasks_json}")
    if missing:
        print(f"WARNING: {len(missing)} image(s) failed to upload and were skipped: {missing[:10]}")

    if args.commit:
        announce_stage("import", "start")
        result = client.import_tasks(tasks)
        print(f"Imported into Label Studio project {args.project_id}: {result}")
        announce_stage("import", "done")
        print("Pre-annotation complete.")
    else:
        print(f"Dry run complete — inspect {args.tasks_json}, then SEND (or re-run with --commit) to import.")


if __name__ == "__main__":
    main()
