"""Sample text regions (record headers, body text, captions, page headers, ...) from scanned
pages, get an initial GLM-OCR guess for each, and import them into Label Studio as pre-annotated
tasks for a human to correct into ground-truth transcriptions.

The counterpart of HisTableFinder's labelstudio_preannotate/preannotate_ocr.py, which does the
same for table crops. client.py and convert_ocr.py are copied from there unchanged, and the tasks
use the same labeling config, so one Label Studio project and one export script
(HisTableFinder's export_ocr_captions.py) serve both:

    <View>
      <Image name="image" value="$captioning"/>
      <Header value="Expected OCR:"/>
      <TextArea name="caption" toName="image" .../>
    </View>

The regions come from the pipeline's own YOLO layout model and are cropped with the padding the
pipeline uses for that kind of region, so a model fine-tuned on the corrected crops sees the same
input it gets in the pipeline. The layout label is part of each crop's filename. "Record-header"
is not a layout label: it is a Section-header or Title whose text starts with a record number,
kept apart so the headers that split the records get their own share of the sample.

Setup, in the .env file in the repo root (see .env.example, never pass the token on the CLI):
    LS_URL=https://your-labelstudio-host      # default: http://localhost:8080
    LS_TOKEN=...                               # Label Studio API token
    LS_PROJECT_ID=1                             # which project to add tasks to

Usage:
    # Dry run - writes labelstudio_tasks.json and the sampled crops for inspection
    uv run python -m src.utils.labelstudio_preannotate.preannotate_regions --pages /path/to/images

    # Once it looks right, actually import:
    uv run python -m src.utils.labelstudio_preannotate.preannotate_regions --pages /path/to/images --commit
"""

import argparse
import json
import os
import random
import re

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.processors.record_processor import OCRProcessor
from src.processors.record_processor.pipeline import IOManager, VisionAnalyzer
from src.processors.record_processor.record_processor import RecordProcessor
from src.utils.labelstudio_preannotate.client import LabelStudioClient
from src.utils.labelstudio_preannotate.convert_ocr import text_to_prediction

_LS_URL_DEFAULT = "http://localhost:8080"

RECORD_HEADER_LABEL = "Record-header"
# Looser than the pipeline's header check, which also wants a dash: the headers it misses belong here too
RECORD_NUMBER_PATTERN = re.compile(r"\W*\d{3,6}\b")

# The labels the layout model finds on these pages, tables have their own labeling rounds in HisTableFinder
DEFAULT_LABELS = [RECORD_HEADER_LABEL, "Section-header", "Text", "Caption", "Page-header", "Page-footer"]


@dataclass
class RegionCrop:
    label: str
    path: Path
    guess: str


def crop_filename(page_stem: str, label: str, box: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    return f"{page_stem}_{label.lower()}_x{x0}_y{y0}_w{x1 - x0}_h{y1 - y0}.png"


def sample_regions(processor: RecordProcessor, page_paths: list[Path], labels: list[str], per_label: int,
                   per_page: int, crops_dir: Path, excluded_names: list[str], rng: random.Random) -> list[RegionCrop]:
    """ Goes over PAGE_PATHS until every label has PER_LABEL crops, taking at most PER_PAGE regions
        of a label from one page so the sample is spread over the pages. A page is 50+ MB in memory,
        so its crops are read and saved before the next page is opened.
    """

    manager = processor.record_manager
    header_labels = processor.config.header_candidate_labels
    counts = Counter()
    crops = []

    def wanted(label: str, taken_on_page: Counter) -> bool:
        return label in labels and counts[label] < per_label and taken_on_page[label] < per_page

    for number, page_path in enumerate(page_paths, start=1):
        if all(counts[label] >= per_label for label in labels):
            break

        image = manager.open_page(page_path)
        if image is None:
            continue

        regions = manager.vision_analyzer.detect_layout(image)
        rng.shuffle(regions)
        taken_on_page = Counter()

        for region in regions:
            label = region["label"]
            is_header = label in header_labels
            # Only its text tells whether a header is a record header
            if not (wanted(label, taken_on_page) or (is_header and wanted(RECORD_HEADER_LABEL, taken_on_page))):
                continue

            # Same crop and prompt as the pipeline uses for this kind of region
            padding = processor.config.padding if is_header else OCRProcessor.BLOCK_PADDING
            prompt = VisionAnalyzer.HEADER_OCR_PROMPT if is_header else OCRProcessor.BLOCK_OCR_PROMPT

            x0, y0, x1, y1 = region["bbox"]
            box = (max(0, x0 - padding), max(0, y0 - padding), min(image.width, x1 + padding), min(image.height, y1 + padding))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            crop = image.crop(box)
            try:
                guess = manager.glm_ocr.ocr(crop, prompt=prompt)
            except Exception as e:
                print(f"FAILED to get an OCR guess for {label} {list(box)} on {page_path.name}: {type(e).__name__}: {e}")
                continue

            if is_header and RECORD_NUMBER_PATTERN.match(guess):
                label = RECORD_HEADER_LABEL
            if not wanted(label, taken_on_page):
                continue

            name = crop_filename(page_path.stem, label, box)
            # Label Studio puts its own prefix in front of an uploaded filename, so match by suffix
            if any(existing.endswith(name) for existing in excluded_names):
                continue

            crop.save(crops_dir / name)
            crops.append(RegionCrop(label, crops_dir / name, guess))
            counts[label] += 1
            taken_on_page[label] += 1
            print(f"--- {name} ---\n{guess}")

        print(f"[{number}/{len(page_paths)}] {page_path.name}: " + ", ".join(f"{label} {counts[label]}/{per_label}" for label in labels))

    return crops


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="Folder of scanned pages to sample regions from")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Layout labels to sample (default: %(default)s)")
    parser.add_argument("--per-label", type=int, default=20, help="How many crops to sample of each label (default: %(default)s)")
    parser.add_argument("--per-page", type=int, default=2, help="At most this many crops of one label from one page (default: %(default)s)")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Stop after this many pages, also when a label is still short of crops (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the sample")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--glm-checkpoint", type=Path, default=None,
                        help="GLM-OCR checkpoint for the OCR guess (default: the base GLM-OCR model)")
    parser.add_argument("--crops-dir", type=Path, default=Path("ocr_preanno_crops"), help="Local directory to save the sampled crops to")
    parser.add_argument("--exclude-dir", type=Path,
                        help="Directory of crops from a previous round, matched by filename suffix so they are never sampled again")
    parser.add_argument("--project-id", type=int, default=os.environ.get("LS_PROJECT_ID") or None, help="Label Studio project ID to add tasks to (env: LS_PROJECT_ID)")
    parser.add_argument("--ls-url", default=os.environ.get("LS_URL") or _LS_URL_DEFAULT, help="Label Studio base URL (env: LS_URL)")
    parser.add_argument("--ls-token", default=os.environ.get("LS_TOKEN"), help="Label Studio API token (env: LS_TOKEN)")
    parser.add_argument("--data-key", default="captioning", help="Task data key matching the Image tag's value=\"$...\" in the labeling config")
    parser.add_argument("--from-name", default="caption", help="TextArea control name in the labeling config")
    parser.add_argument("--to-name", default="image", help="Image control name in the labeling config")
    parser.add_argument("--model-version", default=None, help="Model version tag on each prediction (default: derived from --glm-checkpoint)")
    parser.add_argument("--tasks-json", type=Path, default=Path("labelstudio_tasks.json"), help="Where to write the generated tasks for inspection")
    parser.add_argument("--commit", action="store_true", help="Actually upload images and import into Label Studio. Default: dry-run, only writes --tasks-json")
    args = parser.parse_args()

    if not args.pages.exists():
        raise SystemExit(f"{args.pages} not found")
    if not args.weights.exists():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    if args.glm_checkpoint is not None and not args.glm_checkpoint.exists():
        raise SystemExit(f"GLM-OCR checkpoint not found: {args.glm_checkpoint}")
    if args.exclude_dir and not args.exclude_dir.exists():
        raise SystemExit(f"{args.exclude_dir} not found")
    if args.commit and not args.project_id:
        raise SystemExit("--project-id (or LS_PROJECT_ID) is required to --commit")
    if args.commit and not args.ls_token:
        raise SystemExit("--ls-token (or LS_TOKEN) is required to --commit")

    model_version = args.model_version or f"{args.glm_checkpoint.name if args.glm_checkpoint else 'glm-ocr-base'}-preanno"
    excluded_names = [p.name for p in args.exclude_dir.glob("*.png")] if args.exclude_dir else []

    page_paths = IOManager.collect_image_files(args.pages)
    if not page_paths:
        raise SystemExit(f"No page images found under {args.pages}")
    rng = random.Random(args.seed)
    rng.shuffle(page_paths)
    page_paths = page_paths[:args.max_pages]

    processor = RecordProcessor(skip_ocr=True, skip_tables=True, yolo_model_path=str(args.weights),
                                glm_checkpoint=str(args.glm_checkpoint) if args.glm_checkpoint else None)

    args.crops_dir.mkdir(parents=True, exist_ok=True)
    crops = sample_regions(processor, page_paths, args.labels, args.per_label, args.per_page,
                           args.crops_dir, excluded_names, rng)
    if not crops:
        raise SystemExit(f"No {args.labels} regions found under {args.pages}")
    print(f"Sampled {len(crops)} crop(s): " + ", ".join(f"{label} {count}" for label, count in Counter(crop.label for crop in crops).items()))

    guesses = {crop.path: crop.guess for crop in crops}
    crop_paths = [crop.path for crop in crops]

    if args.commit:
        client = LabelStudioClient(args.ls_url, args.ls_token, args.project_id)
        print(f"Uploading {len(crop_paths)} image(s) to Label Studio project {args.project_id}...")
        server_paths = client.upload_images(crop_paths)
    else:
        client = None
        server_paths = {p.name: f"<not uploaded — dry run>/{p.name}" for p in crop_paths}

    tasks = []
    missing = []
    for path in crop_paths:
        if path.name not in server_paths:
            missing.append(path.name)
            continue
        prediction = text_to_prediction(guesses[path], model_version, args.from_name, args.to_name)
        tasks.append({"data": {args.data_key: server_paths[path.name]}, "predictions": [prediction]})

    args.tasks_json.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tasks)} task(s) written to {args.tasks_json}")
    if missing:
        print(f"WARNING: {len(missing)} image(s) failed to upload and were skipped: {missing[:10]}")

    if args.commit:
        result = client.import_tasks(tasks)
        print(f"Imported into Label Studio project {args.project_id}: {result}")
    else:
        print(f"Dry run — inspect {args.tasks_json} and {args.crops_dir}, then re-run with --commit to import.")


if __name__ == "__main__":
    main()
