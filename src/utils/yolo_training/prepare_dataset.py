"""Build a YOLO training dataset from the annotated tasks of one or more Label Studio projects with a
RectangleLabels config (see src/utils/labelstudio_preannotate/labelstudio-config/yolo.xml), optionally
merged with an existing YOLO dataset so a fine-tune doesn't forget the pages it was trained on before.

The labels are written straight from each task's latest annotation instead of Label Studio's own YOLO
export, which leaves out images uploaded to its storage and numbers the classes in its own order. Here the
class indices are taken from --weights, so they always match the model being fine-tuned.

Output layout (what ultralytics expects):
    <output>/images/{train,val}/<page>.jpg
    <output>/labels/{train,val}/<page>.txt
    <output>/data.yaml

Setup, in the .env file in the repo root (see .env.example, never pass the token on the CLI):
    LS_URL=https://your-labelstudio-host      # default: http://localhost:8080
    LS_TOKEN=...                               # Label Studio API token

Usage:
    # Project 121 alone, 20% of its pages held out for validation:
    uv run python -m src.utils.yolo_training.prepare_dataset --project-id 121 --output datasets/layout_121

    # Merged with the dataset the current model was fine-tuned on, keeping that dataset's own split:
    uv run python -m src.utils.yolo_training.prepare_dataset --project-id 121 \\
        --extra-dataset /path/to/Staatsblad_preanno_2/YOLO/data.yaml --output datasets/layout_121_merged
"""

import argparse
import os
import random
import shutil

from collections import Counter
from pathlib import Path

import yaml

from dotenv import load_dotenv

from src.utils.labelstudio_preannotate.client import LabelStudioClient

_LS_URL_DEFAULT = "http://localhost:8080"
_SPLITS = ("train", "val")


def fetch_tasks(client: LabelStudioClient) -> list[dict]:
    """ All tasks of the client's project, with their annotations. """

    import requests

    resp = requests.get(f"{client.base_url}/api/projects/{client.project_id}/export", headers=client.headers,
                        params={"exportType": "JSON", "download_all_tasks": "true"}, timeout=600)
    resp.raise_for_status()
    return resp.json()


def task_to_yolo_lines(task: dict, class_index: dict[str, int], from_name: str, skipped: Counter) -> list[str] | None:
    """ The YOLO label lines for the latest non-cancelled annotation of TASK, or None if it has none.
        Rectangles with a label that isn't in CLASS_INDEX are counted in SKIPPED and left out. """

    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    annotation = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")

    lines = []
    for result in annotation["result"]:
        if result.get("type") != "rectanglelabels" or result.get("from_name") != from_name:
            continue
        value = result["value"]
        if value.get("rotation"):
            skipped["rotated"] += 1
            continue
        for label in value["rectanglelabels"]:
            if label not in class_index:
                skipped[label] += 1
                continue
            # Label Studio stores the top-left corner and size as percentages, YOLO the normalized centre and size
            w, h = value["width"] / 100, value["height"] / 100
            cx, cy = value["x"] / 100 + w / 2, value["y"] / 100 + h / 2
            lines.append(f"{class_index[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def export_project(client: LabelStudioClient, output: Path, class_index: dict[str, int], val_fraction: float,
                   seed: int, data_key: str, from_name: str) -> Counter:
    """ Downloads every annotated page of the client's project into OUTPUT, split into train/val. """

    tasks = fetch_tasks(client)
    labelled = []
    skipped = Counter()
    for task in tasks:
        lines = task_to_yolo_lines(task, class_index, from_name, skipped)
        if lines is not None:
            labelled.append((task, lines))
    print(f"Project {client.project_id}: {len(labelled)}/{len(tasks)} task(s) annotated")
    if skipped:
        print(f"  WARNING: left out rectangles not matching a model class: {dict(skipped)}")

    random.Random(seed).shuffle(labelled)
    n_val = round(len(labelled) * val_fraction)

    counts = Counter()
    for number, (task, lines) in enumerate(labelled, start=1):
        split = "val" if number <= n_val else "train"
        server_path = task["data"][data_key]
        name = Path(server_path).name
        (output / "images" / split / name).write_bytes(client.download_file(server_path))
        (output / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        counts[split] += 1
        if number % 20 == 0 or number == len(labelled):
            print(f"  downloaded {number}/{len(labelled)}")
    return counts


def copy_extra_dataset(data_yaml: Path, output: Path, names: dict[int, str]) -> Counter:
    """ Copies the train/val images and labels of an existing YOLO dataset into OUTPUT, keeping its split.
        Its classes must be the same as the model's, in the same order. """

    config = yaml.safe_load(data_yaml.read_text())
    extra_names = config["names"] if isinstance(config["names"], dict) else dict(enumerate(config["names"]))
    if {int(k): v for k, v in extra_names.items()} != names:
        raise SystemExit(f"Classes of {data_yaml} don't match the model's:\n  {extra_names}\n  {names}")

    root = Path(config.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root

    counts = Counter()
    for split in _SPLITS:
        images_dir = root / config[split]
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        for image in sorted(p for p in images_dir.iterdir() if p.is_file()):
            label = labels_dir / f"{image.stem}.txt"
            if (output / "images" / split / image.name).exists():
                raise SystemExit(f"{image.name} is in both the Label Studio export and {data_yaml}")
            shutil.copy2(image, output / "images" / split / image.name)
            if label.exists():
                shutil.copy2(label, output / "labels" / split / label.name)
            counts[split] += 1
    print(f"{data_yaml}: copied {counts['train']} train / {counts['val']} val page(s)")
    return counts


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-id", type=int, action="append", required=True, help="Label Studio project to export, repeatable")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write the dataset to, must not exist yet")
    parser.add_argument("--extra-dataset", type=Path, action="append", default=[], help="data.yaml of an existing YOLO dataset to merge in, repeatable")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="Model whose class names and order the labels use")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Share of each project's pages held out for validation (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the train/val split (default: %(default)s)")
    parser.add_argument("--ls-url", default=os.environ.get("LS_URL") or _LS_URL_DEFAULT, help="Label Studio base URL (env: LS_URL)")
    parser.add_argument("--ls-token", default=os.environ.get("LS_TOKEN"), help="Label Studio API token (env: LS_TOKEN)")
    parser.add_argument("--data-key", default="image", help="Task data key matching the Image tag's value=\"$...\" in the labeling config")
    parser.add_argument("--from-name", default="label", help="RectangleLabels control name in the labeling config")
    args = parser.parse_args()

    if not args.ls_token:
        raise SystemExit("--ls-token (or LS_TOKEN) is required")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists, pick a new --output")
    if not args.weights.exists():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    for data_yaml in args.extra_dataset:
        if not data_yaml.exists():
            raise SystemExit(f"{data_yaml} not found")
    if not 0 <= args.val_fraction < 1:
        raise SystemExit(f"--val-fraction must be in [0, 1): {args.val_fraction}")

    # Imported here so --help and argument errors don't wait for torch
    from ultralytics import YOLO

    names = dict(YOLO(str(args.weights)).names)
    class_index = {name: index for index, name in names.items()}

    for kind in ("images", "labels"):
        for split in _SPLITS:
            (args.output / kind / split).mkdir(parents=True)

    totals = Counter()
    for project_id in args.project_id:
        client = LabelStudioClient(args.ls_url, args.ls_token, project_id)
        totals += export_project(client, args.output, class_index, args.val_fraction, args.seed, args.data_key, args.from_name)
    for data_yaml in args.extra_dataset:
        totals += copy_extra_dataset(data_yaml, args.output, names)

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(args.output.resolve()), "train": "images/train", "val": "images/val",
                                         "names": names}, sort_keys=False, allow_unicode=True))
    print(f"Wrote {data_yaml}: {totals['train']} train / {totals['val']} val page(s)")


if __name__ == "__main__":
    main()
