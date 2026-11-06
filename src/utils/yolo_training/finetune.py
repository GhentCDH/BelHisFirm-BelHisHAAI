"""Fine-tune a YOLO layout checkpoint on a dataset built by prepare_dataset.py. Continues from --model instead
of a base COCO/ImageNet checkpoint, with a lower learning rate and fewer epochs suited to a small dataset.
The best weights end up in <project>/<name>/weights/best.pt; copy them over model/best.pt to use them in the
pipeline.

By default this script only PRINTS the command that would start training (the same script with --run, plus the
equivalent `yolo detect train ...` CLI call). Nothing is trained until you add --run.

Usage:
    uv run python -m src.utils.yolo_training.finetune --data datasets/layout_121/data.yaml --name finetune-121

    # Freeze the backbone (first 10 layers) so only the head adapts:
    uv run python -m src.utils.yolo_training.finetune --data datasets/layout_121/data.yaml --freeze 10

    # Scanned portrait pages with very tall tables: YOLO26 (no DFL box-size cap), document-friendly augmentation.
    # --model yolo26l.pt is downloaded automatically; --preset document sets the augmentation values listed in
    # DOCUMENT_PRESET below (each can still be overridden, e.g. --scale 0.3):
    uv run python -m src.utils.yolo_training.finetune --data datasets/layout_121/data.yaml \\
        --model yolo26l.pt --imgsz 1280 --preset document --name yolo26l-1280

    # Add --run to the printed command to actually start training.
"""

import argparse
import shlex
import sys

from pathlib import Path

# Settings for scanned, portrait document pages, applied by --preset document. Pages are consistently framed and
# printed on similar paper, a mirrored page is not a real layout, and big zoom/shift clips tall table boxes.
DOCUMENT_PRESET: dict[str, float | int | bool] = {
    "epochs": 100,
    "patience": 30,
    "mosaic": 0.0,
    "fliplr": 0.0,
    "scale": 0.15,
    "translate": 0.02,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "cos_lr": True,
}

# Float augmentation arguments that are only passed to Ultralytics when set (explicitly or via the preset);
# otherwise Ultralytics' own defaults apply.
AUGMENTATION_KEYS = ("fliplr", "scale", "translate", "degrees", "shear", "perspective", "hsv_h", "hsv_s", "hsv_v")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Path to the dataset's data.yaml")
    parser.add_argument(
        "--model",
        default="model/best.pt",
        help="Trained .pt checkpoint to fine-tune from. A bare Ultralytics name such as yolo26l.pt is downloaded automatically",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Default 50 (100 with --preset document)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Should match the checkpoint's original training imgsz")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device index, list of indices (e.g. '0,1'), or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience in epochs. Default 10 (30 with --preset document)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate; lower than from-scratch training to avoid catastrophic forgetting. Ignored when --optimizer is 'auto'")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer. With 'auto' (default) Ultralytics picks the optimizer and learning rate itself and ignores --lr0; set e.g. SGD or AdamW to make --lr0 take effect")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N layers (e.g. the backbone) so only later layers are updated")
    parser.add_argument("--no-mosaic", action="store_true", help="Disable mosaic augmentation (stitching 4 training images together)")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic probability (0-1). Default 1.0, or 0.0 with --preset document. --no-mosaic wins over this")
    parser.add_argument("--preset", choices=["document"], default=None, help="Apply document-scan defaults (see DOCUMENT_PRESET in this file); explicit arguments override them")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability. Default Ultralytics 0.5, preset 0.0 (mirrored pages are unrealistic)")
    parser.add_argument("--scale", type=float, default=None, help="Random zoom range +/- fraction. Default Ultralytics 0.5, preset 0.15. Large zoom can clip tall boxes")
    parser.add_argument("--translate", type=float, default=None, help="Random shift fraction. Default Ultralytics 0.1, preset 0.02")
    parser.add_argument("--degrees", type=float, default=None, help="Random rotation in degrees. Default Ultralytics 0.0, preset 0.0")
    parser.add_argument("--shear", type=float, default=None, help="Random shear in degrees. Default Ultralytics 0.0, preset 0.0")
    parser.add_argument("--perspective", type=float, default=None, help="Random perspective. Default Ultralytics 0.0, preset 0.0")
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=None, help="Hue jitter. Default Ultralytics 0.015, preset 0.0")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=None, help="Saturation jitter. Default Ultralytics 0.7, preset 0.3")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=None, help="Brightness jitter. Default Ultralytics 0.4, preset 0.2")
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=None, help="Cosine learning-rate schedule. Default off, on with --preset document")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Directory to write training runs to")
    parser.add_argument("--name", default="finetune", help="Run name, under --project")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted run at --project/--name")
    parser.add_argument("--run", action="store_true", help="Actually start training. Without this flag only the command is printed")
    return parser


def pick(value, key: str, preset: str | None):
    """An explicit argument wins; otherwise the preset's value; otherwise None (= leave to Ultralytics' default)."""
    if value is not None:
        return value
    return DOCUMENT_PRESET.get(key) if preset == "document" else None


def build_train_args(args: argparse.Namespace) -> dict:
    """Keyword arguments for YOLO.train(), with unset optional values left out."""
    if args.no_mosaic:
        mosaic = 0.0
    else:
        mosaic = pick(args.mosaic, "mosaic", args.preset)
        mosaic = 1.0 if mosaic is None else mosaic

    train_args = {
        "data": str(args.data.resolve()),
        "epochs": pick(args.epochs, "epochs", args.preset) or 50,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": pick(args.patience, "patience", args.preset) or 10,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "freeze": args.freeze,
        "mosaic": mosaic,
        "project": str(args.project.resolve()),
        "name": args.name,
        "resume": args.resume,
        "pretrained": True,
    }
    for key in AUGMENTATION_KEYS:
        value = pick(getattr(args, key), key, args.preset)
        if value is not None:
            train_args[key] = value
    cos_lr = pick(args.cos_lr, "cos_lr", args.preset)
    if cos_lr is not None:
        train_args["cos_lr"] = cos_lr
    return {key: value for key, value in train_args.items() if value is not None}


def print_command(args: argparse.Namespace, train_args: dict) -> None:
    script_args = [a for a in sys.argv[1:] if a != "--run"]
    script_command = shlex.join(["uv", "run", "python", "-m", "src.utils.yolo_training.finetune", *script_args, "--run"])
    cli_command = shlex.join(["yolo", "detect", "train", f"model={args.model}", *(f"{k}={v}" for k, v in train_args.items())])

    print("Command to start training (run this):\n")
    print(f"  {script_command}\n")
    print("Equivalent Ultralytics CLI call:\n")
    print(f"  {cli_command}\n")
    print(f"Weights will end up in {Path(train_args['project']) / args.name / 'weights' / 'best.pt'}")
    if args.optimizer == "auto":
        print("Note: --optimizer is 'auto', so Ultralytics ignores --lr0. Pass --optimizer SGD (or AdamW) to use it.")
    if "26" not in Path(args.model).stem:
        print(
            "Note: unless this checkpoint is YOLO26, its DFL box head can only predict boxes up to about 960 px tall "
            f"at the training size, i.e. {960 / args.imgsz:.0%} of the page height at imgsz {args.imgsz}. Taller tables are cut off."
        )


def main() -> None:
    args = build_parser().parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found")
    # A bare name like yolo26l.pt is downloaded by Ultralytics; anything that looks like a path must exist
    if any(sep in args.model for sep in "/\\") and not Path(args.model).exists():
        raise SystemExit(f"{args.model} not found")

    train_args = build_train_args(args)

    if not args.run:
        print_command(args, train_args)
        return

    # Imported here so --help and argument errors don't wait for torch
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(**train_args)


if __name__ == "__main__":
    main()
