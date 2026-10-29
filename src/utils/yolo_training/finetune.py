"""Fine-tune an already-trained YOLO layout checkpoint on a dataset built by prepare_dataset.py. Continues
from --model instead of a base COCO/ImageNet checkpoint, with a lower learning rate and fewer epochs suited
to a small dataset. The best weights end up in <project>/<name>/weights/best.pt; copy them over
model/best.pt to use them in the pipeline.

Usage:
    uv run python -m src.utils.yolo_training.finetune --data datasets/layout_121/data.yaml --name finetune-121

    # Freeze the backbone (first 10 layers) so only the head adapts:
    uv run python -m src.utils.yolo_training.finetune --data datasets/layout_121/data.yaml --freeze 10
"""

import argparse

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Path to the dataset's data.yaml")
    parser.add_argument("--model", type=Path, default=Path("model/best.pt"), help="Trained .pt checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning is usually done with fewer epochs than full training")
    parser.add_argument("--imgsz", type=int, default=1024, help="Should match the checkpoint's original training imgsz")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device index, list of indices (e.g. '0,1'), or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate; lower than from-scratch training to avoid catastrophic forgetting")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N layers (e.g. the backbone) so only later layers are updated")
    parser.add_argument("--no-mosaic", action="store_true", help="Disable mosaic augmentation (stitching 4 training images together)")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Directory to write training runs to")
    parser.add_argument("--name", default="finetune", help="Run name, under --project")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted run at --project/--name")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found")
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found")

    # Imported here so --help and argument errors don't wait for torch
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        freeze=args.freeze,
        mosaic=0.0 if args.no_mosaic else 1.0,
        project=str(args.project.resolve()),
        name=args.name,
        resume=args.resume,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
