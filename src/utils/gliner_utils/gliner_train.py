""" Trains a LoRA adapter for GLiNER2.5 (boundary architecture) on all entity types of IEschema.json. The adapter is
    used by the Joint IE extractor (entity_extractor.get_joint(adapter)) and by labelstudio_cli --adapter.

    uv run python -m src.utils.gliner_utils.gliner_train --out model/gliner/adapters/v25_joint
    uv run python -m src.utils.gliner_utils.gliner_train --data-dir model/gliner/data_round2 --out model/gliner/adapters/v25_round2

    A part of train.json (--val-split, 10%) is held out as validation set: the eval loss is computed every epoch, the
    best epoch is saved to <out>/best and training stops early when it stops improving. Score the adapter on the
    untouched test.json with gliner_eval.

    The data is train.json of the data folder, made by gliner_convert from the Label Studio export (the older
    data is model/gliner/data, copied from the ghentcdh-glinerv2-tutorial). The last epoch is saved to <out>/final.
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL

# Long records fragment the GPU memory, without this a few of them run out of it at batch size 1
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("model/gliner/data")


def clean_entities(text: str, entities: dict[str, list[str]], processor=None) -> tuple[dict[str, list[str]], int]:
    """ The entities that are really in the text. The boundary architecture refuses a sample with an empty entity,
        one that is not found verbatim (the export has a few with a literal backslash-n) or one that is not made of
        whole words (a partial word such as 'sociétair'), the span architecture silently skips them. Both get the
        same cleaned data. Returns the entities and how many were dropped.

        Args: processor: The model's processor, to also drop what does not align to its word tokens. """
    kept = {label: [value for value in values if value.strip() and value in text] for label, values in entities.items()}
    if processor is not None:
        for bad in processor.find_unalignable_entities(text, kept):
            kept[bad["entity_type"]][bad["value_index"]] = None
        kept = {label: [value for value in values if value is not None] for label, values in kept.items()}
    return kept, sum(len(v) for v in entities.values()) - sum(len(v) for v in kept.values())


def load_samples(path: Path, processor=None) -> list[dict]:
    """ The {text, entities} samples of a train/test file, without the entities that are not in the text. """
    samples = json.loads(path.read_text(encoding="utf-8"))
    dropped = 0
    for sample in samples:
        sample["entities"], count = clean_entities(sample["text"], sample["entities"], processor)
        dropped += count
    if dropped:
        logger.info("%s: dropped %d entities that are empty or not in their text", path.name, dropped)
    return samples


def train_adapter(model_name: str, data_file: Path, out_dir: Path, epochs: int, seed: int,
                  batch_size: int, effective_batch: int, val_split: float = 0.1, patience: int = 3) -> Path:
    """ Trains the LoRA adapter on a fresh copy of the base model. With a validation split the best epoch (lowest eval
        loss) is saved to out_dir/best and returned, else the last epoch, out_dir/final. """
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from gliner2.training.data import InputExample

    model = AutoExtractor.from_pretrained(model_name)
    if getattr(model, "architecture", None) != "boundary":
        raise SystemExit(f"{model_name} is not a GLiNER2.5 (boundary) checkpoint")
    examples = [InputExample(text=item["text"], entities=item["entities"])
                for item in load_samples(data_file, model.processor)]
    val_examples = []
    if val_split > 0:
        random.Random(seed).shuffle(examples)
        n_val = max(1, round(len(examples) * val_split))
        examples, val_examples = examples[n_val:], examples[:n_val]
    settings = dict(
        output_dir=str(out_dir), experiment_name=out_dir.name, num_epochs=epochs, seed=seed,
        batch_size=batch_size, gradient_accumulation_steps=effective_batch // batch_size, encoder_lr=1e-5, task_lr=5e-4,
        use_lora=True, lora_r=8, lora_alpha=16.0, lora_dropout=0.0, lora_target_modules=["encoder"],
        save_adapter_only=True, logging_steps=50,
        **(dict(eval_strategy="epoch", save_best=True, metric_for_best="eval_loss", greater_is_better=False,
                early_stopping=True, early_stopping_patience=patience) if val_examples else dict(eval_strategy="no")),
    )
    settings.update(fp16=False, bf16=True)

    logger.info("Training the adapter on %d examples, %d for validation (%s)", len(examples), len(val_examples), model_name)
    trainer = GLiNER2Trainer(model=model, config=TrainingConfig(**settings))
    results = trainer.train(train_data=examples, eval_data=val_examples or None)
    logger.info("Adapter done in %.0f s", results["total_time_seconds"])

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir / ("best" if val_examples and (out_dir / "best").exists() else "final")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a BelHisFirm LoRA adapter for GLiNER2.5.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GLiNER2.5 base checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="Folder for the adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder with train.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of train.json held out for validation, early stopping and the best epoch (0 = none)")
    parser.add_argument("--patience", type=int, default=3, help="Epochs without a better eval loss before stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1, choices=[1, 2, 4, 8, 16],
                        help="Per step, the records are long so 4 does not fit in 24 GB, gradients are accumulated up to "
                             "--effective-batch")
    parser.add_argument("--effective-batch", type=int, default=16, help="Samples per optimizer step")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    final = train_adapter(args.model, args.data_dir / "train.json", args.out, args.epochs, args.seed, args.batch_size,
                          args.effective_batch, args.val_split, args.patience)
    logger.info("Saved %s", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
