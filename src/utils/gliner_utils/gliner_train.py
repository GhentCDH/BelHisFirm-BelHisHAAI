""" Trains the BelHisFirm LoRA adapters for GLiNER2.5 (boundary architecture): one adapter per domain (see DOMAINS),
    or one for all labels (--single).

    uv run python -m src.processors.text_processor.gliner_train --out model/gliner/adapters/v25
    uv run python -m src.processors.text_processor.gliner_train --out model/gliner/adapters/v25 --domains date person
    uv run python -m src.processors.text_processor.gliner_train --out model/gliner/adapters/v25 --single

    The data is the Label Studio export of the BelHisFirm annotations, converted by GLiNER2_LoRA_BelHisFirm.ipynb of
    the ghentcdh-glinerv2-tutorial (train.json and train_<domain>.json), copied to model/gliner/data. New data made
    with gliner_convert is used through --data-dir.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, DOMAINS, SINGLE_ADAPTER

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


def train_adapter(model_name: str, data_file: Path, name: str, out_dir: Path, epochs: int, seed: int,
                  batch_size: int, effective_batch: int) -> Path:
    """ Trains one LoRA adapter on a fresh copy of the base model and saves it to out_dir/name. """
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from gliner2.training.data import InputExample

    model = AutoExtractor.from_pretrained(model_name)
    if getattr(model, "architecture", None) != "boundary":
        raise SystemExit(f"{model_name} is not a GLiNER2.5 (boundary) checkpoint")
    examples = [InputExample(text=item["text"], entities=item["entities"])
                for item in load_samples(data_file, model.processor)]
    settings = dict(
        output_dir=str(out_dir / name), experiment_name=name, num_epochs=epochs, seed=seed,
        batch_size=batch_size, gradient_accumulation_steps=effective_batch // batch_size, encoder_lr=1e-5, task_lr=5e-4,
        use_lora=True, lora_r=8, lora_alpha=16.0, lora_dropout=0.0, lora_target_modules=["encoder"],
        save_adapter_only=True, eval_strategy="no", logging_steps=50,
    )
    settings.update(fp16=False, bf16=True)

    logger.info("Training %s adapter on %d examples (%s)", name, len(examples), model_name)
    trainer = GLiNER2Trainer(model=model, config=TrainingConfig(**settings))
    results = trainer.train(train_data=examples)
    logger.info("%s adapter done in %.0f s", name, results["total_time_seconds"])

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir / name / "final"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train BelHisFirm LoRA adapters for GLiNER2.5.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GLiNER2.5 base checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="Folder for the adapters")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder with train.json and train_<domain>.json")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS), help="Domains to train")
    parser.add_argument("--single", action="store_true", help="Train one adapter on all labels (train.json) instead")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1, choices=[1, 2, 4, 8, 16],
                        help="Per step, the records are long so 4 does not fit in 24 GB, gradients are accumulated up to "
                             "--effective-batch")
    parser.add_argument("--effective-batch", type=int, default=16, help="Samples per optimizer step")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    jobs = {SINGLE_ADAPTER: args.data_dir / "train.json"} if args.single else \
        {domain: args.data_dir / f"train_{domain}.json" for domain in args.domains}
    for name, data_file in jobs.items():
        final = train_adapter(args.model, data_file, name, args.out, args.epochs, args.seed, args.batch_size,
                              args.effective_batch)
        logger.info("Saved %s", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
