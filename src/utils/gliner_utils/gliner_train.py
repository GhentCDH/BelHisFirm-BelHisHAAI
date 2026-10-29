""" Trains LoRA adapters for GLiNER2.5 (boundary architecture) on the entity types of IEschema.json: one adapter per
    domain (the "domains" of the schema, the default) or one for all entity types (--single). Used by entities.EntityExtractor
    (and labelstudio_cli --adapter-dir, gliner_eval), a single adapter also by the Joint IE extractor
    (entity_extractor.get_joint(adapter)).

    uv run python -m src.utils.gliner_utils.gliner_train --out model/gliner/adapters/v25_domains
    uv run python -m src.utils.gliner_utils.gliner_train --out model/gliner/adapters/v25_domains --domains date person
    uv run python -m src.utils.gliner_utils.gliner_train --out model/gliner/adapters/v25_single --single
    uv run python -m src.utils.gliner_utils.gliner_train --data-dir model/gliner/data_round2 --out model/gliner/adapters/v25_round2

    The data is train.json of the data folder, made by gliner_convert from the Label Studio export (the older data is
    model/gliner/data, copied from the ghentcdh-glinerv2-tutorial). A domain adapter learns the entity types of its domain
    from the samples that have at least one of them. A part of the samples (--val-split, 10%) is held out as validation
    set: the eval loss is computed every epoch, the best epoch is saved to <out>/[<domain>/]best and training stops early
    when it stops improving, the last epoch is saved to final. Score the adapters on the untouched test.json with gliner_eval.
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
from src.utils.gliner_utils.blocks import clean_value, load_domains

# Long records fragment the GPU memory, without this a few of them run out of it at batch size 1
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("model/gliner/data")


def clean_entities(text: str, entities: dict[str, list[str]], processor=None) -> tuple[dict[str, list[str]], list[dict]]:
    """ The entities that are really in the text. The boundary architecture refuses a sample with an empty entity,
        one that is not found verbatim (the export has a few with a literal backslash-n) or one that is not made of
        whole words (a partial word such as 'sociétair'), the span architecture silently skips them. Both get the
        same cleaned data. Returns the entities and the dropped ones as {label, value, reason}.

        Args: processor: The model's processor, to also drop what does not align to its word tokens. """
    dropped = []
    kept = {}
    for label, values in entities.items():
        kept[label] = []
        for value in map(clean_value, values):
            if not value:
                dropped.append({"label": label, "value": value, "reason": "empty"})
            elif value not in text:
                dropped.append({"label": label, "value": value, "reason": "not in the text"})
            elif value not in kept[label]:
                kept[label].append(value)
    if processor is not None:
        bad_indexes = {(bad["entity_type"], bad["value_index"]) for bad in processor.find_unalignable_entities(text, kept)}
        for label, index in sorted(bad_indexes, key=lambda item: -item[1]):
            dropped.append({"label": label, "value": kept[label].pop(index), "reason": "not whole words"})
    return kept, dropped


def load_samples(path: Path, processor=None, dropped_out: list | None = None) -> list[dict]:
    """ The {text, entities} samples of a train/test file, without the entities that are not in the text.

        Args: dropped_out: Gets the dropped entities, each as {label, value, reason, text}. """
    samples = json.loads(path.read_text(encoding="utf-8"))
    dropped = []
    for sample in samples:
        sample["entities"], found = clean_entities(sample["text"], sample["entities"], processor)
        dropped += [{**item, "text": sample["text"]} for item in found]
    if dropped:
        logger.info("%s: dropped %d entities that are empty or not in their text", path.name, len(dropped))
    if dropped_out is not None:
        dropped_out.extend(dropped)
    return samples


def show_dropped(dropped: list[dict], context: int = 40) -> None:
    """ Prints the dropped entities with the reason and where they would be in the text. """
    for item in dropped:
        text, value = item["text"], item["value"]
        # Where it nearly is: its first word, else the start of the text
        at = text.find(value.split()[0]) if value.split() and value.split()[0] in text else 0
        snippet = text[max(at - context, 0):at + len(value) + context].replace("\n", "\\n")
        print(f"[{item['label']}] {value!r}  ({item['reason']})\n    ...{snippet}...")
    print(f"\n{len(dropped)} dropped: " + ", ".join(f"{reason} {sum(i['reason'] == reason for i in dropped)}"
                                                    for reason in sorted({i['reason'] for i in dropped})))


def restrict(samples: list[dict], labels: list[str] | None) -> list[dict]:
    """ The samples with only these entity types (all of them as keys), without those that have none of them. """
    if labels is None:
        return samples
    reduced = [{"text": s["text"], "entities": {label: s["entities"].get(label, []) for label in labels}} for s in samples]
    return [s for s in reduced if any(s["entities"].values())]


def train_adapter(model_name: str, data_file: Path, out_dir: Path, epochs: int, seed: int,
                  batch_size: int, effective_batch: int, val_split: float = 0.1, patience: int = 3,
                  labels: list[str] | None = None) -> Path:
    """ Trains the LoRA adapter on a fresh copy of the base model. With a validation split the best epoch (lowest eval
        loss) is saved to out_dir/best and returned, else the last epoch, out_dir/final.

        Args: labels: Only these entity types, of the samples that have one, None for all. """
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from gliner2.training.data import InputExample

    model = AutoExtractor.from_pretrained(model_name)
    if getattr(model, "architecture", None) != "boundary":
        raise SystemExit(f"{model_name} is not a GLiNER2.5 (boundary) checkpoint")
    examples = [InputExample(text=item["text"], entities=item["entities"])
                for item in restrict(load_samples(data_file, model.processor), labels)]
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
    class Trainer(GLiNER2Trainer):
        def _log_metrics(self, metrics, prefix: str = "") -> None:
            """ The trainer only puts the train loss on its progress bar and never prints the eval metrics. """
            super()._log_metrics(metrics, prefix)
            if prefix == "eval" and isinstance(metrics, dict):
                logger.info("Validation, epoch %.0f: %s", metrics.get("epoch", 0), "  ".join(
                    f"{key.removeprefix('eval_')}={value:.4f}" for key, value in metrics.items()
                    if key not in ("step", "epoch") and isinstance(value, float)))

    trainer = Trainer(model=model, config=TrainingConfig(**settings))
    results = trainer.train(train_data=examples, eval_data=val_examples or None)
    logger.info("Adapter done in %.0f s", results["total_time_seconds"])

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir / ("best" if val_examples and (out_dir / "best").exists() else "final")


def main(argv: list[str] | None = None) -> int:
    domains = load_domains()
    parser = argparse.ArgumentParser(description="Train BelHisFirm LoRA adapters for GLiNER2.5.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GLiNER2.5 base checkpoint")
    parser.add_argument("--out", required=True, type=Path,
                        help="Folder for the adapters: one subfolder per domain, or the adapter itself with --single")
    parser.add_argument("--domains", nargs="+", choices=list(domains), default=list(domains), help="Domains to train")
    parser.add_argument("--show-dropped", action="store_true",
                        help="Only list the entities of train.json that are dropped before training, with the reason, and exit")
    parser.add_argument("--single", action="store_true", help="Train one adapter on all entity types instead")
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

    if args.show_dropped:
        from gliner2 import AutoExtractor

        dropped: list[dict] = []
        load_samples(args.data_dir / "train.json", AutoExtractor.from_pretrained(args.model).processor, dropped)
        show_dropped(dropped)
        return 0

    jobs = {None: None} if args.single else {domain: domains[domain] for domain in args.domains}
    for domain, labels in jobs.items():
        logger.info("Adapter: %s", f"{domain} ({', '.join(labels)})" if domain else "all entity types")
        final = train_adapter(args.model, args.data_dir / "train.json", args.out / domain if domain else args.out,
                              args.epochs, args.seed, args.batch_size, args.effective_batch, args.val_split,
                              args.patience, labels)
        logger.info("Saved %s", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
