""" Entity extraction with GLiNER2.5 (or GLiNER2, any checkpoint of --model) and the LoRA adapters of gliner_train, without relations (see entity_extractor for
    the Joint IE extraction). The adapters are:
    - none: one pass with all entity types of the schema
    - a single adapter (a folder with adapter_config.json, or its best/ or final/): one pass with all entity types
    - a folder with one adapter per domain (the domains of IEschema.json): one pass per domain, with its entity types
"""

import logging
from pathlib import Path

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, SCHEMA_PATH, find_adapter
from src.utils.gliner_utils.blocks import load_domains, load_labels

logger = logging.getLogger(__name__)

SINGLE = "single"
DEFAULT_THRESHOLD = 0.5


def find_adapters(adapter_dir: Path, domains: dict[str, list[str]]) -> dict[str, Path]:
    """ name -> adapter folder: {"single": ...} for one adapter, else one per domain that has one. """
    try:
        return {SINGLE: find_adapter(adapter_dir)}
    except FileNotFoundError:
        pass
    found = {}
    for domain in domains:
        try:
            found[domain] = find_adapter(Path(adapter_dir) / domain)
        except FileNotFoundError:
            continue
    if not found:
        raise FileNotFoundError(f"No adapters found in {adapter_dir}")
    return found


class EntityExtractor:
    """ Loads GLiNER2.5 once (plus its adapters) and extracts entities from strings. """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, adapter_dir: Path | None = None,
                 schema_path: Path = SCHEMA_PATH, threshold: float | None = None) -> None:
        """ threshold: minimum confidence for every type, the per-type `threshold` of the schema or 0.5 when left out. """
        import torch
        from gliner2 import AutoExtractor  # Heavy import, only needed once an extractor is built

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.threshold = model_name, threshold
        self.labels = load_labels(schema_path)
        self.domains = load_domains(schema_path)

        logger.info("Loading %s on %s", model_name, device)
        self.model = AutoExtractor.from_pretrained(model_name, map_location=device)
        self.adapters = self._load_adapters(adapter_dir) if adapter_dir else {}
        logger.info("Architecture: %s, adapters: %s", getattr(self.model, "architecture", None), list(self.adapters) or "none")

    def _load_adapters(self, adapter_dir: Path) -> dict[str, Path]:
        """ Attaches the adapters to the model, one named PEFT adapter each. """
        from peft import PeftModel

        found = find_adapters(Path(adapter_dir), self.domains)
        for number, (name, path) in enumerate(found.items()):
            if number == 0:
                self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
            else:
                self.model.load_adapter(str(path), adapter_name=name)
        self.model.eval()
        return found

    def _passes(self) -> list[tuple[str | None, list[str]]]:
        """ (adapter name, entity types) for every pass over the texts. """
        if not self.adapters:
            return [(None, list(self.labels))]
        if SINGLE in self.adapters:
            return [(SINGLE, list(self.labels))]
        return [(domain, labels) for domain, labels in self.domains.items() if domain in self.adapters]

    def _schema(self, labels: list[str]):
        entities = {label: {"description": self.labels[label].get("description", ""),
                            "threshold": self.threshold if self.threshold is not None
                            else self.labels[label].get("threshold", DEFAULT_THRESHOLD)} for label in labels}
        base = self.model.get_base_model() if self.adapters else self.model
        return base.create_schema().entities(entities)

    def extract_texts(self, texts: list[str]) -> list[list[dict]]:
        """ Entities per text as dicts (label, text, confidence, start, end), in text order. Pass all texts at once,
            every adapter is then activated only once. """
        results: list[list[dict]] = [[] for _ in texts]
        indexed = [(number, text) for number, text in enumerate(texts) if text.strip()]
        for adapter, labels in self._passes():
            if not indexed:
                break
            if adapter is not None:
                self.model.set_adapter(adapter)
            found = self.model.batch_extract_long(
                [text for _, text in indexed], self._schema(labels), threshold=self.threshold or DEFAULT_THRESHOLD,
                include_confidence=True, include_spans=True)
            for (number, _), result in zip(indexed, found):
                for label, entities in result.get("entities", {}).items():
                    results[number].extend(
                        {"label": label, "text": entity["text"], "confidence": round(float(entity["confidence"]), 4),
                         "start": entity["start"], "end": entity["end"]} for entity in entities)
        return [sorted(entities, key=lambda e: (e["start"], -e["end"])) for entities in results]
