""" GLiNER named-entity extraction on the full text of a record (reads ocr_data.json, writes entities.json).

    Uses GLiNER2.5 (boundary architecture), with or without the BelHisFirm LoRA adapters:
    - no adapters: one pass with all labels of the schema
    - adapter folder with one subfolder per domain (see DOMAINS): one pass per domain, each with its own adapter
    - adapter folder that is itself an adapter (trained with --single): one pass with all labels
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fastino/gliner2.5-multi-v1"
# Trained with gliner_train on the same architecture as DEFAULT_MODEL, used when it exists (gitignored, not in the repo)
DEFAULT_ADAPTER_DIR = Path("model/gliner/adapters/v25")
SCHEMA_PATH = Path(__file__).with_name("gliner_schema_BelHisFirm.json")
OCR_FILE = "ocr_data.json"
ENTITIES_FILE = "entities.json"
SINGLE_ADAPTER = "single"
# Block labels (YOLO layout classes) that make up the record's running text
TEXT_LABELS = ("Text", "Section-header")
# How a page limit is divided over the layout labels, in parts
DEFAULT_SPLIT = {"Section-header": 20, "Text": 80}

# The label groups the domain adapters are trained on
DOMAINS = {
    "date": ["date", "time_period"],
    "organisation": ["organization_name", "corporate_body", "legal_structure", "office_type"],
    "legal": ["legal_act", "deed_number", "archive_reference"],
    "person": ["person", "honorific_title", "occupation", "corporate_position"],
    "share": ["number_of_shares", "share_class", "share_nominal_value", "capital_category", "number_of_votes"],
    "location": ["country", "city", "street_address", "building_name"],
}
DOMAIN_OF = {label: domain for domain, labels in DOMAINS.items() for label in labels}


def load_labels(schema_path: Path = SCHEMA_PATH) -> dict[str, dict]:
    """ label -> {"description": ..., "threshold": ...} from the schema file (the one the adapters were trained with). """
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))["entities"]


def labelstudio_config(labels: dict[str, dict]) -> str:
    """ Label Studio labeling config for the NER project the pre-annotations go to (`label` on `text`, as in the
        training data export). Paste it into the project's Labeling Interface > Code. """
    lines = ["<View>", '  <Labels name="label" toName="text">']
    lines += [f'    <Label value="{label}"/>' for label in labels]
    lines += ["  </Labels>", '  <Text name="text" value="$text"/>', "</View>"]
    return "\n".join(lines)


def box_area(item: dict) -> float:
    """ Surface of the bbox [x1, y1, x2, y2] of a layout region or block, 0 without one. """
    bbox = item.get("bbox")
    return max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0) if bbox else 0


def pick_largest(items: list[dict], limit: int | None, split: dict[str, float] | None = None) -> list[dict]:
    """ At most `limit` of the items (dicts with a `label` and a `bbox`), the largest by surface, divided over the
        labels as `split` says (e.g. 20 Section-header, 80 Text: 2 and 8 of 10). A label that has fewer items than its
        share leaves the places to the largest of the others. Returns them in their original order. """
    if not limit or len(items) <= limit:
        return items

    split = split or DEFAULT_SPLIT
    total = sum(split.values())
    raw = {label: limit * weight / total for label, weight in split.items()}
    quota = {label: int(share) for label, share in raw.items()}
    for label in sorted(raw, key=lambda label: raw[label] - quota[label], reverse=True)[:limit - sum(quota.values())]:
        quota[label] += 1

    by_size = sorted(range(len(items)), key=lambda i: box_area(items[i]), reverse=True)
    picked: list[int] = []
    for label, places in quota.items():
        picked += [i for i in by_size if items[i].get("label") == label][:places]
    # Places a label cannot fill, and labels that are not in the split, go to the largest that are left
    picked += [i for i in by_size if i not in picked][:limit - len(picked)]
    return [items[i] for i in sorted(picked)]


def find_adapter(path: Path) -> Path | None:
    """ The folder with the adapter files: the folder itself, or the trainer's final/ checkpoint inside it. """
    for candidate in (path, path / "final"):
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


class EntityExtractor:
    """ Loads GLiNER once (plus its adapters) and extracts entities from strings, blocks or record folders. """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda", adapter_dir: Path | None = None,
                 schema_path: Path = SCHEMA_PATH, threshold: float | None = None,
                 block_labels: tuple[str, ...] | None = TEXT_LABELS, max_blocks_per_page: int | None = None,
                 split: dict[str, float] | None = None) -> None:
        """ threshold: minimum confidence for every label, the per-label thresholds of the schema when left out.
            block_labels: Which detections of the OCR (YOLO layout labels) are read, None for all of them.
            max_blocks_per_page: At most this many detections per page, the largest ones by surface, divided over
            the layout labels as split says (pick_largest), None for all of them. """
        from gliner2 import AutoExtractor  # Heavy import, only needed once an extractor is built

        self.model_name = model_name
        self.threshold = threshold
        self.block_labels = block_labels
        self.max_blocks_per_page = max_blocks_per_page
        self.split = split
        self.labels = load_labels(schema_path)
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None

        logger.info("Loading %s on %s", model_name, device)
        self.model = AutoExtractor.from_pretrained(model_name, map_location=device)
        self.architecture = getattr(self.model, "architecture", None)
        if self.architecture != "boundary":
            raise ValueError(f"{model_name} is not a GLiNER2.5 (boundary) checkpoint")
        self.adapters = self._load_adapters()
        logger.info("Architecture: %s, adapters: %s", self.architecture, list(self.adapters) or "none")

    def _load_adapters(self) -> dict[str, Path]:
        """ Attaches the adapters found in adapter_dir to the model, one named PEFT adapter per domain. """
        if self.adapter_dir is None:
            return {}

        single = find_adapter(self.adapter_dir)
        found = {SINGLE_ADAPTER: single} if single else {
            domain: path for domain in DOMAINS
            if (path := find_adapter(self.adapter_dir / domain)) is not None
        }
        if not found:
            raise FileNotFoundError(f"No adapters found in {self.adapter_dir}")

        from peft import PeftModel

        for number, (name, path) in enumerate(found.items()):
            if number == 0:
                self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
            else:
                self.model.load_adapter(str(path), adapter_name=name)
        self.model.eval()
        return found

    def _passes(self) -> list[tuple[str | None, dict[str, dict]]]:
        """ (adapter name, labels) for every pass over the texts. """
        if not self.adapters:
            return [(None, self.labels)]
        if SINGLE_ADAPTER in self.adapters:
            return [(SINGLE_ADAPTER, self.labels)]
        return [(domain, {label: self.labels[label] for label in labels if label in self.labels})
                for domain, labels in DOMAINS.items() if domain in self.adapters]

    def _schema(self, labels: dict[str, dict]):
        entities = {
            label: {"description": config.get("description", ""),
                    "threshold": self.threshold if self.threshold is not None else config.get("threshold", 0.5)}
            for label, config in labels.items()
        }
        base = self.model.get_base_model() if self.adapters else self.model
        return base.create_schema().entities(entities)

    def extract_texts(self, texts: list[str]) -> list[list[dict]]:
        """ Entities per text as flat dicts (label, domain, text, confidence, start, end), in text order.
            Pass all texts at once, with adapters every domain adapter is activated only once. """
        results: list[list[dict]] = [[] for _ in texts]
        indexed = [(number, text) for number, text in enumerate(texts) if text.strip()]
        if not indexed:
            return results

        for adapter, labels in self._passes():
            if adapter is not None:
                self.model.set_adapter(adapter)
            try:
                found = self.model.batch_extract_long(
                    [text for _, text in indexed], self._schema(labels),
                    threshold=self.threshold if self.threshold is not None else 0.5,
                    include_confidence=True, include_spans=True)
            except Exception:
                logger.exception("Extraction failed for %s, its labels are left out", adapter or "all labels")
                continue

            for (number, _), result in zip(indexed, found):
                for label, entities in result.get("entities", {}).items():
                    results[number].extend(
                        {"label": label, "domain": DOMAIN_OF.get(label), "text": entity["text"],
                         "confidence": round(float(entity["confidence"]), 4),
                         "start": entity["start"], "end": entity["end"]}
                        for entity in entities)

        return [sorted(entities, key=lambda e: (e["start"], -e["end"])) for entities in results]

    def extract_text(self, text: str) -> list[dict]:
        """ Entities of one string. """
        return self.extract_texts([text])[0]

    def build_blocks(self, ocr_data: dict) -> list[dict]:
        """ The detections (text blocks) with OCR text of all pages of an ocr_data.json structure, in reading order,
            with where they are. block_index is the position in the page of ocr_data.json, also when only some of
            the detections of a page are taken. """
        blocks = []
        for page_idx, page in enumerate(ocr_data.get("pages", [])):
            page_blocks = [
                {"text": block["text"].strip(), "page": page_idx, "block_index": block_idx, "bbox": block.get("bbox"),
                 "label": block.get("label"), "column": block.get("column")}
                for block_idx, block in enumerate(page.get("blocks", []))
                if (self.block_labels is None or block.get("label") in self.block_labels)
                and (block.get("text") or "").strip()
            ]
            blocks.extend(pick_largest(page_blocks, self.max_blocks_per_page, self.split))
        return blocks

    def extract_blocks(self, ocr_data_list: list[dict]) -> list[dict]:
        """ Entities of several ocr_data.json structures, each block is its own text, as in the training data.
            Returns per structure: record_id, record_title, blocks (text, page, block_index, bbox, entities). """
        records = [(ocr_data, self.build_blocks(ocr_data)) for ocr_data in ocr_data_list]
        entities = iter(self.extract_texts([block["text"] for _, blocks in records for block in blocks]))
        return [
            {"record_id": ocr_data.get("record_id"), "record_title": ocr_data.get("record_title"),
             "blocks": [{**block, "entities": next(entities)} for block in blocks]}
            for ocr_data, blocks in records
        ]

    def extract_records(self, record_folders: list[Path]) -> list[dict]:
        """ entities.json content for every record folder (see build_result), run over all records at once. """
        ocr_data_list = [json.loads((folder / OCR_FILE).read_text(encoding="utf-8")) for folder in record_folders]
        return [self.build_result(record) for record in self.extract_blocks(ocr_data_list)]

    def extract_record(self, record_folder: Path) -> dict:
        """ entities.json content of one record folder. """
        return self.extract_records([record_folder])[0]

    def extract_ocr_data(self, ocr_data: dict) -> dict:
        """ entities.json content of an ocr_data.json structure, saved or still in memory. """
        return self.build_result(self.extract_blocks([ocr_data])[0])

    def build_result(self, record: dict) -> dict:
        """ Flat entities.json structure of an extract_blocks record. start/end are relative to the block's text. """
        return {
            "record_id": record["record_id"],
            "record_title": record["record_title"],
            "model": self.model_name,
            "architecture": self.architecture,
            "adapters": {name: str(path) for name, path in self.adapters.items()},
            "threshold": self.threshold,
            "blocks": record["blocks"],
            "entities": [
                {**entity, "page": block["page"], "block_index": block["block_index"], "bbox": block["bbox"]}
                for block in record["blocks"] for entity in block["entities"]
            ],
        }

    def to_labelstudio_tasks(self, result: dict, record_folder: str) -> list[dict]:
        """ One Label Studio task per detection (text block) of a record, in reading order, with the entities as
            pre-annotations. Detections without entities are tasks too, to find what the model missed. The `source`
            in the task data identifies the detection, to leave out what is already in the project.
            Same labeling config as the training data: a `label` Labels tag on a `text` Text tag. """
        version = "+".join([self.model_name, *self.adapters]) if self.adapters else self.model_name
        tasks = []
        for block in result["blocks"]:
            task = {"data": {
                "text": block["text"], "source": f"{record_folder}/page_{block['page'] + 1:03d}/block_{block['block_index']:03d}",
                "record_id": result["record_id"], "record_title": result["record_title"], "record_folder": record_folder,
                "page": block["page"] + 1, "block_index": block["block_index"], "bbox": block["bbox"],
                "yolo_label": block["label"], "column": block["column"]}}
            if block["entities"]:
                task["predictions"] = [{"model_version": version, "result": [
                    {"from_name": "label", "to_name": "text", "type": "labels", "score": entity["confidence"],
                     "value": {"start": entity["start"], "end": entity["end"], "text": entity["text"],
                               "labels": [entity["label"]]}}
                    for entity in block["entities"]]}]
            tasks.append(task)
        return tasks

    @staticmethod
    def save(record_folder: Path, result: dict) -> Path:
        """ Writes entities.json next to the record's ocr_data.json. """
        path = record_folder / ENTITIES_FILE
        # The blocks are already in ocr_data.json, entities.json only keeps the flat entity list
        path.write_text(json.dumps({key: value for key, value in result.items() if key != "blocks"},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        return path
