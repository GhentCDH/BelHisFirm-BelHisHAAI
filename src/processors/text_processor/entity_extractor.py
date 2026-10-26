""" GLiNER2.5 named-entity extraction on the full text of a record (reads ocr_data.json, writes entities.json). """

import bisect
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fastino/gliner2.5-multi-v1"
DEFAULT_THRESHOLD = 0.5
OCR_FILE = "ocr_data.json"
ENTITIES_FILE = "entities.json"
# Block labels (YOLO layout classes) that make up the record's running text
TEXT_LABELS = ("Text", "Section-header")

# label -> description, the description is passed to GLiNER as a hint
ENTITY_LABELS = {
    "person": "a person's name, e.g. JULES MAISIN, Cerckel",
    "company": "a company or société name, e.g. Sucrerie de Landen",
    "place": "a city, municipality or address, e.g. Landen",
    "date": "a date, e.g. 25 juin 1909",
    "money": "a monetary amount, e.g. 2 fr. 40 c.",
    "role": "a function or title, e.g. administrateur délégué, receveur",
}


class EntityExtractor:
    """ Loads GLiNER once and extracts entities per record or per string. """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda",
                 threshold: float = DEFAULT_THRESHOLD, labels: dict[str, str] | None = None) -> None:
        from gliner2 import AutoExtractor  # Heavy import, only needed once an extractor is built

        self.model_name = model_name
        self.threshold = threshold
        self.labels = labels or ENTITY_LABELS
        logger.info("Loading %s on %s", model_name, device)
        self.model = AutoExtractor.from_pretrained(model_name, map_location=device)

    def extract_text(self, text: str) -> list[dict]:
        """ Entities of one string as flat dicts (label, text, confidence, start, end), in text order. """
        result = self.model.extract_entities_long(
            text, self.labels, threshold=self.threshold, include_confidence=True, include_spans=True)
        entities = [
            {"label": label, "text": entity["text"], "confidence": round(float(entity["confidence"]), 4),
             "start": entity["start"], "end": entity["end"]}
            for label, found in result.get("entities", {}).items()
            for entity in found
        ]
        return sorted(entities, key=lambda entity: entity["start"])

    @staticmethod
    def build_record_text(ocr_data: dict) -> tuple[str, list[dict]]:
        """ Joins the text blocks of all pages with newlines. Returns the text and, per block, its character
            offset and (page, block_index, bbox) so spans can be traced back to the page. """
        parts, offsets, position = [], [], 0
        for page_idx, page in enumerate(ocr_data.get("pages", [])):
            for block_idx, block in enumerate(page.get("blocks", [])):
                text = (block.get("text") or "").strip()
                if block.get("label") not in TEXT_LABELS or not text:
                    continue
                offsets.append({"start": position, "end": position + len(text), "page": page_idx,
                                "block_index": block_idx, "bbox": block.get("bbox")})
                parts.append(text)
                position += len(text) + 1
        return "\n".join(parts), offsets

    def extract_record(self, record_folder: Path) -> dict:
        """ Entities of one record folder, each traced back to its page, block and bbox. """
        ocr_data = json.loads((record_folder / OCR_FILE).read_text(encoding="utf-8"))
        return self.extract_ocr_data(ocr_data)

    def extract_ocr_data(self, ocr_data: dict) -> dict:
        """ Entities of an ocr_data.json structure ({"pages": [{"blocks": [...]}], ...}), saved or still in memory. """
        text, offsets = self.build_record_text(ocr_data)
        starts = [block["start"] for block in offsets]

        entities = []
        for entity in self.extract_text(text) if text else []:
            idx = bisect.bisect_right(starts, entity["start"]) - 1
            block = offsets[idx] if idx >= 0 and entity["start"] < offsets[idx]["end"] else None
            entities.append({
                **entity,
                "page": block["page"] if block else None,
                "block_index": block["block_index"] if block else None,
                "bbox": block["bbox"] if block else None,
            })

        return {
            "record_id": ocr_data.get("record_id"),
            "record_title": ocr_data.get("record_title"),
            "model": self.model_name,
            "threshold": self.threshold,
            "labels": self.labels,
            "entities": entities,
        }

    @staticmethod
    def save(record_folder: Path, result: dict) -> Path:
        """ Writes entities.json next to the record's ocr_data.json. """
        path = record_folder / ENTITIES_FILE
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
