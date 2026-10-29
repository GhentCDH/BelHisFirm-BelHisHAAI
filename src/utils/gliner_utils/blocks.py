""" The detections (text blocks) of an ocr_data.json and the Label Studio tasks made of them, shared by the GLiNER
    scripts. The entities themselves come from src.processors.text_processor.entity_extractor.
"""

import json
from pathlib import Path

from src.processors.text_processor.entity_extractor import SCHEMA_PATH

CONFIG_PATH = Path(__file__).with_name("labelstudio_config.xml")
# How many characters of the text before and after a detection are sent to Label Studio as context
DEFAULT_CONTEXT_CHARS = 400
OCR_FILE = "ocr_data.json"
# Block labels (YOLO layout classes) that make up the record's running text
TEXT_LABELS = ("Text", "Section-header")
# How a page limit is divided over the layout labels, in parts
DEFAULT_SPLIT = {"Section-header": 20, "Text": 80}


def load_labels(schema_path: Path = SCHEMA_PATH) -> dict[str, dict]:
    """ entity type -> {"description": ...} from the schema file. """
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))["entities"]


def labelstudio_config() -> str:
    """ Label Studio labeling config for the NER project the pre-annotations go to: a `label` Labels tag on a `text`
        Text tag (as in the training data export), grouped labels, and the text around the detection as read-only
        context. Paste it into the project's Labeling Interface > Code. """
    return CONFIG_PATH.read_text(encoding="utf-8")


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


def context_around(texts: list[str], index: int, chars: int) -> tuple[str, str]:
    """ (before, after): the last `chars` characters of the texts before texts[index] and the first `chars` of the
        texts after it, cut at a space and marked with an ellipsis where text is left out. """
    before = " ".join(texts[:index])
    after = " ".join(texts[index + 1:])
    if len(before) > chars:
        before = "… " + before[-chars:].partition(" ")[2]
    if len(after) > chars:
        after = after[:chars].rpartition(" ")[0] + " …"
    return before, after


def build_blocks(ocr_data: dict, block_labels: tuple[str, ...] | None = TEXT_LABELS,
                 max_per_page: int | None = None, split: dict[str, float] | None = None,
                 context_chars: int = DEFAULT_CONTEXT_CHARS) -> list[dict]:
    """ The detections (text blocks) with OCR text of all pages of an ocr_data.json structure, in reading order,
        with where they are. block_index is the position in the page of ocr_data.json, also when only some of
        the detections of a page are taken. context_before/context_after are the text around the detection on
        the page (every detection with text, also those that are not taken), only to show, never read by GLiNER. """
    blocks = []
    for page_idx, page in enumerate(ocr_data.get("pages", [])):
        texts = [(block_idx, block["text"].strip()) for block_idx, block in enumerate(page.get("blocks", []))
                 if (block.get("text") or "").strip()]
        page_texts = [text for _, text in texts]
        position = {block_idx: number for number, (block_idx, _) in enumerate(texts)}
        page_blocks = []
        for block_idx, block in enumerate(page.get("blocks", [])):
            if (block_labels is not None and block.get("label") not in block_labels) or block_idx not in position:
                continue
            before, after = context_around(page_texts, position[block_idx], context_chars)
            page_blocks.append(
                {"text": page_texts[position[block_idx]], "page": page_idx, "block_index": block_idx,
                 "bbox": block.get("bbox"), "label": block.get("label"), "column": block.get("column"),
                 "context_before": before, "context_after": after})
        blocks.extend(pick_largest(page_blocks, max_per_page, split))
    return blocks


def to_labelstudio_tasks(result: dict, record_folder: str, model_version: str) -> list[dict]:
    """ One Label Studio task per detection (text block) of a record, in reading order, with the entities as
        pre-annotations. Detections without entities are tasks too, to find what the model missed. The `source`
        in the task data identifies the detection, to leave out what is already in the project.
        Same labeling config as the training data: a `label` Labels tag on a `text` Text tag, the context is shown
        around it by the config. """
    tasks = []
    for block in result["blocks"]:
        task = {"data": {
            "text": block["text"], "context_before": block.get("context_before", ""),
            "context_after": block.get("context_after", ""), "source": f"{record_folder}/page_{block['page'] + 1:03d}/block_{block['block_index']:03d}",
            "record_id": result["record_id"], "record_title": result["record_title"], "record_folder": record_folder,
            "page": block["page"] + 1, "block_index": block["block_index"], "bbox": block["bbox"],
            "yolo_label": block["label"], "column": block["column"]}}
        if block["entities"]:
            task["predictions"] = [{"model_version": model_version, "result": [
                {"from_name": "label", "to_name": "text", "type": "labels", "score": entity["confidence"],
                 "value": {"start": entity["start"], "end": entity["end"], "text": entity["text"],
                           "labels": [entity["label"]]}}
                for entity in block["entities"]]}]
        tasks.append(task)
    return tasks


def predict_blocks(blocks: list[dict], adapter=None) -> list[dict]:
    """ The blocks with the entities of the Joint IE extractor (type as label, start/end in the block's text). """
    from src.processors.text_processor.entity_extractor import build_schema, extract

    schema = build_schema(adapter=adapter)
    return [{**block, "entities": [
        {"label": e["type"], "text": e["text"], "start": e["start"], "end": e["end"],
         "confidence": round(float(e.get("confidence", 1.0)), 4)}
        for e in extract(block["text"], schema, adapter)["entities"]]} for block in blocks]
