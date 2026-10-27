""" Demo: GLiNER2.5 relation extraction on the test texts of the NER training data (model/gliner/data/test.json).

    uv run python -m src.processors.text_processor.relation_demo --n 3
    uv run python -m src.processors.text_processor.relation_demo --text "M. Joly est administrateur de la Société X."

    Relations are zero-shot: the base model is used without the BelHisFirm adapters (these are trained for entities
    only), so the descriptions in RELATIONS are what steers the model. Entities are extracted in the same pass over the
    same text for comparison.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, EntityExtractor

logger = logging.getLogger(__name__)

TEST_PATH = Path("model/gliner/data/test.json")

RELATIONS = {
    "represented_by": "A person who acts on behalf of another person or an organization (proxy, procuration)",
    "administrator_of": "A person who is an administrator, director, commissioner or manager of an organization",
    "shareholder_of": "A person or organization that owns shares of an organization",
    "established_in": "An organization that has its seat or is established in a city or place",
    "resides_in": "A person who lives in a city or place",
    "notary_in": "A notary who works in a city",
    "dissolved_by": "An organization that is dissolved or put in liquidation by a corporate body or a decision",
}


def relation_pairs(result: dict) -> list[dict]:
    """ Flat [{relation, head, tail, confidence}] from the result of extract(..., include_confidence=True), whatever the
        exact shape of a head/tail is (a string, a (text, confidence) tuple or a dict with text and confidence). """
    def text_and_confidence(side) -> tuple[str, float | None]:
        if isinstance(side, dict):
            return side.get("text", ""), side.get("confidence")
        if isinstance(side, (tuple, list)):
            return side[0], side[1] if len(side) > 1 else None
        return str(side), None

    pairs = []
    for relation, items in result.get("relation_extraction", {}).items():
        for item in items:
            if isinstance(item, dict) and "head" in item:
                head, tail = item["head"], item["tail"]
            else:
                head, tail = item[0], item[1]
            (head_text, head_conf), (tail_text, tail_conf) = text_and_confidence(head), text_and_confidence(tail)
            confidences = [c for c in (head_conf, tail_conf) if c is not None]
            pairs.append({"relation": relation, "head": head_text, "tail": tail_text,
                          "confidence": round(min(confidences), 3) if confidences else None})
    return sorted(pairs, key=lambda p: -(p["confidence"] or 0))


def load_texts(path: Path, n: int) -> list[str]:
    """ The n test texts with the most entities (the richest ones for relations). """
    samples = json.loads(path.read_text(encoding="utf-8"))
    samples.sort(key=lambda s: -sum(len(v) for v in s["entities"].values()))
    return [s["text"] for s in samples[:n]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GLiNER2.5 relation extraction demo.")
    parser.add_argument("--text", default=None, help="Your own text, instead of the test set")
    parser.add_argument("--n", type=int, default=3, help="How many test texts to take")
    parser.add_argument("--test-data", type=Path, default=TEST_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON file for the results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    texts = [args.text] if args.text else load_texts(args.test_data, args.n)
    extractor = EntityExtractor(model_name=args.model, device=args.device, threshold=args.threshold)
    model = extractor.model

    output = []
    for text in texts:
        raw = model.extract_relations(text, RELATIONS, threshold=args.threshold, include_confidence=True)
        pairs = relation_pairs(raw)
        entities = extractor.extract_text(text)
        output.append({"text": text, "relations": pairs, "entities": entities})

        print("=" * 100)
        print(text[:700] + (" …" if len(text) > 700 else ""))
        print("-" * 100)
        print(f"Entities ({len(entities)}): " + "; ".join(f"{e['text']} [{e['label']}]" for e in entities[:25]))
        print(f"Relations ({len(pairs)}):")
        for pair in pairs:
            print(f"  {pair['head']}  --{pair['relation']}-->  {pair['tail']}   ({pair['confidence']})")

    if args.out:
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
