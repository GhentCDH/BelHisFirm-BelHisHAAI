import torch
from transformers import LukeForEntityPairClassification, MLukeTokenizer

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb

MODEL_ID = "studio-ousia/mluke-large-lite-finetuned-kbp37"


def resolve_label(id2label, idx):
    label = id2label.get(idx, f"LABEL_{idx}")
    if label.startswith("LABEL_"):
        return f"relation_{idx}"
    return label


def normalize_entity_pairs(entity_pairs):
    """
    Supported dict keys per pair:
    - entity_one, entity_two
    - text_span_one, text_span_two
    - entity_one_type (optional), entity_two_type (optional)

    Supported tuple/list order per pair:
    (entity_one, entity_two, text_span_one, text_span_two, entity_one_type, entity_two_type)
    """
    normalized = []

    if isinstance(entity_pairs, dict):
        entity_pairs = (entity_pairs,)
    elif isinstance(entity_pairs, tuple) and entity_pairs and not isinstance(entity_pairs[0], (dict, tuple, list)):
        # Single pair passed directly as a flat tuple.
        entity_pairs = (entity_pairs,)

    for pair in entity_pairs:
        if isinstance(pair, dict):
            normalized.append(
                {
                    "entity_one": pair["entity_one"],
                    "entity_two": pair["entity_two"],
                    "text_span_one": pair["text_span_one"],
                    "text_span_two": pair["text_span_two"],
                    "entity_one_type": pair.get("entity_one_type", ""),
                    "entity_two_type": pair.get("entity_two_type", ""),
                }
            )
            continue

        if isinstance(pair, (tuple, list)) and len(pair) >= 4:
            normalized.append(
                {
                    "entity_one": pair[0],
                    "entity_two": pair[1],
                    "text_span_one": tuple(pair[2]),
                    "text_span_two": tuple(pair[3]),
                    "entity_one_type": pair[4] if len(pair) > 4 else "",
                    "entity_two_type": pair[5] if len(pair) > 5 else "",
                }
            )
            continue

        raise ValueError(
            "Each entity pair must be a dict or a tuple/list with at least 4 values: "
            "(entity_one, entity_two, text_span_one, text_span_two, ...)"
        )

    return normalized


# Create text spans for each word
def create_text_spans(text):
    words = text.split()
    char_idx = 0
    for word in words:
        start_idx = text.find(word, char_idx)
        end_idx = start_idx + len(word)
        print({"word": word, "start": start_idx, "end": end_idx})
        char_idx = end_idx


def generate_relations(
    text,
    labels=None,
    entity_one=None,
    entity_two=None,
    text_span_one=None,
    text_span_two=None,
    entity_one_type="",
    entity_two_type="",
    entity_pairs=None,
    num_outputs=5,
    device="cpu",
):

    model_device = torch.device(device)
    num_outputs = max(1, int(num_outputs))
    tokenizer = MLukeTokenizer.from_pretrained(MODEL_ID)
    classifier = LukeForEntityPairClassification.from_pretrained(MODEL_ID).to(model_device)

    if entity_pairs is not None:
        pairs_to_run = normalize_entity_pairs(entity_pairs)
    else:
        if entity_one is None or entity_two is None or text_span_one is None or text_span_two is None:
            raise ValueError(
                "Provide either entity_pairs=[...] or the single-pair arguments "
                "entity_one/entity_two/text_span_one/text_span_two."
            )
        pairs_to_run = [
            {
                "entity_one": entity_one,
                "entity_two": entity_two,
                "text_span_one": tuple(text_span_one),
                "text_span_two": tuple(text_span_two),
                "entity_one_type": entity_one_type,
                "entity_two_type": entity_two_type,
            }
        ]

    all_predictions = []

    for pair in pairs_to_run:
        pair_entity_one = pair["entity_one"]
        pair_entity_two = pair["entity_two"]
        pair_span_one = pair["text_span_one"]
        pair_span_two = pair["text_span_two"]
        pair_entity_one_type = pair["entity_one_type"]
        pair_entity_two_type = pair["entity_two_type"]

        text_span_one_0 = int(pair_span_one[0])
        text_span_one_1 = int(pair_span_one[1])
        text_span_two_0 = int(pair_span_two[0])
        text_span_two_1 = int(pair_span_two[1])

        entity_spans = [(text_span_one_0, text_span_one_1), (text_span_two_0, text_span_two_1)]

        inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = classifier(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            top_k = min(num_outputs, probabilities.shape[-1])
            top_scores, top_indices = torch.topk(probabilities[0], k=top_k)

        id2label = classifier.config.id2label
        predictions = []

        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            label = resolve_label(id2label, idx)
            if labels and label not in labels:
                continue

            predictions.append(
                {
                    "entity_1": {"text": pair_entity_one, "type": pair_entity_one_type, "span": (text_span_one_0, text_span_one_1)},
                    "relation_type": label,
                    "entity_2": {"text": pair_entity_two, "type": pair_entity_two_type, "span": (text_span_two_0, text_span_two_1)},
                    "score": float(score),
                }
            )

        if not predictions:
            best_idx = int(torch.argmax(probabilities, dim=-1)[0].item())
            best_label = resolve_label(id2label, best_idx)
            best_score = float(probabilities[0][best_idx].item())
            predictions.append(
                {
                    "entity_1": {"text": pair_entity_one, "type": pair_entity_one_type, "span": (text_span_one_0, text_span_one_1)},
                    "relation_type": best_label,
                    "entity_2": {"text": pair_entity_two, "type": pair_entity_two_type, "span": (text_span_two_0, text_span_two_1)},
                    "score": best_score,
                    "note": "No prediction matched provided labels; returned best mLUKE prediction.",
                }
            )

        all_predictions.append(
            {
                "pair": {
                    "entity_one": pair_entity_one,
                    "entity_two": pair_entity_two,
                    "text_span_one": (text_span_one_0, text_span_one_1),
                    "text_span_two": (text_span_two_0, text_span_two_1),
                    "entity_one_type": pair_entity_one_type,
                    "entity_two_type": pair_entity_two_type,
                },
                "predictions": predictions,
            }
        )

    print("Predicted Relations (mLUKE):")
    for pair_result in all_predictions:
        pair = pair_result["pair"]
        print("Pair:", pair["entity_one"], "<->", pair["entity_two"])
        for relation in pair_result["predictions"]:
            print(relation["entity_1"])
            print("Label :", relation["relation_type"], "| score:", round(relation["score"], 4))
            print(relation["entity_2"])
            if "note" in relation:
                print("Note :", relation["note"])
            print("---")

    return all_predictions


if __name__ == "__main__":
    labels = None
    text = "MM. Louis van Langenhove et Joseph Alsberge, administrateurs sortants et rééligibles, sont réélus administrateurs."

    # Show text spans
    create_text_spans(text)

    # Generate relations for multiple entity pairs

    # Test_1
    # text = "Le soussigné, Moritz Stern, industriel, demeurant à Francfort-sur-Mein, propriétaire."
    """
    generate_relations(
        text,
        labels,
        entity_pairs=(
            (
                "Moritz Stern",
                "Francfort-sur-Mein",
                (14, 27),
                (52, 71),
                "PERSON",
                "LOCATION",
            ),
            (
                "Moritz Stern",
                "industriel",
                (14, 27),
                (28, 39),
                "PERSON",
                "OCCUPATION",
            ),
            (
                "Moritz Stern",
                "propriétaire",
                (14, 27),
                (72, 85),
                "PERSON",
                "CORPORATE_TITLE",
            ),
        ),
        num_outputs=1,
    )
    """


    text= "MM. Louis van Langenhove et Joseph Alsberge, administrateurs sortants et rééligibles, sont réélus administrateurs."
    
    generate_relations(
        text,
        labels,
        entity_pairs=(
            (
                "Louis van Langenhove",
                "administrateurs",
                (4, 24),
                (52, 71),
                "PERSON",
                "CORPORATE_POSITION",
            ),
            (
                "Joseph Alsberge",
                "administrateurs",
                (28, 44),
                (52, 71),
                "PERSON",
                "CORPORATE_POSITION",
            ),
        ),
        num_outputs=1,
    )
