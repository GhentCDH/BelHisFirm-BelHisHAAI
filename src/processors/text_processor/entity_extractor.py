import json
from functools import lru_cache
from pathlib import Path

import torch
from gliner2.joint_ie import JointIE, JointIEConfig

SCHEMA_PATH = Path(__file__).parent / "IEschema.json"



DEFAULT_MODEL = "fastino/gliner2.5-multi-v1"


def find_adapter(path):
    """ The folder with the adapter files: the folder itself, or the trainer's best/ (else final/) checkpoint inside it. """
    for candidate in (Path(path), Path(path) / "best", Path(path) / "final"):
        if (candidate / "adapter_config.json").exists():
            return candidate
    raise FileNotFoundError(f"No LoRA adapter in {path}")


@lru_cache(maxsize=2)
def get_joint(adapter=None, model_name=DEFAULT_MODEL):
    """ The Joint IE model, with a LoRA adapter (trained with gliner_train) merged into it when given. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if adapter is None:
        return JointIE.from_pretrained(model_name, device=device)

    from gliner2 import AutoExtractor
    from peft import PeftModel

    model = AutoExtractor.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, str(find_adapter(adapter))).merge_and_unload()
    return JointIE(model, device=device)


def build_schema(path=SCHEMA_PATH, adapter=None, model_name=DEFAULT_MODEL):
    """Build a JointIE schema from IEschema.json."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    # pass the full entity dicts so options such as allow_nested are applied
    schema = get_joint(adapter, model_name).create_schema().entities(spec["entities"])
    for rel in spec["relations"]:
        schema = schema.relation(
            rel["name"],
            head=rel["head"],
            tail=rel["tail"],
            description=rel.get("description"),
            directed=rel.get("directed", True),
            allow_self=rel.get("allow_self", False),
            max_per_head=rel.get("max_per_head"),
            max_per_tail=rel.get("max_per_tail"),
            threshold=rel.get("threshold"),
        )
    return schema


def extract(text, schema=None, adapter=None, model_name=DEFAULT_MODEL):
    """Jointly extract entities and relations from text. Returns a dict."""
    schema = schema or build_schema(adapter=adapter, model_name=model_name)
    result = get_joint(adapter, model_name).extract(
        text,
        schema,
        config=JointIEConfig(optimizer="beam", beam_size=32, top_k_entities=64),
    )
    out = result.to_dict()
    out["feasible"] = result.feasible
    if not result.feasible:
        print("Constraints could not be satisfied; an empty graph does not mean no facts.")
    for e in out["entities"]:
        if text[e["start"]:e["end"]] != e["text"]:
            raise ValueError(f"Span mismatch for {e['text']!r} at {e['start']}:{e['end']}")
    return out


RESET = "\033[0m"
# ANSI 256-color backgrounds, cycled per entity type
PALETTE = [110, 150, 180, 217, 183, 116, 222, 151, 189, 174]


def visualize(text, result):
    """Print text with a colored background around each entity, plus a legend and relations."""
    types = sorted({e["type"] for e in result["entities"]})
    color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}

    out, pos = [], 0
    for e in sorted(result["entities"], key=lambda e: e["start"]):
        if e["start"] < pos:  # skip overlapping spans
            continue
        out.append(text[pos:e["start"]])
        out.append(f"\033[48;5;{color[e['type']]}m\033[30m{text[e['start']:e['end']]}")
        out.append(f" {e['type']}{RESET}")
        pos = e["end"]
    out.append(text[pos:])
    print("".join(out))

    print("\nLegend: " + "  ".join(f"\033[48;5;{c}m\033[30m {t} {RESET}" for t, c in color.items()))

    by_id = {e["id"]: e for e in result["entities"]}
    for r in result["relations"]:
        h, t = by_id[r["head"]], by_id[r["tail"]]
        print(f"{h['text']} --{r['type']}--> {t['text']}  ({r['confidence']:.2f})")


if __name__ == "__main__":
    sample = "Devant nous, Jules Jacqué, notaire, de résidence à Bruges, et en présence de MM. Léon De Meulemeester et Louis De Langhe, demeurant tous les deux à Bruges, témoins instrumentaires, s'est réunie l'assemblée générale extraordinaire des actionnaires de la Société anonyme des Pêcheries à Vapeur, ayant son siège à Bruges, constituée le 13 juin 1896, par acte du ministère du notaire soussigné, et modifiée par acte du ministère du même notaire en date du 21 juin 1897, dûment publiés."
    result = extract(sample)
    visualize(sample, result)
    print("\nFull result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
