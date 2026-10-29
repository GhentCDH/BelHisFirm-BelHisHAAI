import json
from pathlib import Path

from gliner2.joint_ie import JointIE, JointIEConfig

SCHEMA_PATH = Path(__file__).parent / "IEschema.json"

joint = JointIE.from_pretrained("fastino/gliner2.5-multi-v1")


def build_schema(path=SCHEMA_PATH):
    """Build a JointIE schema from IEschema.json."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = joint.create_schema().entities(
        {name: e["description"] for name, e in spec["entities"].items()}
    )
    for rel in spec["relations"]:
        schema = schema.relation(
            rel["name"],
            head=rel["head"],
            tail=rel["tail"],
            directed=rel.get("directed", True),
            allow_self=rel.get("allow_self", False),
            max_per_head=rel.get("max_per_head"),
            threshold=rel.get("threshold"),
        )
    return schema


def extract(text, schema=None):
    """Jointly extract entities and relations from text. Returns a dict."""
    schema = schema or build_schema()
    result = joint.extract(
        text, schema, config=JointIEConfig(optimizer="beam", beam_size=32)
    )
    if not result.feasible:
        print("Constraints could not be satisfied.")
    return result.to_dict()


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
