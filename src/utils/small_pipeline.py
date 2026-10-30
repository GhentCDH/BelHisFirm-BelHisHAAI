""" Small end-to-end test of the models on page images: YOLO layout, GLM-OCR on the text blocks, then the Joint IE
    entity extractor (entities and relations) with a LoRA adapter, shown as a page with the boxes and the text with
    its entities highlighted. The models are loaded one after the other, so they do not have to fit in memory together.

    uv run python -m src.utils.small_pipeline path/to/page.tif
    uv run python -m src.utils.small_pipeline path/to/folder_of_pages --max-pages 5
    uv run python -m src.utils.small_pipeline path/to/page.tif --adapter model/gliner/variants/v25_single_original
    uv run python -m src.utils.small_pipeline path/to/page.tif --adapter none      # plain GLiNER, no adapter

    A folder of page images is one record, a pipeline output folder gives one report per record folder, all their pages
    unless --max-pages says otherwise.

    Output (output/ is gitignored), per record in output/small_pipeline/<record>/:
        report.html        open it in a browser: the entities of the whole record, then every page
        *_blocks.jpg       the scan with the detected blocks (red where entities were found)
        *_text.jpg         the text on a white page, where it stands on the scan, entities highlighted
        result.json        the OCR text, entities and relations per block
"""

import argparse
import colorsys
import html
import re
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.processors.text_processor.entity_extractor import DEFAULT_MODEL, SCHEMA_PATH, build_schema, extract, visualize
from src.utils.gliner_utils.blocks import TEXT_LABELS, build_blocks, load_domains
from src.utils.gliner_utils.ocr_input import discover, run_ocr, select_pages

logger = logging.getLogger(__name__)

V2_MODEL = "fastino/gliner2-multi-v1"
DEFAULT_ADAPTER = Path("model/gliner/variants/v2_single_combined")
# Colors per domain, light enough for dark text on top
COLORS = {"date": "#f5d76e", "organisation": "#e0a8ec", "legal": "#f2a1a1", "person": "#9cc7f2", "share": "#8fdcd0",
          "location": "#a6e0a1"}
OTHER_COLOR = "#cccccc"
BOX_COLOR, HIT_COLOR = (150, 150, 150), (220, 40, 40)


def model_for(adapter: Path | None) -> str:
    """ The checkpoint an adapter was trained on, from the variant name (v25_... is GLiNER2.5, anything else GLiNER2). """
    return DEFAULT_MODEL if adapter and adapter.name.startswith("v25") else V2_MODEL


def draw_blocks(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """ The page with a numbered box around every block, red where entities were found in it. """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 400)
    for number, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = block["bbox"]
        color = HIT_COLOR if block["entities"] else BOX_COLOR
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw.text((x1 + 4, y1 + 4), str(number), fill=color, font_size=max(16, image.width // 60))
    return image


FONT_FILES = ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
              "DejaVuSerif.ttf")
LINE_HEIGHT = 1.25
MIN_FONT = 7
SENTENCE_END = (".", ":", "!", "?", ";")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_FILES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def type_colors(domains: dict[str, list[str]]) -> dict[str, str]:
    """ A color per entity type: the color of its domain, in a lighter or darker shade (and a slightly shifted hue) for
        every type of the domain, so the types of one domain can still be told apart. """
    colors = {}
    for domain, labels in domains.items():
        hue, _, saturation = colorsys.rgb_to_hls(*(c / 255 for c in rgb(COLORS.get(domain, OTHER_COLOR))))
        for number, label in enumerate(labels):
            position = number / max(len(labels) - 1, 1)
            red, green, blue = colorsys.hls_to_rgb((hue + 0.03 * (position - 0.5)) % 1, 0.88 - 0.30 * position, saturation)
            colors[label] = "#{:02x}{:02x}{:02x}".format(round(red * 255), round(green * 255), round(blue * 255))
    return colors


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def wrap_block(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[tuple[int | None, str]]:
    """ The lines of a block as (offset of the line in the text, line), the OCR line breaks kept and long lines wrapped at
        the width. A white line (None, "") is put between paragraphs: after a line that ends a sentence and is clearly
        shorter than the others, when the next line starts with a capital or a number. """
    lines, offset = [], 0
    for raw in text.split("\n"):
        current, start = "", offset
        for word_match in re.finditer(r"\S+", raw):
            word = word_match.group()
            if current and font.getlength(f"{current} {word}") > width:
                lines.append((start, current))
                current, start = "", offset + word_match.start()
            if not current:
                start = offset + word_match.start()
            current = f"{current} {word}" if current else word
        if current:
            lines.append((start, current))
        offset += len(raw) + 1

    longest = max((len(line) for _, line in lines), default=0)
    out = []
    for number, (start, line) in enumerate(lines):
        out.append((start, line))
        following = lines[number + 1][1] if number + 1 < len(lines) else ""
        if (following and line.endswith(SENTENCE_END) and len(line) < 0.75 * longest
                and (following[0].isupper() or following[0].isdigit())):
            out.append((None, ""))
    return out


def render_text_page(size: tuple[int, int], blocks: list[dict], colors: dict[str, str]) -> Image.Image:
    """ A white page as large as the scan with the text of every block in the place of its box, the largest font in
        which it fits, and the entities on a background in the color of their domain. A legend runs along the bottom. """
    legend_font = load_font(max(14, size[0] // 110))
    swatch = int(legend_font.size * 1.1)
    legend, x, y = [], 10, 0
    for label, color in colors.items():
        item_width = swatch + 6 + int(legend_font.getlength(label)) + 24
        if x + item_width > size[0] and x > 10:
            x, y = 10, y + int(swatch * 1.5)
        legend.append((x, y, label, color))
        x += item_width
    legend_height = y + int(swatch * 1.5) + 20
    page = Image.new("RGB", (size[0], size[1] + legend_height), "white")
    draw = ImageDraw.Draw(page)

    for block in blocks:
        x1, y1, x2, y2 = block["bbox"]
        width, height = max(x2 - x1, 1), max(y2 - y1, 1)
        font_size = int(min(height, 60))
        while True:
            font = load_font(font_size)
            lines = wrap_block(block["text"], font, width)
            if font_size <= MIN_FONT or len(lines) * font_size * LINE_HEIGHT <= height:
                break
            font_size -= 1

        entities = sorted(block["entities"], key=lambda e: e["start"] - e["end"])  # longest first, shorter on top
        for number, (offset, line) in enumerate(lines):
            if offset is None:
                continue
            y = y1 + number * font_size * LINE_HEIGHT
            for entity in entities:
                a, b = max(entity["start"], offset), min(entity["end"], offset + len(line))
                if a < b:
                    left, right = font.getlength(line[:a - offset]), font.getlength(line[:b - offset])
                    draw.rectangle((x1 + left, y, x1 + right, y + font_size * LINE_HEIGHT),
                                   fill=rgb(colors.get(entity["type"], OTHER_COLOR)))
            draw.text((x1, y), line, fill="black", font=font)

    for x, y, label, color in legend:
        top = size[1] + 10 + y
        draw.rectangle((x, top, x + swatch, top + swatch), fill=rgb(color))
        draw.text((x + swatch + 6, top), label, fill="black", font=legend_font)
    return page


def highlight_html(text: str, entities: list[dict], colors: dict[str, str]) -> str:
    """ The text as HTML with every entity in the color of its domain (the first of overlapping entities wins). """
    out, position = [], 0
    for entity in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        if entity["start"] < position:
            continue
        color = colors.get(entity["type"], OTHER_COLOR)
        out.append(html.escape(text[position:entity["start"]]))
        out.append(f'<mark style="background:{color}" title="{html.escape(entity["type"])} '
                   f'({entity.get("confidence", 0):.2f})">{html.escape(text[entity["start"]:entity["end"]])}'
                   f'<sub>{html.escape(entity["type"])}</sub></mark>')
        position = entity["end"]
    out.append(html.escape(text[position:]))
    return "".join(out)


def relations_html(result: dict) -> str:
    by_id = {e["id"]: e for e in result["entities"]}
    rows = "".join(f'<li>{html.escape(by_id[r["head"]]["text"])} <b>{html.escape(r["type"])}</b> '
                   f'{html.escape(by_id[r["tail"]]["text"])} <small>{r["confidence"]:.2f}</small></li>'
                   for r in result["relations"])
    return f"<ul>{rows}</ul>" if rows else ""


def summary_html(blocks: list[dict], domain_of: dict[str, str], colors: dict[str, str]) -> str:
    """ What was found in the whole record: per domain and entity type the distinct texts with how often they occur,
        and the relations between entities of the same block, counted over the record. """
    found: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    relations: Counter = Counter()
    for block in blocks:
        by_id = {e["id"]: e for e in block["result"]["entities"]}
        for entity in block["entities"]:
            found[domain_of.get(entity["type"], "other")][entity["type"]][" ".join(entity["text"].split())] += 1
        for r in block["result"]["relations"]:
            relations[(" ".join(by_id[r["head"]]["text"].split()), r["type"], " ".join(by_id[r["tail"]]["text"].split()))] += 1

    rows = []
    for domain in [*COLORS, "other"]:
        for label, counter in sorted(found.get(domain, {}).items()):
            values = ", ".join(f"{html.escape(text)}{f' ×{n}' if n > 1 else ''}" for text, n in counter.most_common())
            rows.append(f'<tr><td><mark style="background:{COLORS.get(domain, OTHER_COLOR)}">{domain}</mark></td>'
                        f'<td><mark style="background:{colors.get(label, OTHER_COLOR)}">{html.escape(label)}</mark></td><td>{sum(counter.values())}</td><td>{values}</td></tr>')
    facts = "".join(f"<li>{html.escape(head)} <b>{html.escape(kind)}</b> {html.escape(tail)}"
                    f"{f' ×{n}' if n > 1 else ''}</li>" for (head, kind, tail), n in sorted(relations.items(), key=lambda i: i[0][1]))
    return (f"<h2>Record</h2><table><tr><th>domain</th><th>type</th><th>n</th><th>texts</th></tr>{''.join(rows)}</table>"
            f"<h3>Relations</h3><ul>{facts}</ul>")


def write_report(path: Path, name: str, pages: list[dict], blocks: list[dict], domain_of: dict[str, str], colors: dict[str, str], settings: str) -> None:
    legend = "".join(f'<mark style="background:{color}">{label}</mark> ' for label, color in colors.items())
    body = [summary_html(blocks, domain_of, colors)]
    for page in pages:
        body.append(f'<h2>Page {page["page"] + 1}</h2><div class="page">'
                    f'<a href="{page["image"]}"><img src="{page["image"]}"></a>'
                    f'<a href="{page["text_image"]}"><img src="{page["text_image"]}"></a></div><div class="blocks">')
        for number, block in enumerate(page["blocks"], start=1):
            body.append(f'<div class="block"><span class="n">{number} · {html.escape(str(block["label"]))}</span>'
                        f'<p>{highlight_html(block["text"], block["result"]["entities"], colors)}</p>'
                        f'{relations_html(block["result"])}</div>')
        body.append("</div>")
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>{html.escape(name)}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:20px;color:#222}}
.page{{display:flex;gap:16px;align-items:flex-start}} .page img{{width:100%;border:1px solid #ccc}} .page a{{flex:1}}
.block{{border-bottom:1px solid #ddd;padding:6px 0}} .block p{{white-space:pre-wrap;margin:4px 0}}
.n{{font-size:12px;color:#777}} mark{{padding:1px 3px;border-radius:3px;color:#111}} sub{{font-size:10px;margin-left:3px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ddd;padding:3px 8px;text-align:left;vertical-align:top}}
ul{{margin:4px 0;font-size:13px}} small{{color:#888}}
</style><h1>{html.escape(name)}</h1><p>{html.escape(settings)}</p><p>{legend}</p>{''.join(body)}""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A page image, a folder of page images or a pipeline output folder")
    parser.add_argument("--max-pages", type=int, default=None, help="At most this many random pages over all records (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout weights")
    parser.add_argument("--glm-checkpoint", type=Path, default=None, help="GLM-OCR folder (default: model/glm-ocr)")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                        help="LoRA adapter folder, or 'none' for the plain model (default: %(default)s)")
    parser.add_argument("--model", default=None, help="GLiNER checkpoint (default: the one the adapter name says)")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--out", type=Path, default=Path("output/small_pipeline"))
    parser.add_argument("--print", dest="show", action="store_true", help="Also print every block in the terminal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    adapter = None if args.adapter == "none" else Path(args.adapter)
    model_name = args.model or model_for(adapter)
    if adapter and not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    sources = discover(args.input)
    if not sources:
        raise SystemExit(f"No page images found in {args.input}")
    only_pages = select_pages(sources, args.max_pages or 10 ** 9, args.seed, TEXT_LABELS)

    # 1. YOLO layout + GLM-OCR (freed again when done)
    needs_ocr = [s for s in sources if s.ocr_data is None]
    if needs_ocr:
        run_ocr(needs_ocr, args.weights, args.glm_checkpoint, TEXT_LABELS, only_pages=only_pages)

    # 2. GLiNER Joint IE on every text block, one report per record
    domains = load_domains(args.schema)
    domain_of = {label: domain for domain, labels in domains.items() for label in labels}
    colors = type_colors(domains)
    schema = build_schema(args.schema, adapter, model_name)
    settings = f"model {model_name}, adapter {adapter or 'none'}"

    for source in sources:
        picked = only_pages.get(source.name, set())
        blocks = [b for b in build_blocks(source.ocr_data, TEXT_LABELS) if b["page"] in picked]
        if not blocks:
            print(f"{source.name}: no text blocks")
            continue
        out_dir = args.out / source.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for block in blocks:
            result = extract(block["text"], schema, adapter, model_name)
            block["result"] = result
            block["entities"] = result["entities"]
            if args.show:
                print(f"\n--- {source.name} page {block['page'] + 1}, block {block['block_index']} ({block['label']})")
                visualize(block["text"], result)

        pages = []
        for page_idx in sorted({b["page"] for b in blocks}):
            page_blocks = [b for b in blocks if b["page"] == page_idx]
            image = Image.open(source.images[page_idx]).convert("RGB")
            stem = f"page_{page_idx + 1:03d}"
            draw_blocks(image, page_blocks).save(out_dir / f"{stem}_blocks.jpg", quality=85)
            render_text_page(image.size, page_blocks, colors).save(out_dir / f"{stem}_text.jpg", quality=90)
            pages.append({"page": page_idx, "image": f"{stem}_blocks.jpg", "text_image": f"{stem}_text.jpg", "blocks": page_blocks})

        saved = [{"page": b["page"] + 1, "block": b["block_index"], "label": b["label"], "text": b["text"],
                  "entities": b["entities"], "relations": b["result"]["relations"]} for b in blocks]
        (out_dir / "result.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(out_dir / "report.html", source.name, pages, blocks, domain_of, colors, settings)
        print(f"{source.name}: {len(blocks)} blocks on {len(pages)} pages, {sum(len(b['entities']) for b in blocks)} entities, "
              f"report {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
