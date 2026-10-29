"""Flag scanned pages where the YOLO layout model missed text: measure the ink that lies outside every predicted box.

Classical, no training. Per page: binarize ink (adaptive threshold) -> drop scan edges and long thin rules ->
drop ink under the (padded) predicted boxes -> merge what is left into line/block candidates -> score them.
All sizes are fractions of the working image (the page is rescaled to --work-width first), so any scan size works.

Predictions: one YOLO .txt per image (same stem), lines `class xc yc w h` or `class conf xc yc w h`, normalized (only the last four columns are used).
Use --pred-format ultralytics-json / coco-json if you have a single JSON file instead.

Outputs in --out: results.csv, overlays/<page>.png for flagged pages (all pages with --debug-all),
and with --labels also pr_curve.png.

Usage:
    python -m src.checks.uncovered_ink --images datasets/Staatsblad_preanno_1/images \\
        --preds datasets/Staatsblad_preanno_1/labels --out output/uncovered_ink --sample 5

Extending the ignore mask (non-text that would otherwise look like missed ink): add another boolean mask in
`ignore_mask` and OR it in, e.g. a halftone/picture mask from a large closing, a fixed header band, or a mask
from your own non-text classes (Picture, Table) by treating those predicted boxes as covered, which already works.
"""

import argparse
import json
import random

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class Config:
    work_width: int = 1400              # pages are rescaled to this width; everything below is relative to it
    box_pad: float = 6 / 1400           # padding around predicted boxes, fraction of width (~6 px at 1400)
    margin: float = 0.05                # outer margin ignored (scan edges), fraction of each side
    thresh_block: float = 0.03          # adaptive threshold window, fraction of width
    thresh_c: int = 12                  # adaptive threshold offset: larger = less noise, fainter ink lost
    clahe_clip: float = 2.0             # CLAHE contrast limit before thresholding, 0 = off (faint print gains, noise too)
    clahe_tile: int = 8                 # CLAHE grid: tile_grid x tile_grid regions
    rule_len: float = 0.12              # min length of a ruled line, fraction of width/height
    rule_thick: float = 0.004           # lines up to this thickness (fraction of width) are rules; ink is ignored there
    merge_x: float = 0.03               # horizontal dilation when merging ink into lines, fraction of width
    merge_y: float = 0.004              # vertical dilation, fraction of width
    min_ink: float = 0.0002             # components with less ink than this fraction of the page are noise
    min_density: float = 0.03           # components whose ink fills less of their bbox than this are sparse edge/paper noise
    min_width: float = 0.04             # components narrower than this fraction of width are stray marks
    min_component_frac: float = 0.004   # flag if largest component's ink / page area >= this
    min_total_frac: float = 0.012       # flag if total uncovered ink / page area >= this


def load_boxes(pred_path: Path, w: int, h: int) -> np.ndarray:
    """ Pixel xyxy boxes (n, 4) from a YOLO .txt file; empty array when there is none. """

    if not pred_path.exists():
        return np.zeros((0, 4))
    rows = [list(map(float, ln.split())) for ln in pred_path.read_text().splitlines() if ln.strip()]
    boxes = [(r[-4], r[-3], r[-2], r[-1]) for r in rows]  # xc yc w h are always the last four columns
    return xywhn_to_xyxy(boxes, w, h)


def xywhn_to_xyxy(boxes, w: int, h: int) -> np.ndarray:
    b = np.array(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], axis=1)


def load_json_boxes(path: Path, fmt: str) -> dict[str, list]:
    """ stem -> list of pixel xyxy boxes, from an Ultralytics results JSON or a COCO JSON (xywh pixels). """

    data = json.loads(path.read_text())
    out: dict[str, list] = {}
    if fmt == "coco-json":
        names = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
        for a in data["annotations"]:
            x, y, bw, bh = a["bbox"]
            out.setdefault(names[a["image_id"]], []).append([x, y, x + bw, y + bh])
    else:  # ultralytics-json (model.predict(...).tojson()): [{"name", "box": {x1,y1,x2,y2}}], image name in "image"/"file"
        for d in data:
            b = d["box"]
            out.setdefault(Path(d.get("image", d.get("file", ""))).stem, []).append([b["x1"], b["y1"], b["x2"], b["y2"]])
    return out


def binarize(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """ Ink = darker than its local neighbourhood; copes with uneven, yellowed paper. """

    block = int(cfg.thresh_block * gray.shape[1]) | 1
    if cfg.clahe_clip > 0:
        gray = cv2.createCLAHE(cfg.clahe_clip, (cfg.clahe_tile, cfg.clahe_tile)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.thresh_c)


def ignore_mask(ink: np.ndarray, cfg: Config) -> np.ndarray:
    """ Non-text: the outer margin plus long thin horizontal/vertical rules (page border, separators). """

    h, w = ink.shape
    mask = np.zeros_like(ink)
    m = int(cfg.margin * min(h, w))
    mask[:m], mask[-m:], mask[:, :m], mask[:, -m:] = 255, 255, 255, 255
    for kernel in ((int(cfg.rule_len * w), 1), (1, int(cfg.rule_len * h))):
        # opening with a long 1-px kernel keeps only ink runs at least that long, i.e. rules
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        mask |= lines
    # rules are blurred/thicker than their core: grow the mask a little so rule edges don't survive as ink
    t = max(3, int(cfg.rule_thick * w))
    return cv2.dilate(mask, np.ones((t, t), np.uint8))


def padded_rects(boxes: np.ndarray, pad: int, w: int, h: int) -> list[tuple]:
    """ Predicted boxes grown by pad and clipped to the image, as half-open integer (x1, y1, x2, y2). """

    return [(max(int(x1) - pad, 0), max(int(y1) - pad, 0), min(int(x2) + pad + 1, w), min(int(y2) + pad + 1, h))
            for x1, y1, x2, y2 in boxes]


def covered_mask(shape: tuple, rects: list[tuple]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in rects:
        mask[y1:y2, x1:x2] = 255
    return mask


def free_rects(mask: np.ndarray, region: tuple, rects: list[tuple]) -> list[tuple]:
    """ Tight bounding rectangles of the mask pixels inside region, split (guillotine cuts around the first box
    they hit, recursively) until no rectangle intersects any of the boxes. Pieces are disjoint. """

    x0, y0, x1, y1 = region
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return []
    x0, y0, x1, y1 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
    hit = next(((bx1, by1, bx2, by2) for bx1, by1, bx2, by2 in rects
                if bx1 < x1 and bx2 > x0 and by1 < y1 and by2 > y0), None)
    if hit is None:
        return [(int(x0), int(y0), int(x1), int(y1))]
    bx1, by1, bx2, by2 = hit
    # two ways to cut around the box: full-width strips above/below, or full-height strips left/right.
    # Keep the one whose largest part is bigger, so a column of text beside a box stays in one piece.
    cx0, cx1, cy0, cy1 = max(x0, bx1), min(x1, bx2), max(y0, by1), min(y1, by2)
    horizontal = [(x0, y0, x1, cy0), (x0, cy1, x1, y1), (x0, cy0, cx0, cy1), (cx1, cy0, x1, cy1)]
    vertical = [(x0, y0, cx0, y1), (cx1, y0, x1, y1), (cx0, y0, cx1, cy0), (cx0, cy1, cx1, y1)]
    area = lambda r: max(r[2] - r[0], 0) * max(r[3] - r[1], 0)
    parts = max(horizontal, vertical, key=lambda ps: max(map(area, ps)))
    return [r for part in parts if part[2] > part[0] and part[3] > part[1] for r in free_rects(mask, part, rects)]


def count_lines(residual: np.ndarray, box: tuple) -> int:
    """ Text lines in a component: runs of rows whose ink exceeds a quarter of the median non-empty row. """

    x, y, w, h = box
    rows = residual[y:y + h, x:x + w].sum(axis=1) / 255.0
    active = rows > 0.25 * max(np.median(rows[rows > 0]) if (rows > 0).any() else 0, 1)
    return int(np.sum(active[1:] & ~active[:-1]) + active[0])


def analyse(img: np.ndarray, boxes_px: np.ndarray, scale: float, cfg: Config) -> dict:
    """ boxes_px are in the original image's pixels; returns the scores, components and masks (working scale). """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    boxes = boxes_px * scale

    ink = binarize(gray, cfg)
    rects = padded_rects(boxes, int(round(cfg.box_pad * w)), w, h)
    covered = covered_mask(ink.shape, rects)
    residual = ink & ~covered & ~ignore_mask(ink, cfg)
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # specks

    kx, ky = int(cfg.merge_x * w) | 1, int(cfg.merge_y * w) | 1
    merged = cv2.dilate(residual, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))) & ~covered  # never bridge into a box
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, bw, bh = stats[i][:4]
        mask = (labels == i).astype(np.uint8)
        for x1, y1, x2, y2 in free_rects(mask, (x, y, x + bw, y + bh), rects):
            bw, bh = x2 - x1, y2 - y1
            ink_px = int(np.count_nonzero(residual[y1:y2, x1:x2] & mask[y1:y2, x1:x2] * 255))
            if ink_px < cfg.min_ink * w * h or bw < cfg.min_width * w or ink_px < cfg.min_density * bw * bh:
                continue
            comps.append(dict(bbox=(x1, y1, bw, bh), area=bw * bh, ink=ink_px,
                              lines=count_lines(residual, (x1, y1, bw, bh)), frac=ink_px / (w * h)))
    total = sum(c["ink"] for c in comps) / (w * h)
    top = max((c["frac"] for c in comps), default=0.0)
    return dict(comps=comps, total=total, top=top, boxes=boxes, rects=rects, size=(w, h), gray=gray, residual=residual)


def is_flagged(total: float, top: float, cfg: Config) -> bool:
    return top >= cfg.min_component_frac or total >= cfg.min_total_frac


def save_overlay(res: dict, path: Path, flagged: bool) -> None:
    img = cv2.cvtColor(res["gray"], cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(img)
    tint[res["residual"] > 0] = (0, 0, 255)
    img = cv2.addWeighted(img, 1, tint, 0.6, 0)  # uncovered ink in red
    for x1, y1, x2, y2 in res["boxes"].astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 140, 0), 2)  # predicted boxes: blue (BGR)
    for c in res["comps"]:
        x, y, bw, bh = c["bbox"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(img, f"{c['frac']*100:.2f}% ~{c['lines']} lines", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    banner = f"{'FLAGGED' if flagged else 'ok'}  max={res['top']*100:.2f}%  total={res['total']*100:.2f}%"
    cv2.putText(img, banner, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if flagged else (0, 140, 0), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def evaluate(df: pd.DataFrame, labels_csv: Path, cfg: Config, out: Path) -> None:
    """ Precision/recall at the current thresholds, then a sweep that scales both thresholds together. """

    lab = pd.read_csv(labels_csv)[["filename", "missed"]]
    d = df.merge(lab, on="filename")
    if d.empty:
        print("No filenames in --labels match the processed pages.")
        return

    def pr(factor: float):
        flag = (d.max_component_frac >= cfg.min_component_frac * factor) | (d.total_uncovered_frac >= cfg.min_total_frac * factor)
        tp = int((flag & (d.missed == 1)).sum())
        return tp / max(int(flag.sum()), 1), tp / max(int((d.missed == 1).sum()), 1), int(flag.sum())

    p, r, nf = pr(1.0)
    print(f"\nAt current thresholds ({len(d)} labelled pages): precision={p:.2f} recall={r:.2f} flagged={nf}")
    rows = [(f, *pr(f)) for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8)]
    print("\nthreshold x | min_component | min_total | precision | recall | flagged")
    for f, p, r, nf in rows:
        print(f"{f:>11} | {cfg.min_component_frac*f:>13.4f} | {cfg.min_total_frac*f:>9.4f} | {p:>9.2f} | {r:>6.2f} | {nf:>7}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r for _, _, r, _ in rows], [p for _, p, _, _ in rows], "o-")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02), title="Missed-text flagging")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--preds", required=True, type=Path, help="Folder of YOLO .txt files, or one JSON file")
    parser.add_argument("--pred-format", choices=["yolo", "ultralytics-json", "coco-json"], default="yolo")
    parser.add_argument("--out", type=Path, default=Path("output/uncovered_ink"))
    parser.add_argument("--labels", type=Path, help="CSV with columns filename, missed (0/1)")
    parser.add_argument("--debug-all", action="store_true", help="Save an overlay for every page, not only flagged ones")
    parser.add_argument("--sample", type=int, default=0, help="Process N random pages (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    for name, default in asdict(Config()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config())})

    pages = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.sample:
        pages = random.Random(args.seed).sample(pages, min(args.sample, len(pages)))
    json_boxes = load_json_boxes(args.preds, args.pred_format) if args.pred_format != "yolo" else None

    rows = []
    for page in pages:
        img = cv2.imread(str(page))
        if img is None:
            print(f"skip unreadable {page.name}")
            continue
        h0, w0 = img.shape[:2]
        if json_boxes is not None:
            boxes = np.array(json_boxes.get(page.stem, []), dtype=float).reshape(-1, 4)
        else:
            boxes = load_boxes(args.preds / f"{page.stem}.txt", w0, h0)
        res = analyse(img, boxes, cfg.work_width / w0, cfg)
        flagged = is_flagged(res["total"], res["top"], cfg)
        rows.append(dict(filename=page.name, n_boxes=len(boxes), total_uncovered_frac=round(res["total"], 5),
                         max_component_frac=round(res["top"], 5), n_components=len(res["comps"]), flagged=flagged))
        if flagged or args.debug_all:
            save_overlay(res, args.out / "overlays" / f"{page.stem}.png", flagged)
        print(f"{page.name}: max={res['top']:.4f} total={res['total']:.4f} comps={len(res['comps'])} {'FLAG' if flagged else ''}")

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results.csv", index=False)
    print(f"\nPages processed: {len(df)}\nPages flagged:   {int(df.flagged.sum())}\nResults: {args.out / 'results.csv'}")
    if args.labels:
        evaluate(df, args.labels, cfg, args.out)


if __name__ == "__main__":
    main()
