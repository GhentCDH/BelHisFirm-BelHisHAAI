from PIL import Image


def find_caption_above(table_box, caption_boxes: list, max_gap: int, min_overlap: float):
    """Return the (x0, y0, x1, y1) of the caption box closest above table_box, if any qualifies.

    Ported verbatim from BelHisFirm---HisTableFinder's crop_tables.py.
    """
    tx0, ty0, tx1, _ = table_box
    best = None
    best_gap = None
    for cx0, cy0, cx1, cy1 in caption_boxes:
        overlap = min(tx1, cx1) - max(tx0, cx0)
        if overlap <= 0:
            continue
        min_width = min(tx1 - tx0, cx1 - cx0)
        if min_width <= 0 or overlap / min_width < min_overlap:
            continue
        gap = ty0 - cy1
        if gap < -5 or gap > max_gap:
            continue
        if best_gap is None or gap < best_gap:
            best = (cx0, cy0, cx1, cy1)
            best_gap = gap
    return best


class TableCropper:
    """Builds in-memory table crops from a page image and its detected
    Table/Caption regions (as returned by VisionAnalyzer.get_excluded_regions),
    extending each table crop to include a directly-above caption, if any."""

    def __init__(self, table_config):
        self.table_config = table_config

    def build_crops(self, image: Image.Image, table_regions: list[dict], caption_regions: list[dict]) -> list[dict]:
        """Returns [{"bbox": [x0, y0, x1, y1], "image": PIL.Image crop}, ...], one per table region."""
        caption_boxes = [tuple(r["bbox"]) for r in caption_regions]
        crops = []
        for region in table_regions:
            table_box = tuple(region["bbox"])
            caption_box = find_caption_above(
                table_box, caption_boxes, self.table_config.caption_max_gap, self.table_config.caption_min_overlap
            )

            x0, y0, x1, y1 = table_box
            if caption_box is not None:
                cx0, cy0, cx1, cy1 = caption_box
                x0, y0 = min(x0, cx0), min(y0, cy0)
                x1, y1 = max(x1, cx1), max(y1, cy1)

            margin = self.table_config.crop_margin
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(image.width, x1 + margin)
            y1 = min(image.height, y1 + margin)
            if x1 <= x0 or y1 <= y0:
                continue

            crops.append({"bbox": [x0, y0, x1, y1], "image": image.crop((x0, y0, x1, y1))})
        return crops
