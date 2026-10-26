import cv2 as cv
import numpy as np
from PIL import Image

READING_COLUMN_ORDER = {"single": 0, "left": 0, "spanning": 0, "right": 1}
DUPLICATE_OVERLAP = 0.7


def _overlap_of_smaller(a: list, b: list) -> float:
    """ Share of the smaller box that lies inside the other box, so a box fully inside another scores 1.0. """
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / smaller if smaller > 0 else 0.0


class OCRProcessor:
    """ Transcribes YOLO-detected text blocks with the shared GLM-OCR engine, one call per block. """

    BLOCK_OCR_PROMPT = "Transcribe all text in this image exactly as printed, preserving line breaks."
    BLOCK_PADDING = 5
    MAX_NEW_TOKENS = 2048

    def __init__(self, glm_ocr):
        """ Args: glm_ocr (GLMOCREngine): Shared GLM-OCR engine, injected so the model is only loaded once. """
        self.glm_ocr = glm_ocr

        # Spine detection parameters
        self.spine_vertical_margin = 200
        self.spine_margin = 300


    def find_spine_position(self, image: Image.Image) -> int | None:
        """Find the vertical spine/split position in a two-column image."""
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape
        half_w = w // 2

        top = min(self.spine_vertical_margin, h // 2)
        bottom = max(h - self.spine_vertical_margin, h // 2)
        cropped = gray[top:bottom, :]

        # Extract a vertical strip around the center
        left_bound = max(0, half_w - self.spine_margin)
        right_bound = min(w, half_w + self.spine_margin)
        center_strip = cropped[:, left_bound:right_bound]

        # Threshold to find dark pixels (spine is usually dark)
        thresholded = cv.threshold(center_strip, 180, 255, cv.THRESH_BINARY_INV)[1]

        # Sum vertically to find the column with most dark pixels
        vertical_sum = np.sum(thresholded, axis=0)

        # Check if there's a significant dark line (spine)
        max_darkness = np.max(vertical_sum)
        mean_darkness = np.mean(vertical_sum)

        # Only split if there's a clear dark line (max is significantly above mean)
        if max_darkness > mean_darkness * 1.5:
            spine_offset = np.argmax(vertical_sum)
            return left_bound + spine_offset
        return None

    @staticmethod
    def _drop_overlapping(regions: list[dict]) -> list[dict]:
        """ Keeps the most confident of any regions that overlap heavily, so the same text is not OCR'd twice. """
        kept = []
        for region in sorted(regions, key=lambda r: r.get("confidence", 0), reverse=True):
            if all(_overlap_of_smaller(region["bbox"], other["bbox"]) < DUPLICATE_OVERLAP for other in kept):
                kept.append(region)
        return kept

    def column_of(self, bbox: list, spine: int | None) -> str:
        x1, _, x2, _ = bbox
        if spine is None:
            return "single"
        if x1 < spine < x2:
            return "spanning"
        return "left" if (x1 + x2) / 2 < spine else "right"

    def process_text_blocks(self, image: Image.Image, text_regions: list[dict]) -> dict:
        """ OCRs each text block in reading order (left column top to bottom, then right column).

            Args: image (Image.Image): Page image the regions were detected on.
            Args: text_regions (list[dict]): Regions from VisionAnalyzer.detect_layout, with 'bbox' and 'label'.

            Returns: dict with 'blocks' (bbox, text, column, label per block) and 'spine_position'.
        """
        spine = self.find_spine_position(image)

        text_regions = self._drop_overlapping(text_regions)

        ordered = sorted(
            ({**region, "column": self.column_of(region["bbox"], spine)} for region in text_regions),
            key=lambda r: (READING_COLUMN_ORDER[r["column"]], r["bbox"][1]),
        )

        blocks = []
        for region in ordered:
            x1, y1, x2, y2 = (int(c) for c in region["bbox"])
            pad = self.BLOCK_PADDING
            crop_box = (max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad))

            text = self.glm_ocr.ocr(image.crop(crop_box), prompt=self.BLOCK_OCR_PROMPT, max_new_tokens=self.MAX_NEW_TOKENS)
            if not text.strip():
                continue

            blocks.append({"bbox": [x1, y1, x2, y2], "text": text, "column": region["column"], "label": region["label"]})

        return {"blocks": blocks, "spine_position": int(spine) if spine is not None else None}
