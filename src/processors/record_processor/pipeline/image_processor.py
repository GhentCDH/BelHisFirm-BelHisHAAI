import numpy as np
import cv2 as cv
import PIL.Image as Image

from logging import getLogger

from PIL import ImageDraw

logger = getLogger(__name__)

from src.processors.record_processor.data import ConfigParameter

class ImageProcessor:

    # A region that is partly masked is kept when at least this share of it stays visible
    MIN_VISIBLE_SHARE = 0.2

    def __init__(self, config: ConfigParameter):
        self.config = config

    def find_spine_position(self, image_array: np.ndarray) -> int | None:
        """ Detects the approximate horizontal position of the spine in a page image.

            Args: config (ConfigParameter): Configuration including margins and strip width.
            Args: image_array (np.ndarray): Input image as a grayscale or color array.

            Returns: X-coordinate of the spine if detected, otherwise None.
        """

        if len(image_array.shape) != 2:
            gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
        else:
            gray = image_array

        h, w = gray.shape
        half_w = w // 2

        top = min(self.config.spine_vertical_margin, h // 2)
        bottom = max(h - self.config.spine_vertical_margin, h // 2)
        cropped = gray[top:bottom, :]

        # Extract a vertical strip around the center
        left_bound = max(0, half_w - self.config.spine_margin)
        right_bound = min(w, half_w + self.config.spine_margin)
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
        else:
            return None

    def which_half_is_bbox_on(self, bbox: list, image: Image.Image) -> dict:
        """ Determines which side of an image a bounding box is on.

            Args: config (ConfigParameter): Pipeline configuration object.
            Args: bbox (list): Bounding box to be checked.
            Args: image (Image.Image): Input image as a grayscale or color array.

            Returns: dictionary containing bounding box side and halfline position (if present).
        """

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        image_array = np.array(image)
        halfline = self.find_spine_position(image_array=image_array)

        if halfline is None:
            logger.warning("No spine detected, cannot determine bbox side")
            return {"side": "UNKNOWN", "halfline": None}

        # Check if bbox spans across the halfline
        if x1 < halfline < x2:
            meta = {"side": "MIDDLE", "halfline": halfline}
            return meta
        elif bbox_center_x < halfline:
            meta = {"side": "LEFT", "halfline": halfline}
            return meta
        else:
            meta = {"side": "RIGHT", "halfline": halfline}
            return meta

    @classmethod
    def mask_image(cls, image: Image.Image, header_bbox: list, meta: dict, direction: str) -> Image.Image:
        """ Mask irrelevant parts of a page image based on a header's position and reading direction.

            Args: image (Image.Image): Input image to be masked.
            Args: header_bbox (list): Bounding box of the header in the form [x1, y1, x2, y2].
            Args: meta (dict): Metadata about the header's location, for example: {"side": "LEFT"|"RIGHT"|"MIDDLE"|"UNKNOWN", "halfline": int | None}.
            Args: direction (str): Determines which portion to mask:
                - "above": masks content before the header (useful for starting a record)
                - "below": masks content after the header (useful for ending a record)

            Returns: A new PIL image with irrelevant regions masked (filled with white).
        """

        masked = image.copy()
        draw = ImageDraw.Draw(masked)

        masked_rectangles, _ = cls.mask_layout(masked.size, header_bbox, meta, direction)
        for rectangle in masked_rectangles:
            draw.rectangle(rectangle, fill="white")

        return masked

    @staticmethod
    def mask_layout(image_size: tuple, header_bbox: list, meta: dict, direction: str) -> tuple[list, list]:
        """ How mask_image divides a page for a header, see mask_image for the arguments.

            Returns: The rectangles [x1, y1, x2, y2] that get masked, and the rectangles that stay visible.
        """

        w, h = image_size
        header_y = header_bbox[1]
        side = meta.get("side", "UNKNOWN")
        halfline = meta.get("halfline")

        if halfline is None or side in ("MIDDLE", "UNKNOWN"):
            above, below = [0, 0, w, header_y], [0, header_y, w, h]
            return ([above], [below]) if direction == "above" else ([below], [above])

        left_above, left_below = [0, 0, halfline, header_y], [0, header_y, halfline, h]
        right_above, right_below = [halfline, 0, w, header_y], [halfline, header_y, w, h]
        left, right = [0, 0, halfline, h], [halfline, 0, w, h]

        if side == "LEFT":
            if direction == "above":
                return [left_above], [left_below, right]
            return [left_below, right], [left_above]

        if direction == "above":
            return [left, right_above], [right_below]
        return [right_below], [left, right_above]

    @classmethod
    def mask_regions(cls, regions: list[dict], image_size: tuple, header_bbox: list, meta: dict, direction: str) -> list[dict]:
        """ The layout regions that are left after mask_image with the same arguments, so the
            layout of a masked page does not need a second detection.

            Returns: The regions cut down to their visible part, without the ones that are (almost) fully masked.
        """

        masked_rectangles, visible_rectangles = cls.mask_layout(image_size, header_bbox, meta, direction)

        def intersection(a: list, b: list) -> list | None:
            x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None

        def area(box: list) -> int:
            return (box[2] - box[0]) * (box[3] - box[1])

        visible_regions = []
        for region in regions:
            bbox = region["bbox"]
            if not any(intersection(bbox, rectangle) for rectangle in masked_rectangles):
                visible_regions.append(region)
                continue

            for rectangle in visible_rectangles:
                visible_part = intersection(bbox, rectangle)
                if visible_part and area(visible_part) >= cls.MIN_VISIBLE_SHARE * area(bbox):
                    visible_regions.append({**region, "bbox": [int(c) for c in visible_part]})

        return visible_regions
