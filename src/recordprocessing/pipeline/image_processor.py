import numpy as np
import cv2 as cv
import PIL.Image as Image

from logging import getLogger

from PIL import ImageDraw

logger = getLogger(__name__)

from src.recordprocessing.data import ConfigParameter

class ImageProcessor:

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
        w, h = masked.size
        header_y = header_bbox[1]
        side = meta.get("side", "UNKNOWN")
        halfline = meta.get("halfline")

        if halfline is None or side in ("MIDDLE", "UNKNOWN"):
            if direction == "above":
                draw.rectangle([0, 0, w, header_y], fill="white")
            else:
                draw.rectangle([0, header_y, w, h], fill="white")
        elif side == "LEFT":
            if direction == "above":
                draw.rectangle([0, 0, halfline, header_y], fill="white")
            else:
                draw.rectangle([0, header_y, halfline, h], fill="white")
                draw.rectangle([halfline, 0, w, h], fill="white")
        elif side == "RIGHT":
            if direction == "above":
                draw.rectangle([0, 0, halfline, h], fill="white")
                draw.rectangle([halfline, 0, w, header_y], fill="white")
            else:
                draw.rectangle([halfline, header_y, w, h], fill="white")

        return masked