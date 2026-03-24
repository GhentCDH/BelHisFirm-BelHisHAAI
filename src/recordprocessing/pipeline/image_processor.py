import numpy as np
import cv2 as cv

from recordprocessing.data import ConfigParameter

class ImageProcessor:

    @staticmethod
    def find_spine_position(config: ConfigParameter, image_array: np.ndarray) -> int | None:
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

        top = min(config.spine_vertical_margin, h // 2)
        bottom = max(h - config.spine_vertical_margin, h // 2)
        cropped = gray[top:bottom, :]

        # Extract a vertical strip around the center
        left_bound = max(0, half_w - config.spine_margin)
        right_bound = min(w, half_w + config.spine_margin)
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