import numpy as np
import pytesseract as tesseract

from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter

from src.recordprocessing.pipeline.header_validator import HeaderValidator
from src.recordprocessing.pipeline.image_processor import ImageProcessor
from src.recordprocessing.pipeline.result_processor import ResultProcessor

from src.recordprocessing.data import ConfigParameter, MappedPrediction

from src.recordprocessing.utils import GPUController

import logging
logger = logging.getLogger(__name__)

class VisionAnalyzer:

    yolo_model = YOLO("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/model/best.pt")

    @staticmethod
    def get_excluded_regions(config: ConfigParameter, image: Image.Image) -> list:
        """ Get regions to exclude from OCR based on layout detection.

            Args: config: (ConfigParameter): Configuration object.
            Args: image (Image.Image): Image to check for excluded regions on.

            Returns: A list of dictionaries with 'bbox' and 'label' keys.
            Configure excluded labels via the ConfigParameter object.
        """

        excluded_regions = []

        results = VisionAnalyzer.yolo_model.predict(image, stream=True)
        for result in results:
            mapped_predictions = ResultProcessor.process_result(result)

            for mapped_prediction in mapped_predictions:
                if mapped_prediction.label in config.ocr_excluded_labels:
                    excluded_regions.append({
                        "bbox": [int(c) for c in mapped_prediction.bbox],
                        "label": mapped_prediction.label,
                        "confidence": mapped_prediction.confidence
                    })
                    logger.debug(f"Excluding region: {mapped_prediction.label} at {mapped_prediction.bbox}")

        return excluded_regions

    @staticmethod
    def detect_record_headers(config: ConfigParameter, image: Image.Image) -> list | None:
        """ Uses YOLO model to detect record headers in a given image.

        Args: config: (ConfigParameter): Configuration object.
        Args: image (Image.Image): Image to be checked.

        Returns: list of dictionaries for each header with their bounding box and text.
        """

        results = VisionAnalyzer.yolo_model.predict(image, stream=True)

        if not results:
            logger.info(f"No layout predictions..")
            return None

        image_width, image_height = image.size
        verified_predictions = []

        for result in results:
            mapped_predictions = ResultProcessor.process_result(result)

            for mapped_prediction in mapped_predictions:
                if VisionAnalyzer.is_sus_table(config, mapped_prediction, image_width, image_height):
                    new_predictions = VisionAnalyzer.redetect_region(config, image, mapped_prediction.bbox,
                                                                     mapped_prediction)
                    verified_predictions.extend(new_predictions)
                else:
                    verified_predictions.append(mapped_prediction)

        record_header_predictions = [prediction for prediction in verified_predictions if HeaderValidator.is_record_header_candidate(prediction)]
        if not record_header_predictions:
            logger.info(f"No record headers found in layout predictions...")
            return None

        headers_on_page = []
        for prediction in record_header_predictions:
            bbox = [int(c) for c in prediction.bbox]
            padded_bbox = (
                max(0, bbox[0] - config.padding),
                max(0, bbox[1] - config.padding),
                min(image.width, bbox[2] + config.padding),
                min(image.height, bbox[3] + config.padding),
            )

            cropped = image.crop(padded_bbox)

            # Convert to grayscale
            cropped = cropped.convert("L")

            # Increase contrast
            enhancer = ImageEnhance.Contrast(cropped)
            cropped = enhancer.enhance(2)

            # Sharpen the image
            cropped = cropped.filter(ImageFilter.SHARPEN)

            text = tesseract.image_to_string(cropped, config="--psm 6", lang="fra")
            print(f"\n||\n{text}\n||\n")

            valid = HeaderValidator.is_valid_section_header(text)

            status = "VALID" if valid else "INVALID"
            logger.info(f'{status} recordheader found at {text[:50]}...' if len(
                text) > 50 else f"{status} recordheader found at {text}")

            if valid:
                headers_on_page.append({
                    "bbox": bbox,
                    "text": text})

        # Clear GPU memory after processing headers
        GPUController.clear_gpu_memory()

        return headers_on_page if headers_on_page else None

    @staticmethod
    def redetect_region(config: ConfigParameter, image: Image.Image, bbox: list, original_prediction: MappedPrediction) -> list:
        """ Re-analyzes a region by splitting along the spine if found, running detection on each half.

        Args: config: (ConfigParameter): Configuration object.
        Args: image (Image.Image): Full page image.
        Args: bbox (list): Region to re-analyze [x1, y1, x2, y2].
        Args: original_prediction (MappedPrediction): Prediction for the region.

        Returns: Re-mapped predictions for each half, or original if no spine detected.
        """

        x1, y1, x2, y2 = [int(c) for c in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped)

        spine_pos = ImageProcessor.find_spine_position(config, cropped_array)

        if spine_pos is None:
            # No spine found - keep original prediction
            logger.warning("No spine detected, keeping original prediction!")
            return [original_prediction]

        # Spine found - split and redetect
        logger.info(f"  Spine detected at x={spine_pos}, splitting region into two halves")
        h, w = cropped_array.shape[:2]
        left_half = cropped.crop((0, 0, spine_pos, h))
        right_half = cropped.crop((spine_pos, 0, w, h))

        images = [left_half, right_half]
        offsets = [0, spine_pos]

        mapped_predictions = []

        for image, region_x_offset in zip(images, offsets):
            results = VisionAnalyzer.yolo_model.predict(image, stream=True)

            for result in results:
                mapped_prediction = ResultProcessor.process_result(result, x1 + region_x_offset, y1)

                mapped_predictions.append(mapped_prediction)

        return mapped_predictions

    @staticmethod
    def is_sus_table(config: ConfigParameter, prediction: MappedPrediction, image_width: int, image_height: int) -> bool:
        """ Detects if a prediction spans more than half a page, this could mean that it overrides headers.

        Args: config: (ConfigParameter): Configuration object.
        Args: prediction: (MappedPrediction): prediction made by computer vision model.
        Args: image_width: (int): width of image
        Args: image_height: (int): height of image

        Returns: Is sus table or not.
        """

        """ TEMPORARY HARDCODE """
        if prediction.label != "Table":
            return False

        if prediction.confidence >= config.sus_table_confidence_threshold:
            return False

        bbox = prediction.bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        area_fraction = bbox_area / image_area

        if area_fraction <= config.sus_table_area_threshold:
            logger.info(f"Table detection check passed!")
            return False
        else:
            logger.info(f"Sus table detected: conf={prediction.confidence:.2f}, area_fraction={area_fraction:.2f}")
            return True