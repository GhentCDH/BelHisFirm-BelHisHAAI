import numpy as np

from ultralytics import YOLO
from PIL import Image

from src.processors.record_processor.pipeline.header_validator import HeaderValidator
from src.processors.record_processor.pipeline.image_processor import ImageProcessor
from src.processors.record_processor.pipeline.result_processor import ResultProcessor

from src.processors.record_processor.data import ConfigParameter, MappedPrediction

from src.processors.record_processor.utils import GPUController

import logging
logger = logging.getLogger(__name__)

class VisionAnalyzer:

    HEADER_OCR_PROMPT = "Transcribe the text in this image exactly as printed, including any numbers and punctuation."

    def __init__(self, config: ConfigParameter, yolo_model_file_path: str, glm_ocr):
        """ Args: glm_ocr (GLMOCREngine): Shared GLM-OCR engine, injected so the model is only loaded once. """

        self.yolo_model = YOLO(yolo_model_file_path)

        # Clear cache after model initialization
        GPUController.clear_gpu_memory()

        self.image_processor = ImageProcessor(config)
        self.config = config
        self.glm_ocr = glm_ocr


    def detect_layout(self, image: Image.Image) -> list[dict]:
        """ All YOLO layout regions on a page.

            Returns: A list of dictionaries with 'bbox', 'label' and 'confidence' keys.
        """

        regions = []
        for result in self.yolo_model.predict(image, stream=True):
            for mapped_prediction in ResultProcessor.process_result(result):
                regions.append({
                    "bbox": [int(c) for c in mapped_prediction.bbox],
                    "label": mapped_prediction.label,
                    "confidence": float(mapped_prediction.confidence),
                })
        return regions

    def detect_record_headers(self, image: Image.Image, regions: list[dict]) -> list | None:
        """ Finds the record headers among the layout regions of a given image.

        Args: image (Image.Image): Image to be checked.
        Args: regions (list[dict]): Layout regions of the image, from detect_layout.

        Returns: list of dictionaries for each header with their bounding box and text.
        """

        if not regions:
            logger.info(f"No layout predictions..")
            return None

        image_width, image_height = image.size
        verified_predictions = []

        for region in regions:
            mapped_prediction = MappedPrediction(region["bbox"], region["confidence"], region["label"])

            if self.is_sus_table(mapped_prediction, image_width, image_height):
                new_predictions = self.redetect_region(image, mapped_prediction.bbox, mapped_prediction)
                verified_predictions.extend(new_predictions)
            else:
                verified_predictions.append(mapped_prediction)

        record_header_predictions = [prediction for prediction in verified_predictions if HeaderValidator.is_record_header_candidate(prediction, self.config.header_candidate_labels)]
        if not record_header_predictions:
            logger.info(f"No record headers found in layout predictions...")
            return None

        headers_on_page = []
        for prediction in record_header_predictions:
            bbox = [int(c) for c in prediction.bbox]
            padded_bbox = (
                max(0, bbox[0] - self.config.padding),
                max(0, bbox[1] - self.config.padding),
                min(image.width, bbox[2] + self.config.padding),
                min(image.height, bbox[3] + self.config.padding),
            )

            cropped = image.crop(padded_bbox)

            text = self.glm_ocr.ocr(cropped, prompt=self.HEADER_OCR_PROMPT)

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

    def redetect_region(self, image: Image.Image, bbox: list, original_prediction: MappedPrediction) -> list:
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

        spine_pos = self.image_processor.find_spine_position(cropped_array)

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
            results = self.yolo_model.predict(image, stream=True)

            for result in results:
                mapped_prediction = ResultProcessor.process_result(result, x1 + region_x_offset, y1)

                mapped_predictions.extend(mapped_prediction)

        return mapped_predictions

    def is_sus_table(self, prediction: MappedPrediction, image_width: int, image_height: int) -> bool:
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

        if prediction.confidence >= self.config.sus_table_confidence_threshold:
            return False

        bbox = prediction.bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        area_fraction = bbox_area / image_area

        if area_fraction <= self.config.sus_table_area_threshold:
            logger.info(f"Table detection check passed!")
            return False
        else:
            logger.info(f"Sus table detected: conf={prediction.confidence:.2f}, area_fraction={area_fraction:.2f}")
            return True