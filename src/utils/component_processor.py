import json
import os

from numpy import ndarray
from ultralytics.engine.results import Results

from .page_component import PageComponent
from src.recordprocessing.processor import MappedPrediction

class ComponentProcessor:
    @staticmethod
    def process_result(result: Results, x_offset: int = 0, y_offset: int = 0) -> list[MappedPrediction]:
        """ Processes a single prediction result into a list of MappedPredictions for that prediction.

        Args: result (list[Results]): The prediction result.

        Returns: A list new of MappedPrediction objects.
        """

        mapped_predictions: list[MappedPrediction] = []

        boxes = result.boxes
        class_names = result.names
        classes = boxes.cls.cpu().numpy().astype(int)
        probabilities = boxes.conf.cpu().numpy().astype(float)

        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()

            x_min, y_min, x_max, y_max = xyxy.astype(int)

            bbox = [x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset]
            confidence = probabilities[idx]
            class_id = classes[idx]
            label = class_names[class_id]

            mapped_prediction: MappedPrediction = MappedPrediction(bbox, confidence, label)
            mapped_predictions.append(mapped_prediction)

        return mapped_predictions


    @staticmethod
    def crop_image_to_page_component(image: ndarray, page_component: PageComponent, padding_x : int = 0, padding_y: int = 0) -> ndarray:
        """ Crops an image to a bounding box of a PageComponent.

        Args: image (np.ndarray): The image to be cropped.
        Args: page_component (PageComponent): The PageComponent for the image to be cropped into.

        Returns: cropped image (np.ndarray): The cropped image.
        """

        padding_y_min = page_component.min_y - padding_y
        if padding_y_min < 0:
            padding_y_min = 0

        padding_y_max = page_component.max_y + padding_y
        if padding_y_max > image.shape[0]:
            padding_y_max = image.shape[0] - 1

        padding_x_min = page_component.min_x - padding_x
        if padding_x_min < 0:
            padding_x_min = 0

        padding_x_max = page_component.max_x + padding_x
        if padding_x_max > image.shape[1]:
            padding_x_max = image.shape[1] - 1

        cropped = image[padding_y_min:padding_y_max, padding_x_min:padding_x_max]

        return cropped

    @staticmethod
    def save_page_components(folder_path: str, page_components: list[PageComponent]) -> None:
        """ Saves a list of PageComponent to JSON.

        Args: folder_path (str): The directory in which the JSON will be saved.
        Args results (list[ImageProcessResult]): The list of PageComponent to be saved.

        Returns: None.
        """

        os.makedirs(folder_path, exist_ok=True)

        data_results: list[dict] = []

        # For every page component create a dictionary and store it with the rest
        for result in page_components:
            data = {
                "name": result.name,
                "class_id": int(result.class_id),
                "probability": float(result.confidence),
                "min_x": int(result.min_x),
                "min_y": int(result.min_y),
                "max_x": int(result.max_x),
                "max_y": int(result.max_y),
            }

            data_results.append(data)


        with open(os.path.join(folder_path, "results.json"), "w") as f:
            json.dump(data_results, f, indent=4)