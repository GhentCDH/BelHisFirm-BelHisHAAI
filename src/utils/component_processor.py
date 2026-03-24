import json
import os

from ultralytics.engine.results import Results

from src.recordprocessing.data import MappedPrediction

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
    def save_predictions_to_json(folder_path: str, mapped_prediction: list[MappedPrediction]) -> None:
        """ Saves a list of MappedPrediction to JSON.

        Args: folder_path (str): The directory in which the JSON will be saved.
        Args results (list[MappedPrediction]): The list of MappedPrediction to be saved.

        Returns: None.
        """

        os.makedirs(folder_path, exist_ok=True)

        data_results: list[dict] = []

        # For every page component create a dictionary and store it with the rest
        for result in mapped_prediction:
            data = {
                "bbox": result.bbox,
                "confidence": float(result.confidence),
                "label": result.label,
            }

            data_results.append(data)


        with open(os.path.join(folder_path, "results.json"), "w") as f:
            json.dump(data_results, f, indent=4)