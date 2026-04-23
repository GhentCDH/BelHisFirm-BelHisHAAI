import cv2
import numpy as np
from PIL import Image
from surya.detection import DetectionPredictor
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

IMAGE_PATH = "EHC_B665_O_2025_1892_III-IV_0926.tif"


class TextExtractor2:
    def __init__(self, debug=False, device="cuda"):
        self.lym = DetectionPredictor(device=device)
        self.debug = debug

    def extract_text_lines(self, image_path):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        w, h = image.size

        # surya lay out predictions
        predictions = self.lym([image])

        linepos1 = int(w / 3)
        linepos2 = int((2 * w) / 3)

        def assign_side(box):
            x1, _, x2, _ = box.bbox

            contains_line1 = x1 <= linepos1 <= x2
            contains_line2 = x1 <= linepos2 <= x2

            if contains_line1 and contains_line2:
                return "heading"
            elif contains_line1:
                return "left"
            elif contains_line2:
                return "right"
            else:
                box_center = (x1 + x2) / 2
                dist_to_line1 = abs(box_center - linepos1)
                dist_to_line2 = abs(box_center - linepos2)
                return "left" if dist_to_line1 <= dist_to_line2 else "right"
            
        left_boxes = []
        right_boxes = []
        heading_boxes = []
        page_prediction = predictions[0] if predictions else None
        if page_prediction is not None:
            for box in page_prediction.bboxes:
                side = assign_side(box)
                if side == "left":
                    left_boxes.append(box)
                elif side == "right":
                    right_boxes.append(box)
                else:
                    heading_boxes.append(box)

        def detect_x_outliers_dbscan(boxes, eps_ratio=0.005, min_samples=5):
            if len(boxes) < min_samples:
                return [], []

            x_values = np.array([[float(box.bbox[0])] for box in boxes])
            eps = max(5.0, w * eps_ratio)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_values)
            outlier_indices = [idx for idx, label in enumerate(labels) if label == -1]
            return outlier_indices, labels.tolist()

        def bboxes_from_indices(boxes, indices):
            return [boxes[idx] for idx in indices]

        left_outlier_indices, left_cluster_labels = detect_x_outliers_dbscan(left_boxes)
        right_outlier_indices, right_cluster_labels = detect_x_outliers_dbscan(right_boxes)

        left_outliers = bboxes_from_indices(left_boxes, left_outlier_indices)
        right_outliers = bboxes_from_indices(right_boxes, right_outlier_indices)
        left_outlier_ids = {id(box) for box in left_outliers}
        right_outlier_ids = {id(box) for box in right_outliers}

        line_crops_with_outlier = []
        if page_prediction is not None:
            for box in page_prediction.bboxes:
                x1, y1, x2, y2 = map(int, box.bbox)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                cropped_line = image.crop((x1, y1, x2, y2))
                side = assign_side(box)
                is_outlier = False
                if side == "left":
                    is_outlier = id(box) in left_outlier_ids
                elif side == "right":
                    is_outlier = id(box) in right_outlier_ids

                line_crops_with_outlier.append((cropped_line, is_outlier))

        def save_side_plot(boxes, side_name, color, outlier_indices=None):
            x1_values = [box.bbox[0] for box in boxes]
            y1_values = [box.bbox[1] for box in boxes]

            outlier_indices = outlier_indices or []
            outlier_index_set = set(outlier_indices)
            normal_x = [x for i, x in enumerate(x1_values) if i not in outlier_index_set]
            normal_y = [y for i, y in enumerate(y1_values) if i not in outlier_index_set]
            outlier_x = [x for i, x in enumerate(x1_values) if i in outlier_index_set]
            outlier_y = [y for i, y in enumerate(y1_values) if i in outlier_index_set]

            plt.figure(figsize=(8, 6))
            plt.scatter(normal_x, normal_y, c=color, s=30, label="normal")
            if outlier_x:
                plt.scatter(outlier_x, outlier_y, c="black", s=45, marker="x", label="outlier")
            plt.title(f"{side_name.capitalize()} side: x1 & y1")
            plt.xlabel("x1 position")
            plt.ylabel("y1 position")
            plt.xlim(0, w)
            plt.ylim(h, 0)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")

            output_file = f"debug_{side_name}_x1_&_y1.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            return output_file

        # This is for debugging
        if self.debug:

            # Show the image with lines and boxes
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            cv2.line(cv_image, (linepos1, 0), (linepos1, h), (0, 255, 255), 3)
            cv2.line(cv_image, (linepos2, 0), (linepos2, h), (0, 255, 255), 3)

            for box in left_boxes:
                x1, y1, x2, y2 = map(int, box.bbox)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for box in right_boxes:
                x1, y1, x2, y2 = map(int, box.bbox)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for box in heading_boxes:
                x1, y1, x2, y2 = map(int, box.bbox)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            output_path = "debug_page_thirds.png"
            cv2.imwrite(output_path, cv_image)

            side_plots = {
                "left": save_side_plot(left_boxes, "left", "green", left_outlier_indices),
                "right": save_side_plot(right_boxes, "right", "red", right_outlier_indices),
                "heading": save_side_plot(heading_boxes, "heading", "blue"),
            }
        else:
            side_plots = {}

        return line_crops_with_outlier


if __name__ == "__main__":
    extractor = TextExtractor2(debug=False)
    result = extractor.extract_text_lines(IMAGE_PATH)






