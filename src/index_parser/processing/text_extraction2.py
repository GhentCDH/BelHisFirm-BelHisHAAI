import cv2
import numpy as np
from PIL import Image
from surya.detection import DetectionPredictor
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path

class TextExtractor2:
    def __init__(self, debug=False, device="cuda", binarize=False):
        self.lym = DetectionPredictor(device=device)
        self.debug = debug
        self.binarize = binarize

    def _straighten(self, pil_image):
        gray = np.array(pil_image.convert("L"))
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return pil_image

        all_angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in (l[0] for l in lines)]
        # Keep only near-horizontal lines — verticals (borders, margins) corrupt the median
        angles = [a for a in all_angles if abs(a) < 45]
        if not angles:
            return pil_image

        median_angle = np.median(angles)

        if abs(median_angle) > 20:
            return pil_image

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_image), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)

    def _binarize(self, pil_image):
        gray = np.array(pil_image.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    def extract_text_lines(self, image_path, debug_dir=None):
        with Image.open(image_path) as pil_image:
            source_image = self._straighten(pil_image.copy())
            detection_image = self._binarize(source_image).convert("RGB") if self.binarize else source_image.convert("RGB")
        w, h = source_image.size

        # surya lay out predictions
        predictions = self.lym([detection_image])

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
            
        def merge_overlapping_boxes(boxes, iou_threshold=0.05):
            if not boxes:
                return []

            class _Box:
                def __init__(self, bbox):
                    self.bbox = bbox

            sorted_by_pos = sorted(boxes, key=lambda b: (b.bbox[1], b.bbox[0]))
            current = list(sorted_by_pos[0].bbox)

            merged = []
            for box in sorted_by_pos[1:]:
                x1, y1, x2, y2 = box.bbox
                cx1, cy1, cx2, cy2 = current
                inter = max(0, min(x2, cx2) - max(x1, cx1)) * max(0, min(y2, cy2) - max(y1, cy1))
                union = (x2 - x1) * (y2 - y1) + (cx2 - cx1) * (cy2 - cy1) - inter
                if union > 0 and inter / union > iou_threshold:
                    current = [min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]
                else:
                    merged.append(_Box(tuple(current)))
                    current = [x1, y1, x2, y2]
            merged.append(_Box(tuple(current)))
            return merged

        page_prediction = predictions[0] if predictions else None
        all_boxes = merge_overlapping_boxes(page_prediction.bboxes) if page_prediction else []

        left_boxes = []
        right_boxes = []
        heading_boxes = []
        for box in all_boxes:
            side = assign_side(box)
            if side == "left":
                left_boxes.append(box)
            elif side == "right":
                right_boxes.append(box)
            else:
                heading_boxes.append(box)

        def detect_x_outliers_dbscan(boxes, side="?", eps_ratio=0.007, min_samples=2):
            if len(boxes) < min_samples:
                return [], []

            x_values = np.array([float(box.bbox[0]) for box in boxes])
            eps = max(5.0, w * eps_ratio)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_values.reshape(-1, 1))

            unique_labels = set(labels) - {-1}

            if not unique_labels:
                if self.debug:
                    print(f"[DBSCAN:{side}] eps={eps:.1f}  eps_ratio={eps_ratio}  min_samples={min_samples}")
                    print(f"         all {len(boxes)} points are noise → all marked as outliers")
                return list(range(len(boxes))), labels.tolist()

            cluster_medians = {lbl: float(np.median(x_values[labels == lbl])) for lbl in unique_labels}

            # Rightmost cluster is the indentation/outlier cluster; all others are normal.
            # When only one cluster exists it is treated as normal (only noise is outlier).
            if len(unique_labels) > 1:
                outlier_label = max(cluster_medians, key=cluster_medians.get)
            else:
                outlier_label = None  # single cluster → no cluster is an outlier

            outlier_indices = [
                idx for idx, lbl in enumerate(labels)
                if lbl == -1 or lbl == outlier_label
            ]
            normal_count = len(boxes) - len(outlier_indices)

            if self.debug:
                cluster_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
                print(f"[DBSCAN:{side}] eps={eps:.1f}  eps_ratio={eps_ratio}  min_samples={min_samples}")
                for lbl in sorted(unique_labels):
                    marker = " ← outlier (rightmost)" if lbl == outlier_label else " ← normal"
                    print(f"         cluster {lbl}: median_x1={cluster_medians[lbl]:.1f}  n={cluster_counts[lbl]}{marker}")
                print(f"         noise points  : {int(np.sum(labels == -1))}")
                print(f"         normal        : {normal_count}")
                print(f"         outliers      : {len(outlier_indices)} / {len(boxes)}")

            return outlier_indices, labels.tolist()

        def bboxes_from_indices(boxes, indices):
            return [boxes[idx] for idx in indices]

        left_outlier_indices, left_cluster_labels = detect_x_outliers_dbscan(left_boxes, side="left")
        right_outlier_indices, right_cluster_labels = detect_x_outliers_dbscan(right_boxes, side="right")

        left_outliers = bboxes_from_indices(left_boxes, left_outlier_indices)
        right_outliers = bboxes_from_indices(right_boxes, right_outlier_indices)
        left_outlier_ids = {id(box) for box in left_outliers}
        right_outlier_ids = {id(box) for box in right_outliers}

        def sort_boxes_by_side(boxes):
            side_priority = {"heading": 0, "left": 1, "right": 2}

            def sort_key(box):
                side = assign_side(box)
                x1, y1, _, _ = box.bbox
                return (side_priority[side], y1, x1)

            return sorted(boxes, key=sort_key)

        line_crops_with_outlier = []
        if all_boxes:
            sorted_boxes = sort_boxes_by_side(all_boxes)
            for box in sorted_boxes:
                x1, y1, x2, y2 = map(int, box.bbox)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                side = assign_side(box)
                if side == "heading":
                    continue

                cropped_line = source_image.crop((x1, y1, x2, y2))
                is_outlier = (side == "left" and id(box) in left_outlier_ids) or \
                             (side == "right" and id(box) in right_outlier_ids)

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

            stem = Path(image_path).stem
            out_dir = Path(debug_dir) if debug_dir else Path(".")
            output_file = out_dir / f"{stem}_debug_{side_name}_x1_y1.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            return output_file

        # This is for debugging
        if self.debug:
            stem = Path(image_path).stem
            out_dir = Path(debug_dir) if debug_dir else Path(".")

            cv_image = cv2.cvtColor(np.array(detection_image), cv2.COLOR_RGB2BGR)

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

            cv2.imwrite(str(out_dir / f"{stem}_debug_bbox.png"), cv_image)

            side_plots = {
                "left": save_side_plot(left_boxes, "left", "green", left_outlier_indices),
                "right": save_side_plot(right_boxes, "right", "red", right_outlier_indices),
                "heading": save_side_plot(heading_boxes, "heading", "blue"),
            }
        else:
            side_plots = {}

        return line_crops_with_outlier


if __name__ == "__main__":
    image_path = "EHC_B665_O_2025_1892_III-IV_0926.tif"
    extractor = TextExtractor2(debug=False)
    result = extractor.extract_text_lines(image_path)






