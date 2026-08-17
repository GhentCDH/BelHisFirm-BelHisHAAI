import os
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from pathlib import Path
import pandas as pd
import re

class Text_Line_Extractor:
    # Constructor
    def __init__(self):
        self.image = None
        self.worker_images = []  # List to hold images to be processed
        self.temp_folder = ""  # Temporary folder to save intermediate results
    
    # Private method for pre-processing image (erosion to enhance certain features)
    def __preprocessor(self, image):
        eroded_kernel_size = (2, 12)  # Kernel size for erosion
        eroded_iterations = 1  # Number of times to apply erosion
        kernel = np.ones(eroded_kernel_size, np.uint8)
        preprocessed_image = cv.erode(image, kernel, iterations=eroded_iterations)
        return preprocessed_image
    
    # Private method for splitting the image into two halves (left and right)
    def __splitter(self, image):
        edges = cv.Canny(image, 50, 150, apertureSize=3)  # Edge detection
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)  # Line detection

        angles_new = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # Angle calculation for lines
                angles_new.append(angle)

        median_angle_new = np.median(angles_new)  # Median angle for rotation correction

        # Limit rotation to ±20 degrees
        if abs(median_angle_new) <= 20:
            # Rotate the image based on the calculated median angle
            (h_new, w_new) = image.shape[:2]
            center_new = (w_new // 2, h_new // 2)
            M_new = cv.getRotationMatrix2D(center_new, median_angle_new, 1.0)
            image = cv.warpAffine(image, M_new, (w_new, h_new), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        # If angle exceeds ±20 degrees, no rotation is performed

        # Apply dilation to enhance the image for splitting
        kernel = np.ones((10, 2), np.uint8)
        dia_image = cv.dilate(image, kernel, iterations=1)
        h, w = image.shape
        w = round(w / 2)
        gray = dia_image[0:h, w - 75:w + 75]
        vertical_sum = np.sum(gray, axis=0)
        spine_position = np.argmin(vertical_sum) + w
        image_left = image[0:h, 0:spine_position-75]
        image_right = image[0:h, spine_position-75:(w * 2)]
        return image_left, image_right

    # Private method for running YOLO detector on the image and saving detected boxes
    def __yolo_detector(self, image, id):
        if not os.path.isdir(os.path.join(self.temp_folder, id)):
            os.mkdir(os.path.join(self.temp_folder, id))  # Create folder for temporary results

        preprocessed_image = self.__preprocessor(image)  # Preprocess the image before detection

        h, w = image.shape  # Get image dimensions

        # Inference with YOLO model
        model = YOLO('index_parser/model/best.pt') 
        results = model(cv.cvtColor(preprocessed_image, cv.COLOR_GRAY2BGR))

        # Calculate bounding boxes from YOLO results
        i = 0
        bounding_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bounding_boxes.append((0, y1, w, y2))

            bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))  # Sort bounding boxes
            
            combined_boxes = self.combine_overlapping_boxes_with_iou(bounding_boxes)  # Combine overlapping boxes

            # Save individual box images
            for i, (x1, y1, x2, y2) in enumerate(combined_boxes):
                box_image = image[max(y1-5, 0):min(y2+5, image.shape[0]), 0:w]
                # Only save if the image is not empty
                if box_image.size > 0 and box_image.shape[0] > 0 and box_image.shape[1] > 0:
                    cv.imwrite(os.path.join(self.temp_folder, id, f'{id}_{i}.png'), box_image)
                i += 1
    
    # Calculate the Intersection over Union (IoU) between two boxes
    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        nx1, ny1, nx2, ny2 = box2
        
        ix1 = max(x1, nx1)
        iy1 = max(y1, ny1)
        ix2 = min(x2, nx2)
        iy2 = min(y2, ny2)

        intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (nx2 - nx1) * (ny2 - ny1)

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area != 0 else 0  # Return IoU value
        return iou

    # Combine overlapping bounding boxes using IoU threshold
    def combine_overlapping_boxes_with_iou(self, bounding_boxes, iou_threshold=0.05):
        combined_boxes = []
        current_box = bounding_boxes[0]

        for i in range(0, len(bounding_boxes)):
            next_box = bounding_boxes[i]

            iou = self.calculate_iou(current_box, next_box)
            
            if iou > iou_threshold:
                x1, y1, x2, y2 = current_box
                nx1, ny1, nx2, ny2 = next_box
                current_box = (
                    min(x1, nx1),  # min x1
                    min(y1, ny1),  # min y1
                    max(x2, nx2),  # max x2
                    max(y2, ny2)   # max y2
                )
            else:
                combined_boxes.append(current_box)
                current_box = next_box
        combined_boxes.append(current_box)

        return combined_boxes
    
    # Extract all numbers from text using regular expressions
    def extract_numbers(self, text):
        return [int(num) for num in re.findall(r'\d+', text)]
    
    # Private method for detecting sentences and returning their position in X-axis
    def __sentence_detector(self, image_path):
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        # Apply dilation and morphological operations
        image = cv.dilate(image, (7,7), iterations=2)
        kernel = np.ones((3,3), np.uint8)
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        
        # Apply denoising
        cv.fastNlMeansDenoising(image, image, 50, 7, 21)
        
        # Threshold to get binary image
        _, binary_image = cv.threshold(image, 150, 255, cv.THRESH_BINARY_INV)
        
        # Find contours and sort them from left to right
        h, w = image.shape
        contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])
        bounding_boxes = [cv.boundingRect(c) for c in contours]
        
        # Helper function to check if two bounding boxes are close
        def are_boxes_close(box1, box2, threshold=10):
            return box2[0] <= box1[0] + box1[2] + threshold

        min_width = 15
        min_height = 20
        first_word_box = None

        # Iterate through bounding boxes to detect sentence lines
        for i, initial_box in enumerate(bounding_boxes):
            first_word_box = initial_box
            for box in bounding_boxes[i+1:]:
                if are_boxes_close(first_word_box, box):
                    first_word_box = (
                        min(first_word_box[0], box[0]),                
                        min(first_word_box[1], box[1]),                
                        max(first_word_box[0] + first_word_box[2], box[0] + box[2]) - min(first_word_box[0], box[0]),  
                        max(first_word_box[1] + first_word_box[3], box[1] + box[3]) - min(first_word_box[1], box[1])   
                    )
                else:
                    break
            if first_word_box[2] >= min_width and first_word_box[3] >= min_height:
                x, y, width, height = first_word_box
                cv.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
                break
            else:
                first_word_box = None
        return x / w if first_word_box else 0
    
    # Main method to extract text lines from the image
    def text_line_extractor(self, image_path, splits, temp_folder="index_parser/~temp"):
        self.temp_folder = temp_folder
        image_path = Path(image_path)
        i = 0
        self.image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Read image
        padding_size = 50
        self.image = cv.copyMakeBorder(self.image, padding_size, padding_size, padding_size, padding_size, 
                          cv.BORDER_CONSTANT, value=255)
        self.worker_images.append(self.image)
        while i < splits:
            temp_split_images = []
            for image in self.worker_images:
                image_left, image_right = self.__splitter(image)  # Split image into left and right
                temp_split_images.append(image_left)
                temp_split_images.append(image_right)
            self.worker_images = temp_split_images
            i += 1
        for id, image in enumerate(self.worker_images):
            self.__yolo_detector(image, str(id))  # Apply YOLO detection on each split image
        self.temp_folder = Path(self.temp_folder).resolve()  # Resolve to full path
        
        data = []

        # Get sorted file paths of detected text regions
        file_paths = list(self.temp_folder.glob('*/*.png'))
        sorted_file_paths = sorted(file_paths, key=lambda p: (int(p.parent.name), self.extract_numbers(p.stem)))

        previous_intendation = None
        valid_intendation = False

        last_id = 0

        # Collect data for CSV output
        for path in filter(lambda p: p.is_file() and p.suffix == ".png", sorted_file_paths):
            id = int(path.parts[-2])  # Extract the ID from file path
            image_name = path.name
            x_coord = self.__sentence_detector(path)  # Detect sentence positions

            # Commented out intention validation logic

            data.append({'id': id, 'image': image_name, 'Xnorm': x_coord})  # Store data
            last_id = id

        # Save extracted data to CSV file
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_folder, f"{image_path.stem}.csv")
        df.to_csv(csv_path, index=False)
    
    def reset(self):
        self.image = None
        self.worker_images = []
        self.temp_folder = ""