import cv2 as cv
import numpy as np
from doctr.models import detection_predictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
import io
from PIL import Image
import os
import json


def padd_image(image, color=(255, 255, 255)):
    top, bottom, left, right = 20, 20, 20, 20

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        val = int(color if np.isscalar(color) else int(np.mean(color)))
    else:
        r, g, b = map(int, color)
        val = (b, g, r)
    
    return cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=val)


def extract_content(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)
        id_to_filename = {img["id"]: os.path.basename(img["file_name"]) for img in data["images"]}
        id_to_label = {cat["id"]: cat["name"] for cat in data["categories"]}
        results = [
            (ann["bbox"], id_to_filename[ann["image_id"]], id_to_label[ann["category_id"]])
            for ann in data["annotations"]]

    return results

def main(file, image_folder):
    
    bbox = extract_content(file)

    image_name = None
    page_list = []
    
    for idx, bbox_smol in enumerate(bbox):

        if image_name != bbox_smol[1]:
        
            if image_name != None:
                with open(os.path.join("/mnt/UGent_Share/ghentcdh_belhisfirm/Workspace/LabeledData", f"transcription_{image_name}_{idx}.txt"), "a") as f:
                    for ann in page_list:
                        f.write(str(ann))
                page_list = []
            
        image_name = bbox_smol[1]

        image = cv.imread(os.path.join(image_folder, image_name))


        list_of_tags = ["Tabulair_aandeelhouders", ]
        
        if bbox_smol[2] in list_of_tags:
        
            x1 = int(bbox_smol[0][0])
            y1 = int(bbox_smol[0][1])
            w = int(bbox_smol[0][2])
            h = int(bbox_smol[0][3])

            x2, y2 = x1 + w, y1 + h

            # Check if box is valid and within image bounds
            img_h, img_w = image.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or w <= 0 or h <= 0:
                continue

            cropped_image = image[y1:y2, x1:x2]

            padded_image = padd_image(cropped_image)
            cv.imwrite(f"experiments/LabelStudio/output/cropped_{image_name}_{idx}.png", padded_image)

    


if __name__ == "__main__":
    main("/home/bas/Documents/Visual Code Data/project-8-at-2025-09-02-07-09-853136ae/result.json", "/home/bas/Documents/Visual Code Data/project-8-at-2025-09-02-07-09-853136ae/images")

