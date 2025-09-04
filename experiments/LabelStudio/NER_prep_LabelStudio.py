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


"""
def padd_image(image, color=(255, 255, 255)):
    h, w = image.shape[:2]
    top = (h - 28) // 2
    bottom = h - 28 - top
    left = (w - 28) // 2
    right = w - 28 - left

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        val = int(color if np.isscalar(color) else int(np.mean(color)))
    else:
        r, g, b = map(int, color)
        val = (b, g, r)
    
    return cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=val)


def split_function(image):
    if len(image.shape) != 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    w = round(w / 2)
    gray = gray[300:h - 300, w - 30:w + 30]
    thresholded = cv.threshold(gray, 180, 255, cv.THRESH_BINARY_INV)[1]
    vertical_sum = np.sum(thresholded, axis=0)
    spine_position = np.argmin(vertical_sum) + w
    image_1 = image[0:h, 0:spine_position]
    image_2 = image[0:h, spine_position:w * 2]

    cv.imwrite('links.png', image_1)
    cv.imwrite('rechts.png', image_2)

    return image_1, image_2

def warp_function(image):
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, median_angle, 1.0).astype(np.float32)

    rotated = cv.warpAffine(
        image, M, (w, h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE
    )
    return rotated


class BoxtoLine:

    def __init__(self):
        self.bbox_list = []
        self.line_bbox_list = []

    @staticmethod
    def clamp_box(x0, y0, x1, y1):
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            return x0, y0, x1, y1
    
    @staticmethod
    def vert_iou_calc(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        if x1 <= x3 and x2 >= x4 and y1 <= y3 and y2 >= y4:
            return 1.0
        else:
            inter_w = max(0.0, 0.1)
            inter_h = max(0.0, min(y2, y4) - max(y1, y3))

            inter = inter_w * inter_h

            area_a = max(0.0, (0.1) * max(0.0, (y2 - y1)))
            area_b = max(0.0, (0.1) * max(0.0, (y4 - y3)))

            union = area_a + area_b - inter

            return inter / union 
    
    @staticmethod
    def resolve_bbox(bbox1, bbox2):

        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        bbox3 = (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))

        return bbox3

    def transform_boxes_to_line(self, words_array): 
        # restructure
        self.bbox_list = [(box[0][0], box[0][1], box[2][0], box[2][1]) for box in words_array]
        # clamp boxes
        self.bbox_list = [self.clamp_box(x0, y0, x1, y1) for (x0, y0, x1, y1) in self.bbox_list]
        # purge small boxes
        self.bbox_list = [box for box in self.bbox_list if (box[2] - box[0]) > 0.01 or (box[3] - box[1]) > 0.01]
        # Sort by vertical position and then horizontal position (halfway point)
        restructured_bbox_list = self.bbox_list.copy()
        restructured_bbox_list.sort(key=lambda b: (0.5*(b[1]+b[3]), b[0]))
        # Sort by horizontal position for later use
        self.bbox_list.sort(key=lambda b: (b[1], b[0]))

        word_counter = 0
        

        for idx, bbox in enumerate(restructured_bbox_list):
            if idx == 0:
                line = bbox
                word_counter = 1
                continue
            iou = self.vert_iou_calc(line, bbox)
            if iou > 0.5:
                word_counter += 1
                line = self.resolve_bbox(line, bbox)
            else:
                self.line_bbox_list.append((line, word_counter))
                line = bbox
                word_counter = 1

        self.line_bbox_list.append((line, word_counter)) 
    

def resize_crop(crop, max_h, max_w, color):
    h, w = crop.shape[:2]
    top = (max_h - h) // 2
    bottom = max_h - h - top
    left = 0
    right = max_w - w

    if crop.ndim == 2 or (crop.ndim == 3 and crop.shape[2] == 1):
        val = int(color if np.isscalar(color) else int(np.mean(color)))
    else:
        r, g, b = map(int, color)
        val = (b, g, r)
    return cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_CONSTANT, value=val)

def get_most_color(crop):
    pixels = crop.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_common_rgb = tuple(colors[counts.argmax()][::-1])
    return most_common_rgb

def prepare_words_for_line_matching(line_list, bbox_list, image):

    h, w = image.shape[:2]
    total_words_counted = 0
    cropped_images_per_line = []

    for line in line_list:

        words_in_line = line[1]
        in_loop_word_counter = 0
        store_bbox = []

        while in_loop_word_counter < words_in_line:
            box = bbox_list[total_words_counted]
            # Extract these pairs
            h, w = image.shape[:2]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            store_bbox.append((x1, y1, x2, y2))

            in_loop_word_counter += 1
            total_words_counted += 1

        
        # Sort the store_bbox by x1 (leftmost coordinate)
        store_bbox.sort(key=lambda b: b[0])
        cropped_images_of_line = []

        for box in store_bbox:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            # Normalization
            crop = image[y1:y2, x1:x2]
            # Append the cropped image to the list
            cropped_images_of_line.append(crop)

        cropped_images_per_line.append(cropped_images_of_line)

    return cropped_images_per_line



def main(image_path):

    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image_name = image_path.split('/')[-1].split('.')[0]

    image = warp_function(image)
    image_1, image_2 = split_function(image)

    model = detection_predictor('db_resnet50', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)

    word_array_list = []

    images = [image_1, image_2]

    for idx, image in enumerate(images):
        padded_image = image
        #padded_image = padd_image(image, color=get_most_color(image))
        out = model([padded_image])
        words_array = out[0]['words']
        word_array_list.append(words_array)
        image_drawing = image.copy()
        images[idx] = padded_image
        
        for i, box in enumerate(words_array):
            
            # Extract these pairs, I think the others can work to
            x1 = box[0][0]
            y1 = box[0][1]
            x2 = box[2][0]
            y2 = box[2][1]

            # Normalization
            h, w = padded_image.shape[:2]
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv.rectangle(image_drawing, pt1, pt2, (0, 255, 0), 2)



    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=128 * 28 * 28, max_pixels=2056 * 28 * 28)

    for id, words_array in enumerate(word_array_list):
        box_to_line = BoxtoLine()
        box_to_line.transform_boxes_to_line(words_array)
        bbox_list = box_to_line.bbox_list
        line_list = box_to_line.line_bbox_list
    
        lines_collage = []

        cropped_images = prepare_words_for_line_matching(line_list, bbox_list, images[id])
        
        for idx, crop_line_image in enumerate(cropped_images):
            h, w = None, None
            for crop in crop_line_image:
                if h is None:
                    h, w = crop.shape[:2]
                    color = get_most_color(crop)
                else:
                    h_new = crop.shape[0]
                    h = max(h, h_new)

            padded_line = []

            for crop in crop_line_image:
                padded_image = resize_crop(crop, h, crop.shape[1] + 10, color=color)
                padded_line.append(padded_image)
            
            combined_line_image = np.hstack(padded_line)

            lines_collage.append(combined_line_image)

        for crop in lines_collage:
                crop = base64.b64encode(cv.imencode('.png', crop)[1]).decode('utf-8')
                prompt = (f"Transcribe all text and numbers in the image. Ignore other text artefacts. If no text is present, respond with 'No text found'. ")
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{crop}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
         
                text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(message)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=120)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                with open(os.path.join("experiments/LabelStudio", f"qwen_text_transcription_{image_name}_{id}.txt"), "a") as f:
                    f.write(f"{text[0].strip()}\n")
        print(f"Text transcription for image {id} completed.")

"""



def padd_image(image, color=(255, 255, 255)):
    h, w = image.shape[:2]
    top = (h - 5) // 2
    bottom = h - 5 - top
    left = (w - 5) // 2
    right = w - 5 - left

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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=128 * 28 * 28, max_pixels=2056 * 28 * 28)
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


        list_of_tags = ["Text", "Handtekening"]
        
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

            print(image_name)

            padded_image = padd_image(cropped_image)

            crop = base64.b64encode(cv.imencode('.png', padded_image)[1]).decode('utf-8')
            prompt = (f"Transcribe all text and numbers in the image. Ignore other text artefacts. If no text is present, respond with 'No text found'. ")
            message = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": f"data:image/png;base64,{crop}"},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
            text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
            cleaned_text = text[0].replace('\n', ' ').replace('\\n', ' ').strip()
            if not 'No text found' in text:
                page_list.append(cleaned_text)
                print(f"[OCR]: {cleaned_text}")
            


        

    


if __name__ == "__main__":
    main("/home/bas/Documents/Visual Code Data/project-8-at-2025-09-02-07-09-853136ae/result.json", "/home/bas/Documents/Visual Code Data/project-8-at-2025-09-02-07-09-853136ae/images")

