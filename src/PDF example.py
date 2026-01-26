import os
import cv2 as cv
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
from wand.image import Image as WandImage
from doctr.models import detection_predictor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64



def split_image(image):
    if len(image.shape) != 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    w = round(w / 2)
    gray = gray[0:h, w - 50:w + 50]
    thresholded = cv.threshold(gray, 180, 255, cv.THRESH_BINARY_INV)[1]
    vertical_sum = np.sum(thresholded, axis=0)
    spine_position = np.argmin(vertical_sum) + w
    image_1 = image[0:h, 0:spine_position]
    image_2 = image[0:h, spine_position:w * 2]


    return image_1, image_2

def preprocess_and_deskew(image, crop_top=20, crop_bottom=20, crop_left=0, crop_right=0, deskew_threshold=0.40, filename="unknown"):
    h, w = image.shape[:2]
    cropped = image[crop_top:h-crop_bottom, crop_left:w-crop_right]

    # This is very optional, because these images are rotated in the book
    if filename.split("/")[-1] == "0011.jpg" and crop_right > 0:
        cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)
    if filename.split("/")[-1] == "0013.jpg" and crop_left > 0:
        cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)

    cropped_rgb = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(cropped_rgb)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    with WandImage(file=buf) as wand_img:
        wand_img.deskew(deskew_threshold)
        wand_img.trim()
        wand_buf = np.asarray(bytearray(wand_img.make_blob(format="PNG")), dtype=np.uint8)
        processed = cv.imdecode(wand_buf, cv.IMREAD_COLOR)
    return processed

def group_words_into_lines_iou(words, iou_threshold=0.5):
    if not words:
        return []

    # Sort words top-to-bottom by vertical center, then left-to-right
    words_sorted = sorted(
        words, 
        key=lambda w: (((w["bbox"][1] + w["bbox"][3]) / 2), w["bbox"][0])
    )

    lines = []
    current_line = [words_sorted[0]]
    line_bbox = words_sorted[0]["bbox"]

    def vert_iou(b1, b2):
        _, y1, _, y2 = b1
        _, y3, _, y4 = b2
        inter_h = max(0, min(y2, y4) - max(y1, y3))
        union_h = (y2 - y1) + (y4 - y3) - inter_h
        return inter_h / union_h if union_h > 0 else 0.0

    def merge_bbox(b1, b2):
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2
        return (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))

    for w in words_sorted[1:]:
        bbox = w["bbox"]
        if vert_iou(line_bbox, bbox) >= iou_threshold:
            # Same line → add
            current_line.append(w)
            line_bbox = merge_bbox(line_bbox, bbox)
        else:
            # New line → flush old
            lines.append(current_line)
            current_line = [w]
            line_bbox = bbox
    if current_line:
        lines.append(current_line)

    # Sort words left-to-right within each line
    for line in lines:
        line.sort(key=lambda w: w["bbox"][0])

    return lines


def build_hocr(words, page_width, page_height, page_num=0, iou_threshold=0.5):
    lines = group_words_into_lines_iou(words, iou_threshold=iou_threshold)

    hocr = [
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
        '<head>',
        '<title>OCR Output</title>',
        '<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>',
        '<meta name="ocr-system" content="docTR+Qwen"/>',
        '<meta name="ocr-system" content="Hallo Frans en Johan!"/>',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word"/>',
        '</head>',
        '<body>',
        f'<div class="ocr_page" id="page_{page_num+1}" title="image; bbox 0 0 {page_width} {page_height}; ppageno {page_num}">',
        f'<div class="ocr_carea" id="block_1" title="bbox 0 0 {page_width} {page_height}">',
        '<p class="ocr_par" id="par_1">'
    ]

    for li, line_words in enumerate(lines, start=1):
        # Compute bbox for the line
        lx0 = min(w["bbox"][0] for w in line_words)
        ly0 = min(w["bbox"][1] for w in line_words)
        lx1 = max(w["bbox"][2] for w in line_words)
        ly1 = max(w["bbox"][3] for w in line_words)

        hocr.append(f'<span class="ocr_line" id="line_{li}" title="bbox {lx0} {ly0} {lx1} {ly1}">')

        for wi, word in enumerate(line_words, start=1):
            x0, y0, x1, y1 = map(int, word["bbox"])
            text = (
                str(word["text"])
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            hocr.append(
                f'<span class="ocrx_word" id="word_{li}_{wi}" title="bbox {x0} {y0} {x1} {y1}">{text}</span>'
            )

        hocr.append('</span>')  # close line

    hocr += ['</p>', '</div>', '</div>', '</body>', '</html>']
    return "\n".join(hocr)


def main(folder_path):

    # Load models
    detection_model = detection_predictor('db_resnet50', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)

    def process_image_with_doctr(image):
        out = detection_model([image])
        return out[0]['words']

    def process_image_with_qwen(image, words_array):
        words = []

        def clamp_box(x0, y0, x1, y1, image_width, image_height):
            x0, x1 = sorted([max(0, min(x0, image_width - 1)), max(0, min(x1, image_width - 1))])
            y0, y1 = sorted([max(0, min(y0, image_height - 1)), max(0, min(y1, image_height - 1))])
            return x0, y0, x1, y1

        for i, box in enumerate(words_array):
            h, w = image.shape[:2]
            x1 = int(box[0][0] * w)
            y1 = int(box[0][1] * h)
            x2 = int(box[2][0] * w)
            y2 = int(box[2][1] * h)


            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, image.shape[1], image.shape[0])
            crop = image[y1:y2, x1:x2]
            try:
                _, buffer = cv.imencode('.png', crop)
                encoded_image = base64.b64encode(buffer).decode('utf-8')

                prompt = (f"Transcribe the word. Return only the word, no other text.")
                message = [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": f"data:image/png;base64,{encoded_image}"},
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

                generated_ids = qwen_model.generate(**inputs, max_new_tokens=25)
                generated_ids_trimmed = [
                                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                    ]
                text = processor.batch_decode(
                                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                text = f'{text[0].strip()} '
                string = {"text": text, "bbox": (x1, y1, x2, y2)}
                words.append(string)
                print(string)
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                continue
        return words

    if not os.path.exists("~temp"):
        os.makedirs("~temp")

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            left_hocr_path = f"~temp/{file_name}_left.hocr"
            right_hocr_path = f"~temp/{file_name}_right.hocr"

            if os.path.exists(os.path.expanduser(left_hocr_path)) and os.path.exists(os.path.expanduser(right_hocr_path)):
                print(f"⏩ Output for {file_name} already exists, skipping.")
                continue

            image_path = os.path.join(folder_path, file_name)
            image = cv.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            else:
                print("Loaded image:", image_path)
                #image_left, image_right = split_image(image)

                manual = "/home/bas/Documents/Visual Code Data/Manual Edit"

                left_name = f"{file_name.split('.')[0]}-0-0.jpg"
                right_name = f"{file_name.split('.')[0]}-1-0.jpg"

                image_left = os.path.join(manual, left_name)
                image_right = os.path.join(manual, right_name)

                print("Left image path:", image_left)

                image_left = cv.imread(image_left)
                image_right = cv.imread(image_right)

                processed_left = preprocess_and_deskew(image_left, crop_top=150, crop_bottom=150, crop_left=200, filename=left_name)
                processed_right = preprocess_and_deskew(image_right, crop_top=150, crop_bottom=150, crop_right=200, filename=right_name)

                # Save the processed images temporarily for PDF creation
                cv.imwrite(f"~temp/{file_name}_left.jpg", processed_left)
                cv.imwrite(f"~temp/{file_name}_right.jpg", processed_right)
                print(f"Processed and saved split images for {file_name}")
            

                # OCR and text extraction
                words_array_left = process_image_with_doctr(processed_left)
                words_array_right = process_image_with_doctr(processed_right)
                print(f"Detected {len(words_array_left)} words on left, {len(words_array_right)} words on right using Doctr.")

    
                words_left = process_image_with_qwen(processed_left, words_array_left)
                words_right = process_image_with_qwen(processed_right, words_array_right)
                print(f"Recognized {len(words_left)} words on left, {len(words_right)} words on right using Qwen.")

                h_left, w_left = processed_left.shape[:2]
                h_right, w_right = processed_right.shape[:2]

                hocr_left = build_hocr(words_left, w_left, h_left, page_num=0, iou_threshold=0.5)
                hocr_right = build_hocr(words_right, w_right, h_right, page_num=1, iou_threshold=0.5)

                with open(f"~temp/{file_name}_left.hocr", "w", encoding="utf-8") as f:
                    f.write(hocr_left)
                with open(f"~temp/{file_name}_right.hocr", "w", encoding="utf-8") as f:
                    f.write(hocr_right)
                print(f"Saved HOCR files for {file_name}")



main("/home/bas/Documents/Visual Code Data/RF_1944-1945_III_folio")