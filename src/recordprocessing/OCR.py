from pathlib import Path

import torch
from PIL import Image, ImageDraw
from surya.detection import DetectionPredictor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class OCRProcessor:
    def __init__(self):
        self.detection_predictor = DetectionPredictor()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    def detect_lines(self, image: Image.Image) -> list:
        """Detect text lines in an image."""
        predictions = self.detection_predictor([image])
        return predictions[0].bboxes

    def ocr_cropped_line(self, cropped_image: Image.Image) -> str:
        """Run OCR on a cropped text line using Qwen."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": "Transcribe the text"},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""

    def process_image(self, image_path: Path, padding: int = 5) -> list[dict]:
        """Detect lines and OCR each one."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        bboxes = self.detect_lines(image)
        print(f"Detected {len(bboxes)} text lines")

        results = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.bbox
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)

            cropped = image.crop((x1, y1, x2, y2))
            text = self.ocr_cropped_line(cropped)

            results.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
            })
            print(f"Line {i + 1}: {text}")

        return results

    def visualize_detected_lines(self, image_path: Path, output_path: Path = None):
        """Detect text lines and visualize them with bounding boxes."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        bboxes = self.detect_lines(image)

        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if output_path:
            image.save(output_path)
            print(f"Saved visualization to {output_path}")
        else:
            image.show()

        print(f"Detected {len(bboxes)} text lines")
        return image


if __name__ == "__main__":
    sample_image = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/EHC_B665_O_2025_1909_III_0015.jp2")

    ocr = OCRProcessor()

    # Visualize detected lines
    lines_output = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/detected_lines_visualization.jpg")
    ocr.visualize_detected_lines(sample_image, lines_output)

    # Process and OCR all detected lines
    results = ocr.process_image(sample_image)
    print("\n--- OCR Results ---")
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['text']}")
    with open("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - Testing/ocr_results.txt", "w") as f:
        for result in results:
            f.write(f"{result['text']}\n")
