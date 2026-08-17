from tqdm import tqdm
tqdm.disable = True
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.trainer_utils import set_seed
from qwen_vl_utils import process_vision_info
import torch
import cv2 as cv
import base64
import gc


class OCR_Qwen:

    # This initializes the model and processor once and uses them throughout.
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", seed=69) -> None:
        # Instance variables for storing the message, model name, and seed value.
        self.message = None
        self.prompt = "Your output should be all the text and numbers present in the image. Output None if no text/numbers are present."
        self.model_name = model_name
        self.seed = seed

        # Set random seed for reproducibility.
        set_seed(self.seed)
        # Clear GPU memory cache to free up space.
        torch.cuda.empty_cache()

        # Load the pre-trained model and processor.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

        # Define minimum and maximum pixel values for image processing.
        min_pixels = 128 * 28 * 28
        max_pixels = 720 * 28 * 28

        # Load the processor for text, image, and video inputs.
        self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    # Private method to construct the message for the model.
    def __message_constructor(self, image):
        # Constructing a message with an image in base64 format and a prompt.
        self.message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{image}",
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    # Private method to call the model for generating the output text.
    def __qwen_call(self, store_output=False):
        # Apply the chat template to the message and process the vision-related data.
        text = self.processor.apply_chat_template(
            self.message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.message)
        # Prepare inputs for the model.
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # Move inputs to the GPU.
        inputs = inputs.to("cuda")
        # Generate the output text based on the input.
        generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        # Trim the generated output to match the expected format.
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # Decode the generated text.
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Clean the output text by stripping unnecessary characters.
        output_text = str(output_text).strip("[]'")
        
        # If store_output is True, save the result to a file.
        if store_output:
            with open('test_result.txt', 'w', encoding="utf8") as file:
                file.write("")
                file.writelines(output_text)
        
        return output_text

    # Private method to process and prepare the image for inference.
    def __image_processing(self, image_path):
        # Read the image from the given path.
        image = cv.imread(image_path)
        # Add padding around the image to improve OCR results.
        top, bottom, left, right = 10, 10, 10, 10
        padded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])
        # Convert the image to PNG format and encode it in base64.
        _, buffer = cv.imencode('.png', padded_image)
        image = base64.b64encode(buffer).decode('utf-8')
        # Construct the message with the processed image.
        self.__message_constructor(image)
    
    # Clean up resources (model, processor, and GPU memory) to avoid memory leaks.
    def clean_up_ocr(self):
        del self.model
        del self.processor

        # Clear GPU cache and invoke garbage collection if necessary.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()

    # Method to run the OCR inference and process the result.
    def run_inference(self, image_path):
        # Process the image before passing it to the model.
        self.__image_processing(image_path)
        # Get the model's output text from the inference call.
        output_text = self.__qwen_call(store_output=False)
        # Clean and format the output text by removing unwanted characters.
        output_text = str(output_text).strip().replace('"', "").replace("--", "-").replace("—", "-").replace("  ", " ").replace('|', '').replace('\n', '')
        return output_text
