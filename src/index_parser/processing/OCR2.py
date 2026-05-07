from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

_INSTRUCTION = (
    "You are an expert at transcribing historical text lines exactly as written. "
    "Output only the transcribed text — no explanations, labels, or commentary. "
    "Preserve original spelling, capitalisation, and punctuation. "
    "Separate multiple lines with \\n. "
    "Output an empty line for blank input. "
    "Output [ILLEGIBLE] for text that cannot be read.\n\n"
    "Pay special attention to the following:\n"
    "A 5 in this document looks like a bold vintage printed digit with a slightly bent/angled stroke at the top (not fully flat), an open upper section (not looped or closed), and a large rounded bump curving to the right and closing at the bottom."
    "A 3 in this document looks like a bold vintage printed digit with a flat horizontal stroke at the top, two rounded bumps stacked vertically on the right side, and an open left side — the top bump is slightly smaller than the bottom one, with a small horizontal indentation or notch pointing left in the middle where the two curves meet."
)


class OCR:

    def __init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    def run(self, image):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _INSTRUCTION},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()
