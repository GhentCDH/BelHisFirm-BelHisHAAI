import html
import re

import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_MAX_NEW_TOKENS = 2048


class GLMOCREngine:
    """Loads GLM-OCR once and shares it across header detection, line OCR, and table transcription.
    The model is only loaded on the first ocr() call, so a run that never transcribes does not load it."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.processor = None
        self.model = None

    def _load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = (
            GlmOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            .eval()
            .cuda()
        )

    def ocr(self, image: Image.Image, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Transcribe an image crop. GLM-OCR emits <table><tr><td> markup even for
        plain prose, so strip_html is always applied here rather than left to callers."""
        if self.model is None:
            self._load()

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return self._strip_html(raw)

    @staticmethod
    def _strip_html(text: str) -> str:
        text = re.sub(r"</tr>", "\n", text)
        text = re.sub(r"</td>", "  ", text)
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text).strip()
