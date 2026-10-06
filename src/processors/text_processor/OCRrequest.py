import argparse
import base64
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PROMPT = "Transcribe the text in this image. Ignore multiple . in a row."


def load_image(image_path: Path) -> Image.Image:
    """Open an image file, handling TIFF's palette/multi-page quirks."""
    image = Image.open(image_path)

    n_frames = getattr(image, "n_frames", 1)
    if n_frames > 1:
        logger.warning(
            f"Multi-page TIFF detected ({n_frames} pages): {image_path}. "
            "Using first page only."
        )

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    return image


def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URI."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def transcribe_image(
    image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
) -> str:
    """Send an image and a prompt to a vLLM OpenAI-compatible server and return the transcribed text."""
    if not re.match(r'^\w+://', base_url):
        base_url = f"http://{base_url}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    response = requests.post(base_url, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


class OnlineOCR:
    """Drop-in replacement for OCR2.OCR that transcribes line-crop images via one or
    more vLLM OpenAI-compatible endpoints instead of a locally-loaded model."""

    def __init__(self, base_urls=None, model=None, prompt=None, max_tokens=512, max_workers=2):
        self.base_urls = base_urls or [DEFAULT_BASE_URL]
        self.model = model or DEFAULT_MODEL
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    def run(self, image):
        return transcribe_image(image, prompt=self.prompt, base_url=self.base_urls[0],
                                 model=self.model, max_tokens=self.max_tokens)

    def run_batch(self, images):
        """OCR a list of images concurrently, preserving order. Requests are spread
        round-robin across self.base_urls (one entry per servable model instance) and
        capped at self.max_workers in flight at once."""
        if not images:
            return []
        results = [None] * len(images)

        def _fetch(i, image):
            base_url = self.base_urls[i % len(self.base_urls)]
            return i, transcribe_image(image, prompt=self.prompt, base_url=base_url,
                                        model=self.model, max_tokens=self.max_tokens)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_fetch, i, img) for i, img in enumerate(images)]
            for future in tqdm(futures, desc="Running OCR (online)"):
                i, text = future.result()
                results[i] = text
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an image via a vLLM server")
    parser.add_argument("--image_path", type=Path, help="Path to the image to transcribe")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send along with the image")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    args = parser.parse_args()

    image = load_image(args.image_path)
    text = transcribe_image(image, prompt=args.prompt, base_url=args.base_url, model=args.model)
    print(text)
