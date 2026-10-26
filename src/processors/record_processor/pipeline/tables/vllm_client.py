"""Thin OpenAI-compatible client for a vLLM server hosting a vision-language model.

Adapted from BelHisFirm---HisTableFinder's transcribe_tables/vllm_client.py.
Only the text-only structuring path is ported (structure_transcription) -
GLM-OCR already did the image reading (see GLMOCREngine), so this client
never sends the model an image itself.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI
from openai.types.shared_params import ResponseFormatJSONSchema

from .prompt import SYSTEM_PROMPT, USER_PROMPT_FROM_TEXT
from .schema import TABLE_SCHEMA

_MAX_TOKENS = 9216


def _timing_suffix(elapsed: float, usage: Any) -> str:
    tokens = getattr(usage, "completion_tokens", None)
    if not tokens:
        return f"{elapsed:.1f}s"
    return f"{elapsed:.1f}s, {tokens} tok, {tokens / elapsed:.1f} tok/s"


class TranscriptionError(Exception):
    """Raised when a model response can't be turned into a table transcription.

    Carries the raw (unparsed) model output, if any was received, so callers can
    save it for inspection even though it didn't parse as valid JSON.
    """

    def __init__(self, message: str, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def structure_transcription(client: OpenAI, model: str, ocr_text: str, source_name: str) -> dict[str, Any]:
    """Structure an already-OCR'd table transcription (plain text, from
    GLMOCREngine) into the parsed JSON transcription via schema-constrained
    decoding. The model never sees the image, only the already-OCR'd text.

    source_name is used only for logging/error messages.

    The result carries an extra "plain_text" key (the raw ocr_text passed in,
    not part of TABLE_SCHEMA/the model's own output) so the original OCR
    transcription for this crop is always kept alongside the structured JSON.
    """
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        response_format=ResponseFormatJSONSchema(
            type="json_schema",
            json_schema={"name": "table", "schema": TABLE_SCHEMA, "strict": True},
        ),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_FROM_TEXT.format(text=ocr_text)},
        ],
    )
    elapsed = time.monotonic() - start

    content = response.choices[0].message.content
    print(f"--- {source_name} ({_timing_suffix(elapsed, response.usage)}) ---")
    if not content:
        raise TranscriptionError(f"Empty response from model for {source_name}")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise TranscriptionError(f"Invalid JSON from model for {source_name}: {e}", raw_content=content) from e
    result["plain_text"] = ocr_text
    return result
