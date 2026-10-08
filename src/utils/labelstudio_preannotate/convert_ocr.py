"""Convert a plain-text OCR guess into a Label Studio prediction, importable
as a pre-annotation for a TextArea-based labeling config (a caption/transcription
field attached to the whole image, not a region).
"""

from __future__ import annotations


def text_to_prediction(
    text: str,
    model_version: str,
    from_name: str = "caption",
    to_name: str = "image",
) -> dict:
    """Build a single Label Studio prediction dict for one TextArea guess."""
    return {
        "model_version": model_version,
        "result": [
            {
                "type": "textarea",
                "from_name": from_name,
                "to_name": to_name,
                "value": {"text": [text]},
            }
        ],
    }
