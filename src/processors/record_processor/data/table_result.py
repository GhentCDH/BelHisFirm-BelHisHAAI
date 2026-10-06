from dataclasses import dataclass
from PIL import Image

@dataclass
class TableResult:
    page_index: int                # 0-based index into record.images
    bbox: list                     # [x1, y1, x2, y2] in the page image's own coordinates, table+caption merged
    crop_image: Image.Image        # in-memory crop, not yet persisted to disk
    plain_text: str                # GLM-OCR's transcription of the crop (post strip_html)
    structured: dict | None        # TABLE_SCHEMA-shaped dict from vLLM, or None on failure
    error: str | None = None       # the TranscriptionError message on failure
