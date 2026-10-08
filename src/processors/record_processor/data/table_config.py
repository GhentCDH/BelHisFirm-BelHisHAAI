from dataclasses import dataclass, field

# Sub-stages of table processing, in the order they run
TABLE_STEPS = ["crop", "transcribe", "structure", "parse", "excel"]

@dataclass
class TableConfig:
    skip_tables: bool = False
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "qwen3.6-35b-nvfp4"
    table_class_name: str = "Table"
    caption_class_name: str = "Caption"
    caption_max_gap: int = 40
    caption_min_overlap: float = 0.3
    crop_margin: int = 0
    steps: list[str] = field(default_factory=lambda: list(TABLE_STEPS))
    glm_checkpoint: str | None = None   # fine-tuned GLM-OCR checkpoint for headers, body OCR and tables, None = the base model
    structuring_concurrency: int = 8    # structuring calls in flight against the vLLM server at once
    focus_shareholders: bool = False    # only keep an image copy next to the JSON of shareholder registers
