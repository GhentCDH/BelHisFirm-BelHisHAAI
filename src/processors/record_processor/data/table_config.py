from dataclasses import dataclass

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
