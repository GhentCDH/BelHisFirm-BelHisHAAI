from PIL import Image
from dataclasses import dataclass

@dataclass
class Record:
    record_id: int
    record_title: str
    internal_record_number: str
    images: list[Image.Image]
    start_header_bbox: list[float]
    start_header_bbox_meta: dict
    start_header_bbox_page: int
    end_header_bbox: list[float]
    end_header_bbox_meta: dict
    end_header_bbox_page: int
