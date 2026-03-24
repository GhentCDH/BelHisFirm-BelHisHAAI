from dataclasses import dataclass

@dataclass
class ConfigParameter:
    padding: int
    sus_table_confidence_threshold: float
    sus_table_area_threshold: float
    spine_vertical_margin: int
    spine_margin: int
    skip_ocr: bool