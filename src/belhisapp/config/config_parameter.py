from dataclasses import dataclass

@dataclass
class ConfigParameter:
    padding: int
    suspicious_table_confidence_threshold: float
    suspicious_table_area_threshold: float
    spine_vertical_margin: int
    spine_margin: int
    skip_ocr: bool
    ocr_excluded_labels: list[str]