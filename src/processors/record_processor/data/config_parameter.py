from dataclasses import dataclass, field

@dataclass
class ConfigParameter:
    padding: int
    sus_table_confidence_threshold: float
    sus_table_area_threshold: float
    spine_vertical_margin: int
    spine_margin: int
    skip_ocr: bool
    ocr_excluded_labels: set[str]
    header_candidate_labels: set[str]
    # YOLO keeps regions from this confidence up, except for the labels in yolo_label_confidence, which need more
    yolo_confidence: float = 0.25
    yolo_label_confidence: dict[str, float] = field(default_factory=lambda: {"Text": 0.55})