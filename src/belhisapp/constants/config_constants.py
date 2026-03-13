from src.belhisapp.config import ConfigField

class ConfigConstants:
    """ Constants used for loading, saving and displaying the configuration window."""

    # List of fields that the config parser will look for, the key is what is used for matching, the name for label, and the type for saving back to JSON
    CONFIG_FIELDS: list[ConfigField] = [
        ConfigField("padding", "Padding", int),
        ConfigField("suspicious_table_confidence_threshold", "Suspicious Table Confidence Threshold", float),
        ConfigField("suspicious_table_area_threshold", "Suspicious Table Area Threshold", float),
        ConfigField("spine_vertical_margin", "Spine Vertical Margin", int),
        ConfigField("spine_margin", "Spine Margin", int),
        ConfigField("skip_ocr", "Skip OCR", bool),
        ConfigField("ocr_excluded_labels", "OCR Excluded Labels", list[str])
    ]

    CONFIG_FILE_PATH: str = "src/config/config.json"
