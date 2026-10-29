"""Tests for the per-label YOLO confidence thresholds of VisionAnalyzer.

Usage:
    uv run pytest tests
"""

from src.processors.record_processor.data import ConfigParameter, MappedPrediction
from src.processors.record_processor.pipeline.vision_analyzer import VisionAnalyzer


def make_analyzer(**config_overrides) -> VisionAnalyzer:
    """ A VisionAnalyzer with only a config, without loading the YOLO or GLM-OCR models. """

    analyzer = VisionAnalyzer.__new__(VisionAnalyzer)
    analyzer.config = ConfigParameter(50, 0.85, 0.4, 200, 300, True, set(), set(), **config_overrides)
    return analyzer


def prediction(label: str, confidence: float) -> MappedPrediction:
    return MappedPrediction([0, 0, 10, 10], confidence, label)


def test_default_thresholds():
    config = make_analyzer().config
    assert config.yolo_confidence == 0.25
    assert config.yolo_label_confidence == {"Text": 0.55}


def test_text_below_its_threshold_is_dropped():
    analyzer = make_analyzer()
    kept = analyzer.confident_enough([prediction("Text", 0.54), prediction("Text", 0.55), prediction("Text", 0.9)])
    assert [p.confidence for p in kept] == [0.55, 0.9]


def test_other_labels_keep_low_confidence():
    """ Labels without their own threshold are only filtered by yolo_confidence in predict, so a low-confidence
        Table (like the full shareholder table on 1880 page 0210, at 0.41) must survive. """

    analyzer = make_analyzer()
    predictions = [prediction("Table", 0.41), prediction("Section-header", 0.40), prediction("Page-footer", 0.47)]
    assert analyzer.confident_enough(predictions) == predictions


def test_custom_label_thresholds():
    analyzer = make_analyzer(yolo_label_confidence={"Caption": 0.7})
    kept = analyzer.confident_enough([prediction("Caption", 0.6), prediction("Text", 0.3), prediction("Caption", 0.8)])
    assert [(p.label, p.confidence) for p in kept] == [("Text", 0.3), ("Caption", 0.8)]


def test_default_label_thresholds_are_not_shared():
    """ yolo_label_confidence uses a default_factory, so changing one config's dict doesn't change another's. """

    first, second = make_analyzer().config, make_analyzer().config
    first.yolo_label_confidence["Caption"] = 0.9
    assert second.yolo_label_confidence == {"Text": 0.55}
