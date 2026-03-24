from dataclasses import dataclass

@dataclass
class MappedPrediction:
    bbox: list
    confidence: float
    label: str