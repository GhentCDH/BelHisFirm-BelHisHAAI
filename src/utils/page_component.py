from dataclasses import dataclass

@dataclass
class PageComponent:
    name: str
    class_id: int
    confidence: float
    min_x: int
    min_y: int
    max_x: int
    max_y: int