from dataclasses import dataclass
from typing import Any

@dataclass
class ConvertResult:
    success: bool
    result: Any = None
    message: str = ""