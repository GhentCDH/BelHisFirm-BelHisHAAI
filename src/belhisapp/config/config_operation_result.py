from dataclasses import dataclass
from typing import Any

@dataclass
class ConfigOperationResult:
    success: bool
    value: Any = None
    message: str = None