from dataclasses import dataclass

@dataclass
class ConfigOperationResult:
    success: bool
    error: str