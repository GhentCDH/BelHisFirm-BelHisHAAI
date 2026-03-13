from dataclasses import dataclass

@dataclass
class ConfigOperationResult:
    success: bool
    message: str