from dataclasses import dataclass

@dataclass
class LoadConfigResult:
    success: bool
    error: str