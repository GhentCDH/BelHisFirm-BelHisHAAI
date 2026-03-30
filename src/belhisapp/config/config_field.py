from dataclasses import dataclass
from typing import Type, Any

@dataclass
class ConfigField:
    key: str
    name: str
    type: Type[Any]

    value: str = ""