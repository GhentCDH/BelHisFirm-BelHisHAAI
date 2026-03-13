from dataclasses import dataclass
from textual.containers import Container, Center

from src.belhisapp.config.config_input import ConfigInput

@dataclass
class BuildFormResult:
    form: Center
    config_inputs: list[ConfigInput]
