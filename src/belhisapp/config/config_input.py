from textual import events
from textual.widgets import Input

from src.belhisapp.config import ConfigField


class ConfigInput (Input):
    """ Widget used within the config window to get input and map to a JSON file """

    _config_field: ConfigField

    def __init__(self, config_field: ConfigField, classes: str = ""):
        self._config_field = config_field

        super().__init__("", classes=classes)

    def _on_compose(self, event: events.Compose) -> None:
        self.value = f"{self._config_field.value}"

    def set_value(self, value: str) -> None:
        self.value = value
        self._config_field.value = value