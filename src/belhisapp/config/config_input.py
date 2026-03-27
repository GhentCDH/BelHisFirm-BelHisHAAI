from textual import events
from textual.widgets import Input

from src.belhisapp.config import ConfigField

class ConfigInput (Input):
    def __init__(self, config_field: ConfigField, classes: str = "") -> None:
        self.config_field = config_field

        super().__init__("", classes=classes)

    def _on_compose(self, event: events.Compose) -> None:
        self.value = f"{self.config_field.value}"

    def set_value(self, value: str) -> None:
        """ Set value of input text and config field text stored within.

            Args: value (str): The value to be set.

            Returns: None.
        """

        self.value = value
        self.config_field.value = value