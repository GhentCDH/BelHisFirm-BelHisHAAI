from textual.widgets import Input

from src.belhisapp.config import ConfigField


class ConfigInput (Input):
    """ Widget used within the config window to get input and map to a JSON file """

    config_field: ConfigField

    def __init__(self, config_field: ConfigField, classes: str = ""):
        self.config_field = config_field

        super().__init__("", classes=classes)

    def set_value(self, value: str) -> None:
        self.value = value
        self.config_field.value = value