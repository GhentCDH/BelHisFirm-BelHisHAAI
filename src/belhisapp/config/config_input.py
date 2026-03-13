from textual.widgets import Input

from src.belhisapp.config import ConfigField


class ConfigInput (Input):
    """ Widget used within the config window to get input and map to a JSON file """

    def __init__(self, config_field: ConfigField, classes: str = ""):
        self.config_field = config_field

        # Automatically set the text to the given value of the config field.
        super().__init__(f"{config_field.value}", classes=classes)