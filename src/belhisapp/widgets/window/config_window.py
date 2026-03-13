from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Static, Input

from src.belhisapp.config import ConfigInput, ConfigOperationResult, ConfigParser
from src.belhisapp.widgets.window import Window
from src.belhisapp.constants import ConfigConstants, AppConstants

class ConfigWindow(Window):

    _textboxes: list[ConfigInput]

    def __init__(self):

        self._textboxes = []
        rows: list[Horizontal] = []

        # Build a row for every config field with an input box and a label
        for config_field in ConfigConstants.CONFIG_FIELDS:
            label = Static(f"{config_field.name}:", classes="FormLabel")
            textbox = ConfigInput(config_field, classes="FormTextbox")

            # Store the textboxes so we have a reference for later loading
            self._textboxes.append(textbox)

            rows.append(Horizontal(label, textbox, classes="FormRow"))


        form = Center(Vertical(*rows))

        # Define save button
        button: Static = Static("Save", classes="FormButton")

        # Define header
        header: Static = Static(AppConstants.CONFIG_LOGO, classes="FormHeader")

        # Gap between form and button
        gap = Static("")

        super().__init__([header, form, gap, Center(button)])

    def load_json(self):
        result: ConfigOperationResult = ConfigParser.load_config(ConfigConstants.CONFIG_FILE_PATH, self._textboxes, ConfigConstants.CONFIG_FIELDS)

        # Error checking for JSON parsing (TO DO: make the app show a message)
        if not result.success:
            pass