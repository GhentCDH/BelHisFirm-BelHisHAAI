from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Static, Button, Input

from src.belhisapp.widgets.window.window import Window
from src.belhisapp.constants import ConfigWindowConstants

class ConfigWindow(Window):

    _textboxes: list[Input]

    def __init__(self):

        rows: list[Horizontal] = []

        # Textboxes used for JSON parsing
        _textboxes: list[Input] = []

        # Define header
        header: Static = Static(ConfigWindowConstants.CONFIG_LOGO, classes="FormHeader")

        # Build a row for every config field with an input box and a label
        for name in ConfigWindowConstants.CONFIG_FIELDS:
            label = Static(f"{name}:", classes="FormLabel")
            textbox = Input(placeholder=name, classes="FormTextbox")

            # Save textbox separately so we can easily access its value later
            _textboxes.append(textbox)

            rows.append(Horizontal(label, textbox, classes="FormRow"))

        form = Center(Vertical(*rows))

        # Define save button
        button: Static = Static("Save", classes="FormButton")

        # Gap between form and button
        gap = Static("")

        super().__init__([header, form, gap, Center(button)])