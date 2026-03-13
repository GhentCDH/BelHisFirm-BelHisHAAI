from src.belhisapp.config import ConfigField, BuildFormResult, ConfigInput

from textual.containers import Horizontal, Vertical, Center
from textual.widgets import Static

class FormBuilder:

    @staticmethod
    def build_form (config_fields: list[ConfigField]) -> BuildFormResult:
        """ Builds a form that can be displayed by a WindowContainer, and be loaded/saved by ConfigParser.

            Args: config_fields: Fields present on the form.

            Returns: BuildFormResult, an object with all input fields on the form, and the widget to be rendered.
        """

        config_inputs = []
        rows: list[Horizontal] = []

        # Build a row for every config field with an input box and a label
        for config_field in config_fields:
            label = Static(f"{config_field.name}:", classes="FormLabel")
            config_input = ConfigInput(config_field, classes="FormTextbox")

            # Store the textboxes so we have a reference for later loading
            config_inputs.append(config_input)

            rows.append(Horizontal(label, config_input, classes="FormRow"))

        form: Center = Center(Vertical(*rows))

        return BuildFormResult(form, config_inputs)
