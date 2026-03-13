from textual.containers import Center
from textual.widget import Widget
from textual.widgets import Static, Button

from src.belhisapp.config import ConfigInput, ConfigOperationResult, ConfigParser, FormBuilder
from src.belhisapp.widgets.window import Window
from src.belhisapp.constants import ConfigConstants, AppConstants

from src.belhisapp.config.build_form_result import BuildFormResult

class ConfigWindow(Window):

    _form: Center
    _config_inputs: list[ConfigInput]

    _save_button: Button
    _error_log: Static

    def __init__(self):

        # Build form from config fields
        form_build_result: BuildFormResult = FormBuilder.build_form(ConfigConstants.CONFIG_FIELDS)

        self._form: Center = form_build_result.form
        self._config_inputs: list[ConfigInput] = form_build_result.config_inputs

        # Define save button
        self._save_button: Button = Button("Save", classes="FormButton")

        # Define header
        header: Static = Static(AppConstants.CONFIG_LOGO, classes="FormHeader")

        # Gap between form and button
        gap = Static("")
        gap2 = Static("")

        # Define error log
        self._error_log: Static =  Static("", classes="FormError")

        # These are the widgets that will load within the window
        widgets: list[Widget] = [header, self._form, gap, Center(self._save_button), gap2, self._error_log]

        super().__init__(widgets)

    async def on_button_pressed(self, event: Button.Pressed):

        # Save JSON
        if event.button == self._save_button:
            result: ConfigOperationResult = ConfigParser.save_config(ConfigConstants.CONFIG_FILE_PATH, self._config_inputs)

            self._log_result(result)

    def load_json(self):

        # Load JSON
        result: ConfigOperationResult = ConfigParser.load_config(ConfigConstants.CONFIG_FILE_PATH, self._config_inputs, ConfigConstants.CONFIG_FIELDS)

        self._log_result(result)

    def _log_result(self, result: ConfigOperationResult):
        if not result.success:
            self._error_log.styles.color = "red"
        else:
            self._error_log.styles.color = "green"

        self._error_log.update(result.message)