from textual.containers import Center
from textual.widgets import Static, Button

from src.belhisapp.config import ConfigOperationResult, ConfigParser, FormBuilder
from src.belhisapp.widgets.window import Window
from src.belhisapp.constants import ConfigConstants, AppConstants

class ConfigWindow(Window):
    """ Window object representing the configuration window."""

    def __init__(self) -> None:

        # Build form from config fields
        form_build_result = FormBuilder.build_form(ConfigConstants.CONFIG_FIELDS)

        self._form = form_build_result.form

        self._config_inputs = form_build_result.config_inputs

        # Define save button
        self._save_button = Button("Save", classes="FormButton")

        # Define reset button
        self._reset_button = Button("Reset", classes="FormButton")

        # Define header
        self._header = Static(AppConstants.CONFIG_LOGO, classes="FormHeader")

        # Define error log
        self._error_log = Static("", classes="FormError")

        # These are the widgets that will load within the window (Static is a gap)
        widgets = [self._header, Static(""), self._form, Center(self._save_button), Static(""), Center(self._reset_button), Static(""), self._error_log]

        super().__init__(widgets)

    async def on_button_pressed(self, event: Button.Pressed):

        # Save JSON
        if event.button == self._save_button:

            result = ConfigParser.save_config_to_json(ConfigConstants.CONFIG_FILE_PATH, self._config_inputs)

            self._log_result(result)

        # Reset form (Re-load JSON)
        elif event.button == self._reset_button:
            self.load_json()

    def load_json(self) -> None:

        # Load JSON
        result = ConfigParser.load_config(ConfigConstants.CONFIG_FILE_PATH, self._config_inputs, ConfigConstants.CONFIG_FIELDS)

        self._log_result(result)

    def _log_result(self, result: ConfigOperationResult) -> None:

        if not result.success:
            self._error_log.styles.color = "red"
        else:
            self._error_log.styles.color = "green"

        self._error_log.update(result.message)