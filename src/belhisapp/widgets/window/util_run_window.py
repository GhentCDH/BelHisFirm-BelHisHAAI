import asyncio

from typing import Callable

from textual.containers import Center
from textual.widgets import Static, Button

from src.belhisapp.config import ConfigOperationResult, ConfigParser, FormBuilder
from src.belhisapp.type_helper import FunctionInspector
from src.belhisapp.widgets.window import Window

class UtilRunWindow(Window):
    """ Window build from a method, this window will build dynamically build a parameter form to run that method. """

    def __init__(self, label: str, method: Callable) -> None:

        self._method = method

        header = Static(label, classes="FormHeader")

        params = FunctionInspector.get_function_params(method)
        config_fields = FunctionInspector.parse_config_fields_from_function_inspection(params)
        form_build_result = FormBuilder.build_form(config_fields)

        self._config_inputs = form_build_result.config_inputs
        self.execute_button = Button("Execute", classes="FormButton")

        # Status/error log, same convention as ConfigWindow's _error_log
        self._status_log = Static("", classes="FormError")

        super().__init__([header, Static(""), form_build_result.form, Static(""), Center(self.execute_button), Static(""), self._status_log])

    async def on_button_pressed(self, event: Button.Pressed) -> None:

        if event.button != self.execute_button:
            return

        # Convert the textbox values into typed arguments matching the method's parameters
        result = ConfigParser.parse_config(self._config_inputs)

        if not result.success:
            self._log_result(result)
            return

        data = result.value

        self.execute_button.disabled = True
        self._log_result(ConfigOperationResult(True, message="Running..."))

        try:
            # Run off the event loop thread - these methods (e.g. the record pipeline) can
            # take minutes and would otherwise freeze the whole TUI until they finished.
            await asyncio.to_thread(self._method, *data.values())
            self._log_result(ConfigOperationResult(True, message="Finished successfully."))

        except Exception as e:
            self._log_result(ConfigOperationResult(False, message=f"An error occurred while running this utility.\n{e}"))

        finally:
            self.execute_button.disabled = False

    def _log_result(self, result: ConfigOperationResult) -> None:

        if not result.success:
            self._status_log.styles.color = "red"
        else:
            self._status_log.styles.color = "green"

        self._status_log.update(result.message)