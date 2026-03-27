from typing import Callable

from textual.containers import Vertical, Center
from textual.widgets import Static, Button

from src.belhisapp.config import ConfigInput
from src.belhisapp.config import FormBuilder
from src.belhisapp.type_helper import FunctionInspector
from src.belhisapp.widgets.window import Window

class UtilRunWindow(Window):

    _config_inputs: list[ConfigInput]

    def __init__(self, label: str, method: Callable):

        header = Static(label, classes="FormHeader")

        # Build form from given method
        params = FunctionInspector.get_function_params(method)
        config_fields = FunctionInspector.parse_config_fields_from_function_inspection(params)
        form_build_result = FormBuilder.build_form(config_fields)

        self._config_inputs = form_build_result.config_inputs
        self.execute_button = Button("Execute", classes="FormButton")

        super().__init__([header, Static(""), form_build_result.form, Static(""), Center(self.execute_button)])