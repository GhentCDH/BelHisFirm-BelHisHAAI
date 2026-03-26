from typing import Callable

from pyfiglet import Figlet
from textual.widgets import Static

from src.belhisapp.constants import AppConstants
from src.belhisapp.config import ConfigInput
from src.belhisapp.config import FormBuilder
from src.belhisapp.type_helper import FunctionInspector
from src.belhisapp.widgets.window import Window

class UtilRunWindow(Window):

    _config_inputs: list[ConfigInput]

    def __init__(self, label: str, method: Callable):

        f = Figlet('shadow')
        header = Static(f.renderText(label), classes="FormHeader")

        # Build form from given method
        params = FunctionInspector.get_function_params(method)
        config_fields = FunctionInspector.parse_config_fields_from_function_inspection(params)
        form_build_result = FormBuilder.build_form(config_fields)

        self._config_inputs = form_build_result.config_inputs

        super().__init__([header, Static(""), form_build_result.form])
