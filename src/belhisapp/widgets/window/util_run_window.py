from typing import Callable

from textual.containers import Center
from textual.widgets import Static, Button

from src.belhisapp.config import FormBuilder
from src.belhisapp.type_helper import FunctionInspector
from src.belhisapp.widgets.window import Window

class UtilRunWindow(Window):

    def __init__(self, label: str, method: Callable):

        header = Static(label, classes="FormHeader")

        params = FunctionInspector.get_function_params(method)
        config_fields = FunctionInspector.parse_config_fields_from_function_inspection(params)
        form_build_result = FormBuilder.build_form(config_fields)

        self._config_inputs = form_build_result.config_inputs
        self.execute_button = Button("Execute", classes="FormButton")

        """ Whoever might read this in the future:
        
            # How to read values from the config inputs
            result = ConfigParser.parse_config(self._config_inputs)
    
            if result.success:
                data = result.value
    
                # Run this windows method with arguments
                method(*data.values())
    
            result.message -> If a parsing error occurred (ex: string in an int field) this message will diagnose that
        """

        super().__init__([header, Static(""), form_build_result.form, Static(""), Center(self.execute_button)])