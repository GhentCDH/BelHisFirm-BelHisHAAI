import inspect
from typing import Callable

from src.belhisapp.config import ConfigField


class FunctionInspector:

    @staticmethod
    def get_function_params(function: Callable) -> list[dict]:
        """ Inspects a given function for its parameters.

            Args: function (Callable): Function to inspect.

            Returns: A list of dictionaries containing parameter name, type, and their default value if present.
        """
        sig = inspect.signature(function)
        params = []
        for name, param in sig.parameters.items():

            # Determine if the parameter has a default
            default = param.default if param.default != inspect.Parameter.empty else None

            # Determine the type annotation (defaults to string)
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else str

            params.append({"name": name, "default": default, "type": annotation})
        return params

    @staticmethod
    def parse_config_fields_from_function_inspection(params: list[dict], auto_rename: bool = True) -> list[ConfigField]:
        """ Converts a list of dictionaries containing parameter name, type, and default value to a list of ConfigFields.

            Args: params (list[dict]): List of dictionaries containing parameter name, type, and default value.

            Returns: A new list of ConfigFields based off given parameters.
        """

        config_fields = []

        for param in params:

            display_name = param["name"]
            if auto_rename:
                display_name = display_name.replace("_", " ").title()

            config_field = ConfigField(param["name"], display_name, param["type"], param["default"])
            config_fields.append(config_field)

        return config_fields