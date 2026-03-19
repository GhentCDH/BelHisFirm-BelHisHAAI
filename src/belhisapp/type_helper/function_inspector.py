import inspect
from typing import Callable

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

