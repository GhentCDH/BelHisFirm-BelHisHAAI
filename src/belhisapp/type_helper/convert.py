from .convert_result import ConvertResult
from typing import get_origin

class Convert:

    @staticmethod
    def convert_value(value: str, target_type: type) -> ConvertResult:
        """ Helper to convert string input to a specified type.

        Args: value (str): Value to be converted.
        Args: target_type (type): Type to convert to.

        Returns: Converted value.
        """

        try:
            origin = get_origin(target_type) or target_type

            if origin is bool:
                result = value.lower() in ("true", "yes", "1", "on")
                return ConvertResult(True, result)

            if origin is list:
                # (If anyone ever needs to restrict list items, do it here)

                items = [item.strip() for item in value.split(",")] if value.strip() else []
                return ConvertResult(True, items)

            if value.strip() == "":
                return ConvertResult(True, target_type())

            converted_val = target_type(value)
            return ConvertResult(True, converted_val)

        except (ValueError, TypeError) as e:
            return ConvertResult(False, message=f"Cannot convert '{value}' to {target_type.__name__}: {e}")

        except Exception as e:
            return ConvertResult(False, message=f"Unexpected error: {e}")