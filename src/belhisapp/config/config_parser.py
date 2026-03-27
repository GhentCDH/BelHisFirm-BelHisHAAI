import json
from typing import Any

from .config_operation_result import ConfigOperationResult
from src.belhisapp.config import ConfigField, ConfigInput
from src.belhisapp.type_helper import Convert, ConvertResult

class ConfigParser:

    @staticmethod
    def load_config(filepath: str, config_inputs: list[ConfigInput], config_fields: list[ConfigField]) -> ConfigOperationResult:
        """ Load the configuration file into a list of ConfigInputs.

        Args: filepath (str): Filepath of the JSON configuration file.
        Args: config_inputs (list[ConfigInput]): List of ConfigInputs which will have their values loaded.
        Args: config_fields (list[ConfigField]): Representing the fields that need to be parsed.

        Returns: LoadConfigResult object, with operation success status.
        """

        try:
            with open(filepath, "r") as file:
                data: dict[str, str] = json.loads(file.read())

                # For all config fields that need to be loaded
                for config_field in config_fields:

                    # Find all config inputs that use this config field
                    found_config_inputs: list[ConfigInput] = [config_input for config_input in config_inputs if config_input.config_field == config_field]

                    # Match the config input given key to a JSON value and change the config input value
                    for config_input in found_config_inputs:

                        value = data.get(config_field.key)

                        # Load lists differently
                        if isinstance(value, list):
                            config_input.set_value(", ".join(map(str, value)))
                        else:
                            config_input.set_value(str(value) if value is not None else "")

                return ConfigOperationResult(True, message="Configuration loaded successfully.")

        except Exception as e:
            return ConfigOperationResult(False, message=f"An error occurred while reading the configuration file.\n{e}")

    @staticmethod
    def save_config_to_json(filepath: str, config_inputs: list[ConfigInput]) -> ConfigOperationResult:
        """ Parse the config inputs into a specified JSON file.

        Args: filepath (str): Filepath of the JSON configuration file.
        Args: config_inputs (list[ConfigInput]): List of ConfigInputs which will have their values parsed.
        Args: config_fields (list[ConfigField]): Representing the fields that are present in the JSON configuration file.

        Returns: LoadConfigResult object, with operation success status.
        """

        try:
            result = ConfigParser.parse_config(config_inputs)

            if not result.success:
                return ConfigOperationResult(False, f"{result.message}")

            with open(filepath, "w") as file:
                file.write(json.dumps(result.value))

            return ConfigOperationResult(True, message="Configuration saved successfully.")

        except Exception as e:
            return ConfigOperationResult(False, message=f"An error occurred while saving the configuration.\n{e}")

    @staticmethod
    def parse_config(config_inputs: list[ConfigInput]) -> ConfigOperationResult:
        """ Parse a list of ConfigInputs into a dictionary mapping label to value

            Args: config_inputs (list[ConfigInput]): List of ConfigInputs which will have their values parsed.

            Returns: LoadConfigResult object, with operation success status and new dictionary.
        """

        try:
            data: dict[str, Any] = {}

            # Build a dictionary mapping config input value to their corresponding keys
            for config_input in config_inputs:

                # Convert the string present in the config input value (because it's a textbox) to the type that the config field specified
                convert_result: ConvertResult = Convert.convert_value(config_input.value, config_input.config_field.type)

                # Check if the conversion was valid
                if not convert_result.success:
                    return ConfigOperationResult(False, f"An error occurred while saving the configuration.\n{convert_result.message}")

                data[config_input.config_field.key] = convert_result.result

            return ConfigOperationResult(True, data, message="Configuration parsed successfully.")

        except Exception as e:
            return ConfigOperationResult(False, f"An error occurred while parsing the configuration.\n{e}")