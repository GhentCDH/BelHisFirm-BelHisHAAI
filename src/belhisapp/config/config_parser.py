import json
from typing import Any

from .load_config_result import LoadConfigResult
from src.belhisapp.config import ConfigField, ConfigInput

class ConfigParser:

    @staticmethod
    def load_config(filepath: str, config_inputs: list[ConfigInput],
                    config_fields: list[ConfigField]) -> LoadConfigResult:
        """ Load the configuration file into a list of ConfigInputs.

        Args: filepath (str): Filepath of the JSON configuration file.
        Args: config_inputs (list[ConfigInput]): List of ConfigInputs which will have their values loaded.
        Args: config_fields (list[ConfigField]): Representing the fields that need to be parsed.

        Returns: LoadConfigResult object, with operation success status.
        """

        try:
            with open(filepath, "r") as file:
                data: dict[str, Any] = json.loads(file.read())

                # For all config fields that need to be loaded
                for config_field in config_fields:

                    # Find all config inputs that use this config field
                    found_config_inputs: list[ConfigInput] = [config_input for config_input in config_inputs if config_input.config_field == config_field]

                    # Match the config input given key to a JSON value
                    for config_input in found_config_inputs:
                        config_input.value = str(data.get(config_field.key))

                return LoadConfigResult(True, "")

        except Exception as e:
            return LoadConfigResult(False, f"An error occurred while reading the configuration file.\n{e}")