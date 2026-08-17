import os
import json
import pandas as pd
import datetime
from pathlib import Path

_CRF_DIR = Path(__file__).parent.parent
_OUTPUT_DIR = _CRF_DIR.parent / 'output'
_LABELS_FILE = _CRF_DIR / 'labels' / 'labels.json'

_DELIMITERS = {".", ",", ";", ":", "!", "?", "(", ")", "°", "/", "&", '"', "—", "-"}

# Class for outputting CRF results, either in CSV or Excel format,
# and handling various label transformations and color-coded printing.
class Output_CRF:
    def __init__(self):
        self.columns = []  # To store the header columns for the output file
        self.keys = []  # To store the keys (labels) for predictions
        self.collect_labels()  # Load the labels from the JSON file
        self.check_if_output_exists()  # Check if the output directory exists, if not, create it
        self.list_with_converted_lines = []  # Store the transformed lines to be written to file

    # Check if the output directory exists, if not, create it
    def check_if_output_exists(self):
        if not _OUTPUT_DIR.exists():
            _OUTPUT_DIR.mkdir(parents=True)

    # Collect labels from a JSON file and store them in the object's attributes
    def collect_labels(self):
        with open(_LABELS_FILE) as json_file:
            labels_json = json.load(json_file)

        # Extract columns (values from JSON) and keys (keys from JSON)
        self.columns = [value for key, value in list(labels_json.items())[1:]]  # Exclude the first item
        self.keys = [key for key in list(labels_json.keys())[1:]]  # Exclude the first item (usually "START" or "END")
        self._ad_index = self.keys.index("AD") if "AD" in self.keys else -1
        self._n_index = self.keys.index("N") if "N" in self.keys else -1

    # Create a tuple for each token and its corresponding prediction
    def create_tuple_from_prediction(self, token, prediction):
        tuple_list = []  # Initialize an empty list to store the tuples
        for idx, item in enumerate(token):  # Loop through each token
            color_number = 0  # Initialize the color index for prediction visualization
            if item != 'START' and item != 'END':  # Ignore the START and END tokens
                if prediction[0][idx] in self.keys:  # Check if the prediction is a valid key
                    while prediction[0][idx] != self.keys[color_number]:  # Find the index of the prediction
                        color_number += 1  # Increment the color index until a match is found
                    tuple_list.append((token[idx], prediction[0][idx], color_number))  # Append token, prediction, and color index
                else:
                    tuple_list.append((token[idx], prediction[0][idx], 15))  # Default color index if not in keys
            else:
                pass  # Skip the START and END tokens

        return tuple_list  # Return the list of tuples

    def _neighbor_label(self, tuple_list, i, direction):
        """Return the label of the nearest non-delimiter token in the given direction (-1 or +1)."""
        rng = range(i - 1, -1, -1) if direction < 0 else range(i + 1, len(tuple_list))
        for j in rng:
            if tuple_list[j][0] not in _DELIMITERS:
                return tuple_list[j][1]
        return None

    # Transform a list of tuples into a format suitable for CSV/Excel output
    def transform_line_to_csv_format(self, tuple_list, clean_delimiters=True):
        converted_line = [""] * len(self.keys)

        for i, (token, label, index) in enumerate(tuple_list):
            if not (0 <= index < len(converted_line)):
                continue

            if clean_delimiters and token in _DELIMITERS and label != "AD" and label != "N":
                prev_label = self._neighbor_label(tuple_list, i, -1)
                next_label = self._neighbor_label(tuple_list, i, +1)
                for col_label, col_index in (("AD", self._ad_index), ("N", self._n_index)):
                    if col_index >= 0 and prev_label == col_label and next_label == col_label:
                        converted_line[col_index] += token  # no space: delimiter attaches to preceding word
                continue

            if converted_line[index]:
                sep = "" if converted_line[index].endswith("-") else " "
                converted_line[index] += f"{sep}{token}"
            else:
                converted_line[index] = token

        return converted_line

    # Create a CSV file from the transformed lines
    def create_csv(self, lines, file, index):
        now = datetime.datetime.now()  # Get the current datetime
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")  # Format the datetime for the filename
        df = pd.DataFrame(columns=self.columns)  # Create a new DataFrame with the appropriate columns
        for line in lines:
            df = pd.concat([df, pd.Series(line, index=df.columns)], ignore_index=True)  # Append each line to the DataFrame
        df.to_csv(_OUTPUT_DIR / f'{file}_{index}_{filename}.csv', index=False)
        self.clean()  # Clean up the attributes after saving the file

    # Create an Excel file from the transformed lines
    def create_excel(self, lines, file, index):
        now = datetime.datetime.now()  # Get the current datetime
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")  # Format the datetime for the filename
        df = pd.DataFrame(columns=self.columns)  # Create a new DataFrame with the appropriate columns
        for line in lines:
            df = pd.concat([df, pd.DataFrame([line], columns=df.columns)], ignore_index=True)  # Append each line to the DataFrame
        df.to_excel(_OUTPUT_DIR / f'{file}_{index}_{filename}.xlsx', index=False)
        self.clean()  # Clean up the attributes after saving the file

    # Get the ANSI color code corresponding to a particular color index
    def get_ansi_color_code(self, number):
        ansi_colors = {
            3: "\033[91m",  # Bright Red
            4: "\033[92m",  # Bright Green
            5: "\033[97m",  # Bright White
            0: "\033[94m",  # Bright Blue
            1: "\033[95m",  # Bright Magenta
            2: "\033[96m",  # Bright Cyan
            6: "\033[93m",  # Bright Yellow
            7: "\033[90m",  # Bright Black (Gray)
            8: "\033[31m",  # Red
            9: "\033[32m",  # Green
            10: "\033[33m", # Yellow
            11: "\033[34m", # Blue
            12: "\033[35m", # Magenta
            13: "\033[36m", # Cyan
            14: "\033[37m", # White
            15: "\033[30m", # Black
        }
        return ansi_colors.get(number)  # Return the corresponding ANSI color code for the given index

    # Print the output with the appropriate color coding for each token
    def colored_prediction(self, tuple_list):
        line_out = ""  # Initialize an empty string
        for item in tuple_list:  # Loop through each tuple in the list
            color = self.get_ansi_color_code(item[2])  # Get the color code
            line_out += f" {color}{item[0]}\033[0m"  # Append token with color
        return f"[CRF-Result]: {line_out}"  # Return the formatted string

    def label_prediction(self, tuple_list):
        line_out = ""  # Initialize an empty string
        for item in tuple_list:  # Loop through each tuple in the list
            line_out += f" {item[1]}"  # Append the predicted label
        return f"[CRF-Labels]: {line_out}"  # Return the formatted string

    # Clean up the internal state of the object
    def clean(self):
        self.columns = []  # Reset columns
        self.keys = []  # Reset keys
        self.collect_labels()  # Recollect labels
        self.check_if_output_exists()  # Check if the output directory exists again
        self.list_with_converted_lines = []  # Reset the list of converted lines
