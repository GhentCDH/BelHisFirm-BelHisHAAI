import os
import json
import pandas as pd
import datetime

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
        if not os.path.exists('index_parser/output/'):  # Check if the directory exists
            os.mkdir('index_parser/output/')  # Create the directory if it doesn't exist

    # Collect labels from a JSON file and store them in the object's attributes
    def collect_labels(self):
        # Open and read the JSON file that contains the label information
        with open('index_parser/CRF/labels/labels.json') as json_file:
            labels_json = json.load(json_file)  # Load the JSON data into a Python dictionary
            json_file.close()
        
        # Extract columns (values from JSON) and keys (keys from JSON)
        self.columns = [value for key, value in list(labels_json.items())[1:]]  # Exclude the first item
        self.keys = [key for key in list(labels_json.keys())[1:]]  # Exclude the first item (usually "START" or "END")

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

    # Transform a list of tuples into a format suitable for CSV/Excel output
    def transform_lines_to_csv_format(self, list_with_tuple_lists):
        for tuple_list in list_with_tuple_lists:
            converted_line = []  # Initialize an empty list to hold the converted line
            while len(self.keys) != len(converted_line):  # Ensure the converted line matches the number of keys
                converted_line.append("")  # Fill with empty strings to match the number of keys
            for tuple_ in tuple_list:
                token = tuple_[0]  # Get the token
                index = tuple_[2]  # Get the corresponding color index
                if 0 <= index < len(converted_line):  # Ensure the index is valid
                    if len(converted_line[index]) > 1:  # If there's already text at the index
                        converted_line[index] = f'{converted_line[index]} {token}'  # Append the token
                    else:
                        converted_line[index] = token  # Otherwise, just set the token
            self.list_with_converted_lines.append(converted_line)  # Add the converted line to the list
        return self.list_with_converted_lines  # Return the list of converted lines

    # Create a CSV file from the transformed lines
    def create_csv(self, lines, file, index):
        now = datetime.datetime.now()  # Get the current datetime
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")  # Format the datetime for the filename
        df = pd.DataFrame(columns=self.columns)  # Create a new DataFrame with the appropriate columns
        for line in lines:
            df = pd.concat([df, pd.Series(line, index=df.columns)], ignore_index=True)  # Append each line to the DataFrame
        df.to_csv(f'index_parser/output/{file}_{index}_{filename}.csv', index=False)  # Save the DataFrame to a CSV file
        self.clean()  # Clean up the attributes after saving the file
        self.clean()  # Clean up the attributes after saving the file

    # Create an Excel file from the transformed lines
    def create_excel(self, lines, file, index):
        now = datetime.datetime.now()  # Get the current datetime
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")  # Format the datetime for the filename
        df = pd.DataFrame(columns=self.columns)  # Create a new DataFrame with the appropriate columns
        for line in lines:
            df = pd.concat([df, pd.DataFrame([line], columns=df.columns)], ignore_index=True)  # Append each line to the DataFrame
        df.to_excel(f'index_parser/output/{file}_{index}_{filename}.xlsx', index=False)  # Save the DataFrame to an Excel file
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
    def string_out(self, tuple_list):
        line_out = ""  # Initialize an empty string
        for item in tuple_list:  # Loop through each tuple in the list
            color = self.get_ansi_color_code(item[2])  # Get the color code
            line_out += f" {color}{item[0]}\033[0m"  # Append token with color
        return f"[CRF-Result]: {line_out}"  # Return the formatted string

    # Clean up the internal state of the object
    def clean(self):
        self.columns = []  # Reset columns
        self.keys = []  # Reset keys
        self.collect_labels()  # Recollect labels
        self.check_if_output_exists()  # Check if the output directory exists again
        self.list_with_converted_lines = []  # Reset the list of converted lines
