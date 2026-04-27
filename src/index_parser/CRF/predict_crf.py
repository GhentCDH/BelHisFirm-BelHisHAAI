from openpyxl.styles.builtins import output
import joblib
from pathlib import Path
import datetime

try:
    from .utils.output_crf import Output_CRF
    from .utils.convert_to_features import Convert_To_Features
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.output_crf import Output_CRF
    from utils.convert_to_features import Convert_To_Features

# Class for making predictions using a pre-trained CRF model.
class Predict:
    
    def __init__(self, model_path):

        self.output_lines = []
        self.output = Output_CRF()  # Output_CRF instance to handle result formatting and saving
        self.model = None

        # Load CRF model
        path = Path(model_path)
        if not path.exists():
            print(f"[ERROR] Conditional Random Fields Model (CRF) file not found. Please make sure to select a valid model file: {path.resolve()}")
            return None
        try:
            self.model = joblib.load(path)
        except Exception as e:
            print(f"[ERROR] Failed to load Conditional Random Fields Model (CRF) model '{path.resolve()}': {e}")


    # Method to predict labels for a given set of sentences (input data)
    def predict_single_line(self, line, debug=False):

        if self.model is None:
            print("[ERROR] No Conditional Random Fields Model (CRF) loaded. Please load a model before making predictions.")
            return None
        else:
            # Tokenize the input sentence
            tokenized_input = Convert_To_Features.tokenize_string(line)
            # Extract features from the tokenized sentence for CRF model
            features = Convert_To_Features.generate_token_features(tokenized_input)
            # Use the loaded model to predict the labels based on the extracted features
            predicted_labels = self.model.predict([features])
            # Create a list of tuples representing the token, predicted label, and color index
            tuple_list = self.output.create_tuple_from_prediction(tokenized_input, predicted_labels)
            self.output_lines.append(tuple_list)
            # Output the result to the console with color-coded tokens
            output_string = self.output.colored_prediction(tuple_list=tuple_list)
            output_string_labels = self.output.label_prediction(tuple_list=tuple_list)
            if debug:
                print(output_string)
                print(output_string_labels)
    
    def get_output_no_punctuation(self):
        list_with_converted_lines = []

        for predicted_line in self.output_lines:
            converted_line = self.output.transform_line_to_csv_format(
                predicted_line,
                clean_delimiters=True
            )
            list_with_converted_lines.append(converted_line)

        return list_with_converted_lines


