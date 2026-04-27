import joblib
from .output_crf import Output_CRF
from .convert_to_features import Convert_To_Features

# Class for making predictions using a pre-trained CRF model.
class Predict:
    
    def __init__(self):
        self.model = None  # Placeholder for the CRF model
        self.output = Output_CRF()  # Output_CRF instance to handle result formatting and saving

    # Method to load the model from the file system
    def choose_model(self, modelname):
        from pathlib import Path
        path = Path(modelname)
        if not path.exists():
            print(f"[ERROR] Model file not found: {path.resolve()}")
            return None
        try:
            self.model = joblib.load(path)
        except Exception as e:
            print(f"[ERROR] Failed to load model '{path.resolve()}': {e}")
        return self.model

    # Method to predict labels for a given set of sentences (input data)
    def predict(self, source, file, index):
        sentences_to_be_tranformed = []  # List to store the transformed output for each sentence
        
        # Iterate over each sentence in the source input
        for item in source:
            # Tokenize the input sentence
            tokenized_input = Convert_To_Features.tokenize_string(item)
            # Extract features from the tokenized sentence for CRF model
            features = Convert_To_Features.get_token_features_from_tokenized_list(tokenized_input)
            # Use the loaded model to predict the labels based on the extracted features
            predicted_labels = self.model.predict([features])
            # Create a list of tuples representing the token, predicted label, and color index
            tuple_list = self.output.create_tuple_from_prediction(tokenized_input, predicted_labels)
            sentences_to_be_tranformed.append(tuple_list)  # Add the transformed sentence to the list
            # Output the result to the console with color-coded tokens
        # Output the result to the console with color-coded tokens
        output_string = self.output.string_out(tuple_list=tuple_list)

        # Transform the list of sentence tuples into a format suitable for CSV/Excel
        transformed_lines_for_csv = self.output.transform_lines_to_csv_format(sentences_to_be_tranformed)
        # Save the transformed data as an Excel file
        self.output.create_excel(transformed_lines_for_csv, file, index)
        sentences_to_be_tranformed = []  # Reset the list after processing
        return output_string
