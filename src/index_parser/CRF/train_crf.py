from convert_to_features import Convert_To_Features
import sklearn_crfsuite
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

class Train:
    def __init__(self):
        self.model = None
        self.convert_to_features = Convert_To_Features()
        self.training_data_label, self.training_data_token = [], []
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = [], [], [], []

    def extract_ground_truth(self, csv_file):
        data = pd.read_csv(csv_file)
        data = data.astype(str)  # Convert all cells to string
        grouped_data = data.groupby('id')
        for _, group in tqdm(grouped_data, desc="\033[92m[LOADING]: \033[97mExtracting Ground Truth", ncols=100):
            sentence = " ".join(group['value'].tolist())
            labels = group['key'].tolist()
            tokens, labels = self.convert_to_features.tokenize_training_string(sentence, labels)
            features = self.convert_to_features.get_token_features_from_tokenized_list(tokens)
            self.training_data_token.append(features)
            self.training_data_label.append(labels)

    def create_test_set(self, size=0.1):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(
            self.training_data_token, self.training_data_label, test_size=size, random_state=69
        )

    def construct_model(self, c1, c2, max_iter):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=c1, c2=c2, max_iterations=max_iter, all_possible_transitions=False, verbose=True
        )

    def check_labels(self, csv_file):
        with open('index_parser/CRF/labels/labels.json') as json_file:
            valid_labels = set(json.load(json_file).keys())
        data = pd.read_csv(csv_file)
        for row, label in enumerate(data['key'], start=2):
            if label not in valid_labels and label not in {"START", "END"}:
                print(f"\033[91m[ERROR]: Invalid key in row {row} - {label}\033[97m")
                return False
        return True

    def train(self, ground_truth_path, model_name, c1, c2, max_it):
        if not self.check_labels(ground_truth_path):
            print("\033[91m[ERROR]: \033[97mCSV labels are invalid! Fix them before training.\033[97m")
            return

        self.extract_ground_truth(ground_truth_path)
        self.create_test_set()
        self.construct_model(c1, c2, max_it)

        print("\033[92m[TRAINING]: \033[97mTraining model...")
        self.model.fit(self.training_data_token, self.training_data_label)
        
        print("\033[92m[EVALUATION]: \033[97mGenerating classification report...")
        Y_Pred = self.model.predict(self.X_Test)
        Y_Test_flat = [label for seq in self.Y_Test for label in seq]
        Y_Pred_flat = [label for seq in Y_Pred for label in seq]
        
        accuracy = accuracy_score(Y_Test_flat, Y_Pred_flat)
        
        model_path = f'index_parser/model/{model_name}.pkg'
        joblib.dump(self.model, model_path)
        print(f"\033[92m[SAVED]: \033[97mModel saved as {model_path}\033[97m")
        return accuracy
