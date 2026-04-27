import os
import pandas as pd
import re
from pathlib import Path

def convert_to_valid_gt(path):
    all_words = []
    ids = []
    counter = 1
    for file in os.listdir(path):
        with open(os.path.join(os.path.join(path, file)), 'r', encoding='utf-8') as doc:
                for line in doc.readlines():
                    all_words.append('START')
                    ids.append(counter)
                    line = line.replace('—', '-')
                    line = line.replace('  ', ' ')
                    line = line.replace('"','')
                    line = line.strip()
                    line = re.split(r'[ ,]+', line)
                    for word in line:
                        all_words.append(word)
                        ids.append(counter)
                    all_words.append('END')
                    ids.append(counter)
                    counter += 1
    df = pd.DataFrame({'id': ids, 'value': all_words, 'key': None})
    df.to_csv(Path(__file__).parent / 'BelHisFirm-GT' / 'ground_truth_1913.csv', index=False, decimal=';')

def convert_to_valid_single(file_path):
    all_words = []
    ids = []
    counter = 1

    with open(file_path, 'r', encoding='utf-8') as doc:
        for line in doc.readlines():
            all_words.append('START')
            ids.append(counter)

            # Clean and split the line
            line = line.replace('—', '-')
            line = line.replace('  ', ' ')
            line = line.replace('"', '')
            line = line.strip()
            words = re.split(r'[ ,]+', line)
            processed_words = []
            for word in words:
                if word == '-' and processed_words:  # If word is a hyphen and we have processed words
                    processed_words[-1] += '-'  # Append the hyphen to the last word
                else:
                    processed_words.append(word)
            words = processed_words

            for word in words:
                all_words.append(word)
                ids.append(counter)

            all_words.append('END')
            ids.append(counter)
            counter += 1

    df = pd.DataFrame({'id': ids, 'value': all_words, 'key': None})
    # Set key values for START and END
    for idx, row in df.iterrows():
        if row['value'] == 'START':
            df.at[idx, 'key'] = 'start'
        elif row['value'] == 'END':
            df.at[idx, 'key'] = 'END'
    df.to_csv(os.path.splitext(file_path)[0] + '.csv', index=False, sep=';', decimal=',')  # fixed decimal & sep

    print(f"Processed file: {file_path}")

def update_keys_in_csv(csv_path):
        """
        Load a CSV file, update 'key' values for 'START' and 'END' entries, and save it back.
        
        Args:
            csv_path: Path to the CSV file to process
        """
        # Load the CSV file
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        
        # Update key values for START and END
        for idx, row in df.iterrows():
            if row['value'] == 'START':
                df.at[idx, 'key'] = 'START'
            elif row['value'] == 'END':
                df.at[idx, 'key'] = 'END'
        
        # Save the updated CSV
        output_path = csv_path.replace('.csv', '_updated.csv')
        df.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"Updated and saved: {output_path}")

if __name__ == "__main__":
    convert_to_valid_single(str(Path(__file__).parent.parent / 'output' / 'CRF_Text.txt'))
    #update_keys_in_csv('index_parser/CRF/BelHisFirm-GT/1897_GT.csv')