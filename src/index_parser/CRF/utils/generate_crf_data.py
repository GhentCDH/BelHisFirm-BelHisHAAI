import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.OCR import OCR_Qwen
from processing.text_extraction import Text_Line_Extractor as TextExtraction
import pandas as pd
from tqdm import tqdm
import shutil
from pathlib import Path

def main(image_path):
    # Initialize the OCR and TextExtraction classes
    ocr = OCR_Qwen()
    text_extraction = TextExtraction()
    _base = Path(__file__).parent.parent
    temp_path = _base / "~temp"
    output = _base / "output"
    output_file = output / f"{Path(image_path).name}_ocr_results.txt"

    def create_temp_folder():
        if not temp_path.exists():
            temp_path.mkdir(parents=True)
        if not output.exists():
            output.mkdir(parents=True)
    
    create_temp_folder()  # Create the temporary folder if it doesn't exist
        
    def process_csv(file):
            try:
                data = pd.read_csv(os.path.join(temp_path, file))
            except Exception as e:
                print(f"\033[91m[ERROR]: \033[97mError reading CSV file {file}: {e}")
                return

            # OCR
            progress_bar = tqdm(data.index, desc=f"\033[92m[OCR]:\033[97m Performing OCR", unit="line", ncols=120)
            tuple_list = []  # List to store extracted text and coordinates
            index_watcher = 0

            for index in progress_bar:
                current_id = data['id'][index]  # Get current ID
                if index_watcher == current_id:
                    image_path = os.path.join(str(current_id), data['image'][index])
                    full_path = Path(temp_path, image_path).resolve()
                    text = ocr.run_inference(full_path)  # Run
                    if text != 'None':
                        if '\\n' in text: # re fix for multiple lines in one image
                            tuple_list.append((text.split('\\n')[0], data['Xnorm'][index]))
                            tuple_list.append((text.split('\\n')[1], data['Xnorm'][index]))
                        else:
                            tuple_list.append((text, data['Xnorm'][index]))  # Store result

                    # Update the progress bar with the OCR result
                    progress_bar.set_description(f"\033[92m[OCR]:\033[97m Last Result: {text[:40]}...")
                else:
                    index_watcher += 1 # Move to next ID
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for text, _ in tuple_list:
                            f.write(f"{text}\n")

            # Write all OCR results directly to a text file
            with open(output_file, 'a', encoding='utf-8') as f:
                for text, _ in tuple_list:
                    f.write(f"{text}\n")
                
            progress_bar.close()  # Close progress bar when don

            shutil.rmtree(temp_path)  # Clean up temporary storage folder
            create_temp_folder()  # Recreate the temp folder for the next image
            text_extraction.reset()  # Reset the text extraction object
            

    # Extract lines of text from the image
    text_extraction.text_line_extractor(image_path, 1, temp_folder=temp_path)
    # Process each CSV file generated from text extraction
    for root, dirs, files in os.walk(temp_path):
        for file in files:
            if file.endswith(".csv"):
                process_csv(file)
    
    return output_file
    
        
if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "input"
    for file in os.listdir(input_path):
        image_path = os.path.join(input_path, file)
        output_file = main(image_path)