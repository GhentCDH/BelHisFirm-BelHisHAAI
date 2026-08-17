import os
import time
import sys
import shutil
from pathlib import Path
import pandas as pd
from statistics import stdev, mean
from .processing.text_extraction import Text_Line_Extractor
from .processing.OCR import OCR_Qwen
from .CRF.predict_crf import Predict
from .processing.VIN import verify_intendation
from tqdm import tqdm
import re


class Workflow:
    # Constructor
    def __init__(self, input_path, model_path):
        #self.folder_of_images = input_path if input_path else "index_parser/input"
        if input_path is not None:
            self.folder_of_images = input_path
        else:
            self.folder_of_images = "index_parser/input"
        self.temp_path = Path("index_parser/~temp").expanduser()  # Temporary path for intermediate processing
        #self.crf_model_name = model_path if model_path else "index_parser/model/CRF_1800.pkg"  # Path to the CRF model file
        if model_path is not None:
            self.crf_model_name = model_path
        else: 
            self.crf_model_name = "index_parser/model/CRF_1884.pkg"

        self.splits = 1  # Number of splits for image processing (to be adjusted as needed)
        self.is_indentation_beginning_of_sentence = False  # Whether indentation is considered the start of a new sentence

        # Initialize models
        self.text_line_extractor = None
        self.ocr_predict = None
        self.crf_predict = None

        # Load the necessary models and classes
        self.load()

    # Function to load the necessary classes and models
    def load(self):
        tasks = [
            ("Loading classes...", lambda: setattr(self, 'text_line_extractor', Text_Line_Extractor())),
            ("Loading processing models...", lambda: setattr(self, 'crf_predict', Predict())),
            ("Initializing OCR model...", lambda: setattr(self, 'ocr_predict', OCR_Qwen())),
            ("Choosing CRF model...", lambda: self.crf_predict.choose_model(self.crf_model_name))
        ]

        with tqdm(total=len(tasks), desc="Initializing...", bar_format="\033[97m{desc} {bar} {n_fmt}/{total_fmt}", ncols=60) as pbar:
            for desc, task in tasks:
                pbar.set_description(f"\033[92m[LOADING]: \033[97m{desc}")  # Update progress bar description
                result = task()
                time.sleep(1)  # Longer sleep to make it more visible
                pbar.update(1)
                pbar.set_description("")  # Clear the description after each task

            # After all tasks are done, show a final description
            for _ in range(3):
                print("\033[F", end="")  # Move the cursor up 1 line
                print("\033[K", end="")  # Clear the line
            pbar.set_description(f"\033[92m[LOADING]: \033[97mLoading complete!")
            
        self.process_images()  # Continue with image processing

    # Create the temporary storage folder if it doesn't exist
    def create_temp_folder(self):
        if not self.temp_path.exists():
            self.temp_path.mkdir(parents=True)

    # Process each image file in the input folder
    def process_images(self):
        total_start = time.time()
        print(f"\033[92m[INDEXPARSER]: \033[97mBelHisFirm index to table workflow V.1.1. - @bavercru, @m9k")

        self.create_temp_folder()  # Ensure temp folder exists

        try:
            for image in os.listdir(self.folder_of_images):
                image_path = os.path.join(self.folder_of_images, image)
                self.process_image(image_path)  # Process each image
                
        except KeyboardInterrupt:
            print("\n\033[91m[ERROR]: \033[97mProcess interrupted by user. Cleaning up...\033[0m")
            self.cleanup()  # Perform any necessary cleanup here
            sys.exit(0)  # Exit gracefully

        total_stop = time.time()
        total = total_stop - total_start
        sys.stdout.write(f"| Total time in seconds: {total:.2f} | {self.crf_model_name}\n")
        print(f"\033[94m[INDEXPARSER] \033[97mFinished!")  # Indicate the end of the processing
        self.cleanup()
    
    # Add a cleanup method to handle any cleanup operations when interrupted
    def cleanup(self):
        shutil.rmtree(self.temp_path)  # Clean up temporary storage folder
        self.create_temp_folder()  # Recreate the temp folder
        self.ocr_predict.clean_up_ocr()  # Clean up OCR-related files
        print("\033[92m[CLEANUP]: \033[97mComplete.")

    # Process a single image
    def process_image(self, image_path):
        print(f"\033[92m[PROCESSING]: \033[97mProcessing image: {image_path}...")
        # Extract lines of text from the image
        self.text_line_extractor.text_line_extractor(image_path, self.splits, temp_folder=self.temp_path)

        # Process each CSV file generated from text extraction
        for root, dirs, files in os.walk(self.temp_path):
            for file in files:
                if file.endswith(".csv"):
                    self.process_csv(file)

    def proces_text_list(self, tuple_list, index_watcher, progress_bar, file):
        text_list = []
        tuple_list_coord = []
        index_tuple_counter = 0

        pattern = r'^\d+[.,:;]'

        for tuple_item in tuple_list:
                if re.match(pattern, tuple_item[0]):
                    tuple_list_coord.append(tuple_item[1])
                
        for i, tuple_item in enumerate(tuple_list):
            if re.match(pattern, tuple_item[0]):
                    coord_min_2  = tuple_list_coord[index_tuple_counter-2] if (index_tuple_counter-2) >= 0 else 1
                    coord_min_1 = tuple_list_coord[index_tuple_counter-1] if (index_tuple_counter-1) >= 0 else 1
                    coord_index  = tuple_list_coord[index_tuple_counter]
                    coord_plus_1 = tuple_list_coord[index_tuple_counter+1] if (index_tuple_counter+1) < len(tuple_list_coord) else 1
                    coord_plus_2 = tuple_list_coord[index_tuple_counter+2] if (index_tuple_counter+2) < len(tuple_list_coord) else 1
                    coord_list = [coord_min_2, coord_min_1, coord_index, coord_plus_1, coord_plus_2]
                    index_tuple_counter += 1
                    if verify_intendation(coord_list) == True:
                        try:
                            text_list[-1] = f'{text_list[-1]} {tuple_item[0]}'
                        except IndexError:
                            text_list.append(tuple_item[0])
                    else:
                        text_list.append(tuple_item[0])
            else:
                if text_list:
                    try:
                        if text_list[-1][-1] == '-':
                            text_list[-1] = f'{text_list[-1]}{tuple_item[0]}'
                        else:
                            text_list[-1] = f'{text_list[-1]} {tuple_item[0]}'
                    except IndexError:
                        text_list.append(tuple_item[0])
                else:
                    text_list.append(tuple_item[0])
        
        if text_list:
            print(f"\033[92m[CRF]: \033[97mPredicting...")
            crf_result = self.crf_predict.predict(text_list, file, index_watcher)  # Process CRF
            if crf_result is None:
                print("\033[91m[ERROR]: CRF prediction returned None! Please check the contents of the file. Continuing...")
            else:
                progress_bar.set_description(f"\033[94m[CRF]:\033[97m {crf_result[:40]}...")

    # Process a CSV file containing text line extraction data
    def process_csv(self, file):
        for _ in range(6):
            print("\033[F", end="")  # Move the cursor up 1 line
            print("\033[K", end="")  # Clear the line
        print(f"\033[92m[PROCESSING]: \033[97mProcessing CSV file: {file}...")

        processing_start = time.time()

        try:
            data = pd.read_csv(os.path.join(self.temp_path, file))
        except Exception as e:
            print(f"\033[91m[ERROR]: \033[97mError reading CSV file {file}: {e}")
            return

        # OCR
        progress_bar = tqdm(data.index, desc=f"\033[92m[OCR]:\033[97m Performing OCR", unit="line", ncols=120)
        tuple_list = []  # List to store extracted text and coordinates
        index_watcher = 0

        OCR_start = time.time()  # Start timing the OCR process
        for index in progress_bar:
            current_id = data['id'][index]  # Get current ID
            if index_watcher == current_id:
                image_path = os.path.join(str(current_id), data['image'][index])
                full_path = Path(self.temp_path, image_path).resolve()
                text = self.ocr_predict.run_inference(full_path)  # Run
                if text != 'None':
                    if '\\n' in text: # re fix for multiple lines in one image
                        tuple_list.append((text.split('\\n')[0], data['Xnorm'][index]))
                        tuple_list.append((text.split('\\n')[1], data['Xnorm'][index]))
                    else:
                        tuple_list.append((text, data['Xnorm'][index]))  # Store result

                # Update the progress bar with the OCR result
                progress_bar.set_description(f"\033[92m[OCR]:\033[97m Last Result: {text[:40]}...")
            else:
                self.proces_text_list(tuple_list, index_watcher, progress_bar, file)  # Process the text list
                index_watcher += 1 # Move to next ID
                tuple_list = []
        self.proces_text_list(tuple_list, index_watcher, progress_bar, file)  # Process the last batch
              
        
        OCR_stop = time.time()
        OCR_total = OCR_stop - OCR_start             
        progress_bar.close()  # Close progress bar when done

        processing_stop = time.time()
        processing_total = processing_stop - processing_start  # Time spent processing the current file
        print(f"\033[92m[RESULT]: \033[97mIndex {index_watcher} of {file} | OCR Time: {OCR_total:.2f}s | Total Processing Time: {processing_total:.2f}s", end=" ")

        shutil.rmtree(self.temp_path)  # Clean up temporary storage folder
        self.create_temp_folder()  # Recreate the temp folder for the next image
        self.text_line_extractor.reset()

"""
# Main entry point of the script
if __name__ == "__main__":
    input_path = "path_to_input_folder"  # You can specify the input folder here
    main_instance = Main(input_path)  # Create instance of the Main class
    main_instance.process_images()  # Start processing images"
"""
