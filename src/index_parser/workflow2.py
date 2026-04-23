from processing.text_extraction2 import TextExtractor2
from recordprocessing.OCR import OCRProcessor
import os


class IndexParser:
    def __init__(self, debug=False):
        self.text_extractor = TextExtractor2(debug=debug)
    
    def run(self, folder_path, index_start_page=None, index_end_page=None):
 
        for image_path in os.listdir(folder_path):
            if image_path.endswith((".tif")):
                listje = self.text_extractor.extract_text_lines(os.path.join(folder_path, image_path))
            
        
    

        









if __name__ == "__main__":
    parser = IndexParser(debug=False)
    parser.run("src/index_parser/testdata")