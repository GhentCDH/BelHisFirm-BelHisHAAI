import os
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from .processing.text_extraction2 import TextExtractor2
    from .processing.OCR2 import OCR
    from .CRF.predict_crf import Predict
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from processing.text_extraction2 import TextExtractor2
    from processing.OCR2 import OCR
    from CRF.predict_crf import Predict


class IndexParser:
    def __init__(self, debug=False):
        self.text_extractor = TextExtractor2(debug=debug)
        self.debug = debug  
        self.ocr_system = OCR()
        self.crf_predictor = Predict()
        self.crf_predictor.choose_model(Path("/home/bas/Documents/Visual Code Repos/BelHisFirm-BelHisHAAI/src/index_parser/model/CRF_1884.pkg"))
    
    def run(self, folder_path, index_start_page=None, index_end_page=None):
 
        for idx, image_path in enumerate(os.listdir(folder_path)):
            lines_ocr= []
            if image_path.endswith((".tif")):
                listje = self.text_extractor.extract_text_lines(os.path.join(folder_path, image_path))
                for line in tqdm(listje, desc="Running OCR"):
                    ocr_result = self.ocr_system.run(line[0])
                    lines_ocr.append((ocr_result, line[1]))
        
        combined_lines = []
        for line in lines_ocr:
            if line[1] and combined_lines:
                root_line = combined_lines[-1]
                new_line = root_line + " " + line[0]
                combined_lines[-1] = new_line
            else:
                combined_lines.append(line[0])

        for line in combined_lines:
            result = self.crf_predictor.predict(combined_lines, "output", 0)
        print(result)




if __name__ == "__main__":
    parser = IndexParser(debug=True)
    parser.run(str(Path(__file__).parent / "testdata"))