import os
import sys
import datetime
import pandas as pd
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

_OUTPUT_DIR = Path(__file__).parent / "output"


class IndexParser:
    def __init__(self, debug=False):
        self.text_extractor = TextExtractor2(debug=debug)
        self.debug = debug
        self.ocr_system = OCR()
        self.crf_predictor = Predict("src/index_parser/model/1892-V3.pkg")

    def run(self, folder_path, index_start_page=None, index_end_page=None):
        lines_ocr = []
        for idx, image_path in enumerate(sorted(os.listdir(folder_path))):
            if image_path.endswith(".tif"):
                listje = self.text_extractor.extract_text_lines(os.path.join(folder_path, image_path))
                for line in tqdm(listje, desc=f"Running OCR on {image_path}"):
                    ocr_result = self.ocr_system.run(line[0])
                    lines_ocr.append((ocr_result, line[1]))

        combined_lines = []
        for line in lines_ocr:
            if line[1] and combined_lines:
                combined_lines[-1] = combined_lines[-1] + " " + line[0]
            else:
                combined_lines.append(line[0])

        predicted_texts = []
        for line in combined_lines:
            if "TABLE DU RECUIL" in line or "N°" in line or line == "de l'acte":
                continue
            self.crf_predictor.predict_single_line(line, self.debug)
            predicted_texts.append(line)

        self._save_excel(predicted_texts)

    def _save_excel(self, predicted_texts):
        crf_rows = self.crf_predictor.get_output_no_punctuation()
        columns = self.crf_predictor.output.columns

        rows = []
        for crf_row, full_text in zip(crf_rows, predicted_texts):
            row = dict(zip(columns, crf_row))
            row["Full Text"] = full_text
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns + ["Full Text"])

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_path = _OUTPUT_DIR / f"output_{timestamp}.xlsx"
        df.to_excel(output_path, index=False)
        print(f"[Output] Saved {len(rows)} records to {output_path}")


if __name__ == "__main__":
    parser = IndexParser(debug=True)
    parser.run(str(Path(__file__).parent / "testdata"))