import argparse
import re
import sys
import datetime
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from .processing.text_extraction2 import TextExtractor2
    from .processing.OCR2 import OCR
    from .processing.ocr_postprocessor import OCRPostProcessor
    from .processing.crf_preprocessor import CRFPreProcessor
    from .processing.crf_postprocessor import CRFPostProcessor
    from .processing.excel_formatter import ExcelFormatter
    from .CRF.predict_crf import Predict
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from processing.text_extraction2 import TextExtractor2
    from processing.OCR2 import OCR
    from processing.ocr_postprocessor import OCRPostProcessor
    from processing.crf_preprocessor import CRFPreProcessor
    from processing.crf_postprocessor import CRFPostProcessor
    from processing.excel_formatter import ExcelFormatter
    from CRF.predict_crf import Predict

_OUTPUT_BASE_DIR = Path(__file__).parent / "output"
_IMAGE_SCALE = 0.25


class IndexParser:
    _DEFAULT_MODEL = str(Path(__file__).parent / "model" / "1892-V4.pkg")

    def __init__(self, model_path=None, debug_mode=None, binarize=False):
        self.text_extractor = TextExtractor2(debug=(debug_mode == "bbox"), binarize=binarize)
        self.debug_mode = debug_mode
        self.ocr_system = OCR()
        self.crf_predictor = Predict(model_path or self._DEFAULT_MODEL)
        self.postprocessor = OCRPostProcessor()
        self.crf_preprocessor = CRFPreProcessor()
        self.crf_postprocessor = CRFPostProcessor()
        self.excel_formatter = ExcelFormatter()

    def run(self, folder_path, index_start_page=None, index_end_page=None, output_dir=None):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_dir = Path(output_dir) / timestamp if output_dir else _OUTPUT_BASE_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        folder = Path(folder_path)
        selected = []
        for image_path in sorted(folder.glob("*.tif")):
            match = re.search(r'_(\d+)\.tif$', image_path.name)
            page_num = int(match.group(1)) if match else None
            if index_start_page is not None and (page_num is None or page_num < index_start_page):
                continue
            if index_end_page is not None and (page_num is None or page_num > index_end_page):
                continue
            selected.append(image_path)

        if not selected:
            print("[Warning] No images matched the given page range.")
            return

        for image_path in selected:
            self._save_thumbnail(image_path, output_dir / image_path.name)

            extracted = self.text_extractor.extract_text_lines(image_path, debug_dir=output_dir)

            if self.debug_mode == "bbox":
                continue

            page_lines = []
            for text, is_continuation in tqdm(extracted, desc=f"Running OCR on {image_path.name}"):
                ocr_result = self.ocr_system.run(text)
                if self.debug_mode == "ocr":
                    print(f"[OCR] {ocr_result}")

                ocr_result = self.crf_preprocessor.clean_before_merge(ocr_result)

                parts = [p.strip() for p in re.split(r'\r\n|\r|\n', ocr_result) if p.strip()]
                if len(parts) > 1:
                    print(f"[Warning] OCR produced {len(parts)} lines in one detection on {image_path.name}, splitting")

                for i, part in enumerate(parts):
                    if i == 0 and is_continuation and page_lines:
                        page_lines[-1] = self.postprocessor.combine(page_lines[-1], part)
                    else:
                        page_lines.append(part)

            page_lines = self.postprocessor.process(page_lines)

            seen = set()
            deduped = []
            for line in page_lines:
                if line in seen:
                    print(f"[Warning] Duplicate line removed on {image_path.name}: '{line[:60]}'")
                else:
                    seen.add(line)
                    deduped.append(line)
            page_lines = deduped

            if self.debug_mode == "ocr":
                continue

            predicted_texts = []
            for line in page_lines:
                if self.crf_preprocessor.should_ignore(line):
                    continue
                self.crf_predictor.predict_single_line(line, debug=(self.debug_mode == "crf"))
                predicted_texts.append(line)

            self._save_excel(predicted_texts, output_dir, image_path.stem + ".xlsx")
            self.crf_predictor.reset()

    def _save_thumbnail(self, src_path, dest_path):
        img = Image.open(src_path)
        w, h = img.size
        img = img.resize((int(w * _IMAGE_SCALE), int(h * _IMAGE_SCALE)), Image.LANCZOS)
        img.save(dest_path)

    def _save_excel(self, predicted_texts, output_dir, filename):
        crf_rows = self.crf_predictor.get_output_no_punctuation()
        columns = self.crf_predictor.output.columns

        rows = []
        for crf_row, full_text in zip(crf_rows, predicted_texts):
            row = dict(zip(columns, crf_row))
            row["Full Text"] = full_text
            rows.append(row)

        df = pd.DataFrame(rows, columns=list(columns) + ["Full Text"])
        df = self.crf_postprocessor.process(df)

        output_path = output_dir / filename
        df.to_excel(output_path, index=False)
        self.excel_formatter.format(output_path)
        print(f"[Output] Saved {len(rows)} records to {output_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run IndexParser on a folder of .tif images.")
    arg_parser.add_argument("--folder", type=str, default=str(Path(__file__).parent / "testdata"), help="Path to folder with .tif images")
    arg_parser.add_argument("--start-page", type=int, default=None, help="First page to process (1-based, inclusive)")
    arg_parser.add_argument("--end-page", type=int, default=None, help="Last page to process (1-based, inclusive)")
    arg_parser.add_argument("--model", type=str, default=None, help="Path to CRF model file")
    arg_parser.add_argument("--output", type=str, default=None, help="Base output folder (default: output/ next to this script)")
    arg_parser.add_argument("--binarize", action="store_true", help="Convert line crops to black-and-white using Otsu thresholding before OCR")
    arg_parser.add_argument("--debug", choices=["bbox", "ocr", "crf"], default=None,
                            help="bbox: save bbox images only; ocr: print OCR output, skip CRF; crf: run all, show CRF in terminal")
    args = arg_parser.parse_args()

    index_parser = IndexParser(model_path=args.model, debug_mode=args.debug, binarize=args.binarize)
    index_parser.run(args.folder, index_start_page=args.start_page, index_end_page=args.end_page, output_dir=args.output)