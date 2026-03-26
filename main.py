import argparse
import logging

from pathlib import Path

from src.belhisapp.app import BelhisApp
from src.recordprocessing.record_processor import RecordProcessor

from huggingface_hub import logging as hf_logging

hf_logging.set_verbosity_error()

app = BelhisApp()
app.run()

"""

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Process historical records with OCR")
parser.add_argument("input_folder", type=Path, nargs="?",
                        default=Path("/home/sndr/PycharmProjects/belhisfirm-belhishaai/images"),
                        help="Path to folder containing input images")
parser.add_argument("output_folder", type=Path, nargs="?",
parser.add_argument("output_folder", type=Path, nargs="?",
                        default=Path("/home/sndr/PycharmProjects/belhisfirm-belhishaai/output"),
                        help="Path to output folder for processed records")
parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR processing (only extract and save images)")
args = parser.parse_args()

processor = RecordProcessor(skip_ocr=args.no_ocr)
processor.run(args.input_folder, args.output_folder)

"""