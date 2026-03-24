import argparse
import logging

from pathlib import Path

from src.recordprocessing.processor import RecordProcessor

if __name__ == "__main__":

    """
    app: BelhisApp = BelhisApp()
    app.run()
    """

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process historical records with OCR")
    parser.add_argument("input_folder", type=Path, nargs="?",
                        default=Path("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/images"),
                        help="Path to folder containing input images")
    parser.add_argument("output_folder", type=Path, nargs="?",
                        default=Path("/Users/sander/PycharmProjects/BelHisFirm-BelHisHAAI/output"),
                        help="Path to output folder for processed records")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR processing (only extract and save images)")
    args = parser.parse_args()

    processor = RecordProcessor(skip_ocr=args.no_ocr)
    processor.process_record(args.input_folder, args.output_folder)