import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_INDEX_PARSER_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_INDEX_PARSER_DIR))

from processing.text_extraction2 import TextExtractor2
from processing.OCR2 import OCR
from CRF.utils.convert_to_features import Convert_To_Features

_DELIMITERS = {".", ",", ";", ":", "!", "?", "(", ")", "°", "/", "&", '"', "—", "-"}


def make_ground_truth(
    image_path: str | Path,
    output_path: str | Path | None = None,
    exclude_strings: list[str] | None = None,
    exclude_whole: list[str] | None = None,
) -> Path:
    image_path = Path(image_path)
    exclude_strings = exclude_strings or []
    exclude_whole = exclude_whole or []

    text_extractor = TextExtractor2()
    ocr_system = OCR()

    # Extract line images from the page
    line_crops = text_extractor.extract_text_lines(str(image_path))

    # OCR each line, skip lines matching any exclude pattern
    ocr_results = []
    for crop, is_outlier in tqdm(line_crops, desc="Running OCR"):
        text = ocr_system.run(crop)
        if not text:
            continue
        if any(s in text for s in exclude_strings) or text in exclude_whole:
            continue
        ocr_results.append((text, is_outlier))

    # Combine continuation lines (outlier = indented, belongs to previous entry)
    combined_lines = []
    for text, is_outlier in ocr_results:
        if is_outlier and combined_lines:
            combined_lines[-1] += " " + text
        else:
            combined_lines.append(text)

    # Tokenize, auto-assign D for delimiters and START/END for sentinels
    ids, tokens, keys = [], [], []
    for sentence_id, line in enumerate(tqdm(combined_lines, desc="Tokenizing"), start=1):
        for token in Convert_To_Features.tokenize_string(line):
            ids.append(sentence_id)
            tokens.append(token)
            if token in ("START", "END"):
                keys.append(token)
            elif token in _DELIMITERS:
                keys.append("D")
            else:
                keys.append(None)

    df = pd.DataFrame({"id": ids, "value": tokens, "key": keys})

    if output_path is None:
        output_path = Path(__file__).parent.parent / "BelHisFirm-GT" / f"{image_path.stem}_GT.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Ground truth template saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ground truth template from a page image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    parser.add_argument(
        "--exclude", type=str, action="append", default=None, metavar="STRING",
        help="Exclude OCR lines containing this string (can be repeated)"
    )
    parser.add_argument(
        "--exclude-whole", type=str, action="append", default=None, metavar="STRING",
        help="Exclude OCR lines whose full text matches this string exactly (can be repeated)"
    )
    args = parser.parse_args()

    make_ground_truth(args.image_path, args.output, args.exclude, args.exclude_whole)

# Example usage: uv run src/index_parser/CRF/train/ground_truth_maker.py  "/home/bas/Documents/Visual Code Repos/BelHisFirm-BelHisHAAI/src/index_parser/testdata/EHC_B665_O_2025_1892_III-IV_0926.tif" --exclude "TABLE DU RECUEIL" --exclude "N° d'ordre" --exclude-whole "d'acte"