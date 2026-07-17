import argparse
import datetime
import gc
import itertools
import logging
import re
import sys
import traceback
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

_log = logging.getLogger("index_parser")

# find ./ -maxdepth 3 -name "1889"

_RAINBOW = itertools.cycle([
    "\033[1;38;2;255;0;0m",      # red
    "\033[1;38;2;255;128;0m",    # orange
    "\033[1;38;2;255;255;0m",    # yellow
    "\033[1;38;2;0;255;0m",      # green
    "\033[1;38;2;0;255;255m",    # cyan
    "\033[1;38;2;0;128;255m",    # blue
    "\033[1;38;2;255;0;255m",    # magenta
])
_RESET = "\033[0m"


class _RainbowFormatter(logging.Formatter):
    def format(self, record):
        return next(_RAINBOW) + super().format(record) + _RESET


def _setup_log(output_dir):
    for h in list(_log.handlers):
        _log.removeHandler(h)
        h.close()
    _log.setLevel(logging.INFO)
    _log.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(output_dir / "processing.log", encoding="utf-8")
    fh.setFormatter(fmt)
    _log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(_RainbowFormatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(ch)

try:
    from .processing.text_extraction2 import TextExtractor2
    from .processing.OCR2 import OCR
    from .processing.ocr_postprocessor import OCRPostProcessor
    from .processing.crf_preprocessor import CRFPreProcessor, VALID_QUARTERS
    from .processing.crf_postprocessor import CRFPostProcessor
    from .processing.excel_formatter import ExcelFormatter
    from .processing.rule_based_merger import RuleBasedMerger
    from .processing.index_locator import IndexLocator
    from .CRF.predict_crf import Predict
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from processing.text_extraction2 import TextExtractor2
    from processing.OCR2 import OCR
    from processing.ocr_postprocessor import OCRPostProcessor
    from processing.crf_preprocessor import CRFPreProcessor, VALID_QUARTERS
    from processing.crf_postprocessor import CRFPostProcessor
    from processing.excel_formatter import ExcelFormatter
    from processing.rule_based_merger import RuleBasedMerger
    from processing.index_locator import IndexLocator
    from CRF.predict_crf import Predict

_OUTPUT_BASE_DIR = Path(__file__).parent / "output"
_IMAGE_SCALE = 0.50
_INDEX_OUTPUT_DIRNAME = "Index_Output"
_INDEX_KIND_LABELS = {
    "yearly": "jaarlijkse index",
    "regular": "reguliere (enkele) index",
    "custom": "index (custom anchor)",
}


def _derive_output_dir(folder):
    """Guess the sibling Index_Output/<year>[_<subpart>] dir for input folders shaped
    like .../<collection>/<date>/TIF/<year> or .../<collection>/<date>/TIF/<year>/<subpart>
    (multi-part volumes are sometimes split into Roman-numeral subfolders, e.g. "IV" or
    "III-IV"). Returns None if the folder doesn't match either shape.

    Keying off <year> (rather than folder.name) matters here: folder.name alone would be
    just "IV" for a subpart folder, colliding with every other year's "IV" subpart."""
    parts = folder.resolve().parts
    for i, part in enumerate(parts):
        if part.upper() == "TIF" and i >= 3 and i + 1 < len(parts):
            collection_root = Path(*parts[:i - 2])
            year = parts[i + 1]
            subparts = parts[i + 2:]
            suffix = "_".join([year, *subparts])
            return collection_root / _INDEX_OUTPUT_DIRNAME / suffix
    return None


_GROUPED_FORMAT_MIN_YEAR = 1912


def _extract_year(folder):
    """Extracts the year from a folder path shaped like .../<collection>/<date>/TIF/<year>
    or .../<collection>/<date>/TIF/<year>/<subpart>. Returns None if the path doesn't match."""
    parts = folder.resolve().parts
    for i, part in enumerate(parts):
        if part.upper() == "TIF" and i + 1 < len(parts):
            year_part = parts[i + 1]
            if year_part.isdigit() and len(year_part) == 4:
                return int(year_part)
    return None


def _next_run_dir(base_dir, suffix):
    """Returns base_dir/<suffix>_N for the lowest N (starting at 1) not already taken."""
    n = 1
    while (base_dir / f"{suffix}_{n}").exists():
        n += 1
    return base_dir / f"{suffix}_{n}"


def _next_queue_entry(queue_file, skip=None):
    """Returns (line_index, folder_path) for the first non-blank, non-comment line
    in the queue file (skipping any path already in skip), or None if there isn't
    one."""
    skip = skip or set()
    lines = queue_file.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped not in skip:
            return idx, stripped
    return None


def _remove_queue_line(queue_file, line_index):
    """Removes a single line by index, leaving every other line (including blanks
    and comments) untouched — so lines appended to the file while it was being
    processed are preserved."""
    lines = queue_file.read_text(encoding="utf-8").splitlines()
    del lines[line_index]
    text = "\n".join(lines)
    queue_file.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _append_processed_log(queue_file, folder_path):
    """Appends a completed folder path to a 'processed.txt' file next to the queue
    file — a persistent record of what's been run, independent of queue.txt itself
    (which only ever reflects what's still pending)."""
    log_file = queue_file.with_name("processed.txt")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp}  {folder_path}\n")


def _process_queue_file(index_parser, queue_file, **run_kwargs):
    """Processes folders listed one-per-line in queue_file, removing each line once
    done. Re-reads the file before every entry, so paths appended to the file while
    a previous (possibly long-running) folder is being processed are picked up too.
    Stops as soon as the file has no pending entries left — it does not wait/poll
    for future ones."""
    # Printed rather than logged: _log's handlers belong to whichever folder is currently
    # being processed (reset per-folder by _setup_log inside run()), so a queue-level
    # announcement here would either go nowhere (before the first folder) or land in the
    # previous folder's processing.log.
    queue_file = Path(queue_file)
    if not queue_file.exists():
        print(f"Queue-bestand bestaat niet: {queue_file}")
        return

    failed = set()
    while True:
        entry = _next_queue_entry(queue_file, skip=failed)
        if entry is None:
            print("Geen mappen meer in de queue — gestopt.")
            return
        line_index, folder_path = entry
        print(f"Queue: verwerken van {folder_path}")
        try:
            index_parser.run(folder_path, **run_kwargs)
        except Exception:
            print(f"Fout bij verwerken van {folder_path} — regel blijft in de queue staan "
                  f"(wordt niet nogmaals geprobeerd deze sessie)")
            traceback.print_exc()
            failed.add(folder_path)
        else:
            _remove_queue_line(queue_file, line_index)
            _append_processed_log(queue_file, folder_path)
        finally:
            index_parser.unload_models()


# A bare "<id>.<quarter>" marker (e.g. "2877.2.") looks digits-only after stripping
# periods, but it's a meaningful modern-format entry boundary, not noise like a stray
# page number ("36") — it must reach the rule-based merger, not be discarded here.
_ID_QUARTER_ONLY = re.compile(r'^\d+\.\d\.?$')


def _is_digits_only(text):
    stripped = text.strip()
    if _ID_QUARTER_ONLY.match(stripped):
        return False
    cleaned = re.sub(r'[\s.,]', '', stripped)
    return bool(cleaned) and cleaned.isdigit()


def _is_capitals_only(text):
    letters = re.sub(r'[\d\s.,]', '', text.strip())
    return bool(letters) and letters.isupper()


class _PageLogHandler(logging.Handler):
    """Captures log records for a single page so they can be written to _explain.txt."""
    def __init__(self):
        super().__init__()
        self.messages = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        self.messages.append(self.format(record))

    def clear(self):
        self.messages.clear()


class IndexParser:
    _DEFAULT_MODEL = str(Path(__file__).parent / "model" / "19thcenturyMoniteurV7.pkg")

    def __init__(self, model_path=None, debug_mode=None, binarize=False):
        self.model_path = model_path or self._DEFAULT_MODEL
        self.debug_mode = debug_mode
        self.binarize = binarize
        self.postprocessor = OCRPostProcessor()
        self.crf_preprocessor = CRFPreProcessor()
        self.crf_postprocessor = CRFPostProcessor()
        self.excel_formatter = ExcelFormatter()
        self.rule_merger = RuleBasedMerger()

        # Loaded lazily: no need to pull the OCR/CRF models into GPU memory for a --dry
        # index-detection scan, or the index locator's models when pages are given explicitly.
        self.text_extractor = None
        self.ocr_system = None
        self.crf_predictor = None
        self.index_locator = None

    def _ensure_pipeline_models(self):
        if self.text_extractor is None:
            self.text_extractor = TextExtractor2(debug=(self.debug_mode == "bbox"), binarize=self.binarize)
        if self.ocr_system is None:
            self.ocr_system = OCR()
        if self.crf_predictor is None:
            self.crf_predictor = Predict(self.model_path)

    def _ensure_index_locator(self):
        if self.index_locator is None:
            self.index_locator = IndexLocator()
        return self.index_locator

    def unload_models(self):
        """Drops every loaded model and frees the GPU memory they held. Called between
        queue entries so one folder's model footprint can't carry over — and compound
        into an out-of-memory error — while the next folder's models are loading."""
        self.text_extractor = None
        self.ocr_system = None
        self.crf_predictor = None
        self.index_locator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, folder_path=None, index_start_page=None, index_end_page=None, output_dir=None,
            dry=False, front_index=False, search_window=100, anchors=None, modern_format=False,
            test_image=None):
        if test_image:
            test_image = Path(test_image)
            if not test_image.exists():
                print(f"Test-afbeelding bestaat niet: {test_image}")
                return
            folder = test_image.parent
        else:
            if folder_path is None:
                print("Geen --folder of --test-image opgegeven.")
                return
            folder = Path(folder_path)

        year = _extract_year(folder)
        use_grouped = modern_format and year is not None and year >= _GROUPED_FORMAT_MIN_YEAR

        derived_output = _derive_output_dir(folder) if output_dir is None else None
        if derived_output and test_image:
            derived_output = derived_output.parent / f"{derived_output.name}_test"
        if derived_output:
            output_dir = derived_output

        base_dir = Path(output_dir) if output_dir else _OUTPUT_BASE_DIR
        output_suffix = base_dir.name
        output_dir = _next_run_dir(base_dir, output_suffix)
        output_dir.mkdir(parents=True, exist_ok=True)
        _setup_log(output_dir)
        page_log = _PageLogHandler()
        _log.addHandler(page_log)
        if derived_output:
            _log.info(f"Geen --output opgegeven, gebruik afgeleide output map: {derived_output}")
        if use_grouped:
            _log.info(f"Jaar {year} >= {_GROUPED_FORMAT_MIN_YEAR}: gegroepeerde 1912+ lay-out wordt gebruikt "
                      f"(header/act-koppeling in plaats van per-regel herordening).")

        if test_image:
            _log.info(f"[TEST] Alleen {test_image.name} wordt verwerkt.")
            selected = [test_image]
        else:
            if dry:
                locator = self._ensure_index_locator()
                result = locator.locate(folder, anchors=anchors, search_window=search_window, from_front=front_index)
                if result:
                    index_start_page, index_end_page, kind = result
                    _log.info(f"[DRY] {_INDEX_KIND_LABELS.get(kind, kind)} gedetecteerd: pagina {index_start_page} t/m {index_end_page}")
                else:
                    _log.warning(f"[DRY] Geen index gevonden in het doorzochte venster "
                                  f"({'begin' if front_index else 'einde'} van de map, {search_window} pagina's).")
                return

            if index_start_page is None and index_end_page is None:
                locator = self._ensure_index_locator()
                result = locator.locate(folder, anchors=anchors, search_window=search_window, from_front=front_index)
                if result:
                    index_start_page, index_end_page, kind = result
                    _log.info(f"{_INDEX_KIND_LABELS.get(kind, kind)} automatisch gedetecteerd: pagina {index_start_page} t/m {index_end_page}")
                else:
                    _log.warning("Kon index niet automatisch detecteren — volledige map wordt verwerkt. "
                                  "Gebruik --start-page/--end-page of --front-index om dit te sturen.")

            selected = []
            for image_path in sorted(folder.rglob("*.tif")):
                match = re.search(r'_(\d+)\.tif$', image_path.name)
                page_num = int(match.group(1)) if match else None
                if index_start_page is not None and (page_num is None or page_num < index_start_page):
                    continue
                if index_end_page is not None and (page_num is None or page_num > index_end_page):
                    continue
                selected.append(image_path)

        if not selected:
            _log.warning("Geen afbeeldingen gevonden voor het opgegeven paginabereik.")
            return

        self._ensure_pipeline_models()

        pages_pbar = tqdm(selected, desc="Pagina's verwerkt", unit="pagina")
        for image_path in pages_pbar:
            pages_pbar.set_postfix_str(image_path.name)
            page_log.clear()
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
                    _log.warning(f"OCR detecteerde {len(parts)} regels in één bounding box in {image_path.name} — splitsen")

                for i, part in enumerate(parts):
                    if _is_digits_only(part):
                        _log.warning(f"Regel bevat alleen cijfers, overgeslagen in {image_path.name}: '{part[:60]}'")
                        continue
                    elif _is_capitals_only(part):
                        _log.warning(f"Regel bevat alleen hoofdletters, overgeslagen in {image_path.name}: '{part[:60]}'")
                        continue
                    elif i == 0 and is_continuation and page_lines:
                        page_lines[-1] = self.postprocessor.combine(page_lines[-1], part)
                    else:
                        page_lines.append(part)

            page_lines = self.postprocessor.process(page_lines)

            seen = set()
            deduped = []
            for line in page_lines:
                if line in seen:
                    _log.warning(f"Dubbele regel verwijderd in {image_path.name}: '{line[:60]}'")
                else:
                    seen.add(line)
                    deduped.append(line)
            page_lines = deduped

            linked_entries = None
            if use_grouped:
                # 1912+: no rule-based merge — header/act linking replaces it entirely,
                # since grouping here depends on cross-line state (one header can have
                # many acts) that the simple pairwise end/start rules can't express.
                all_pieces = []
                for line in page_lines:
                    all_pieces.extend(self.crf_preprocessor.split_all_markers(line))
                linked_entries = self.crf_preprocessor.link_grouped_entries(all_pieces)
            else:
                if modern_format:
                    split_lines = []
                    for line in page_lines:
                        split_lines.extend(self.crf_preprocessor.split_trailing_marker(line))
                    page_lines = split_lines

                # Pass 2: rule-based merge — results are never deduplicated or undone
                page_lines = self.rule_merger.process(page_lines, modern=modern_format)

            if self.debug_mode == "ocr":
                continue

            predicted_texts = []
            quarters = []
            company_types = []

            if use_grouped:
                for cleaned_line, quarter, company_type in linked_entries:
                    if quarter is not None and quarter not in VALID_QUARTERS:
                        _log.warning(f"Ongeldig kwartaal '.{quarter}' in {image_path.name}: '{cleaned_line[:60]}'")
                    self.crf_predictor.predict_single_line(cleaned_line, debug=(self.debug_mode == "crf"))
                    predicted_texts.append(cleaned_line)
                    quarters.append(quarter)
                    company_types.append(company_type)
            else:
                for line in page_lines:
                    if self.crf_preprocessor.should_ignore(line):
                        continue
                    if modern_format:
                        cleaned_line, quarter, company_type = self.crf_preprocessor.reorder_modern_line(line)
                    else:
                        cleaned_line, quarter, company_type = self.crf_preprocessor.extract_quarter(line)
                    if quarter is not None and quarter not in VALID_QUARTERS:
                        _log.warning(f"Ongeldig kwartaal '.{quarter}' in {image_path.name}: '{line[:60]}'")
                    self.crf_predictor.predict_single_line(cleaned_line, debug=(self.debug_mode == "crf"))
                    predicted_texts.append(line)
                    quarters.append(quarter)
                    company_types.append(company_type)

            self._save_excel(predicted_texts, output_dir, image_path.stem + ".xlsx",
                             processing_log=page_log.messages.copy(), quarters=quarters,
                             company_types=company_types)
            self.crf_predictor.reset()

    def _save_thumbnail(self, src_path, dest_path):
        img = Image.open(src_path)
        w, h = img.size
        img = img.resize((int(w * _IMAGE_SCALE), int(h * _IMAGE_SCALE)), Image.LANCZOS)
        img.save(dest_path)

    def _save_excel(self, predicted_texts, output_dir, filename, processing_log=None, quarters=None, company_types=None):
        crf_rows = self.crf_predictor.get_output_no_punctuation()
        columns = self.crf_predictor.output.columns
        quarters = quarters or [None] * len(predicted_texts)
        company_types = company_types or [None] * len(predicted_texts)

        rows = []
        for crf_row, full_text, quarter, company_type in zip(crf_rows, predicted_texts, quarters, company_types):
            row = dict(zip(columns, crf_row))
            row["Full Text"] = full_text
            row["Quarter"] = quarter or ""
            row["CompanyType"] = company_type or ""
            rows.append(row)

        df = pd.DataFrame(rows, columns=list(columns) + ["Full Text", "Quarter", "CompanyType"])
        df = self.crf_postprocessor.process(df)

        output_path = output_dir / filename
        df.to_excel(output_path, index=False)
        self.excel_formatter.format(output_path, processing_log=processing_log)
        _log.info(f"{len(rows)} regels opgeslagen naar {output_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run IndexParser on a folder of .tif images.")
    arg_parser.add_argument("--folder", type=str, default=str(Path(__file__).parent / "testdata"), help="Path to folder with .tif images")
    arg_parser.add_argument("--start-page", type=int, default=None, help="First page to process (1-based, inclusive)")
    arg_parser.add_argument("--end-page", type=int, default=None, help="Last page to process (1-based, inclusive)")
    arg_parser.add_argument("--model", type=str, default=None, help="Path to CRF model file")
    arg_parser.add_argument("--output", type=str, default=None,
                            help="Base output folder. If omitted, auto-derived as .../Index_Output/<year> "
                                 "when --folder looks like .../<collection>/<date>/TIF/<year>, "
                                 "otherwise defaults to output/ next to this script")
    arg_parser.add_argument("--binarize", action="store_true", help="Convert line crops to black-and-white using Otsu thresholding before OCR")
    arg_parser.add_argument("--debug", choices=["bbox", "ocr", "crf"], default=None,
                            help="bbox: save bbox images only; ocr: print OCR output, skip CRF; crf: run all, show CRF in terminal")
    arg_parser.add_argument("--dry", action="store_true",
                            help="Only run the quick index-detection scan and report the guessed page range; "
                                 "skip the full OCR/CRF workflow entirely")
    arg_parser.add_argument("--front-index", action="store_true", dest="front_index",
                            help="The index section is at the front of this volume instead of the back, "
                                 "so scan the first --search-window pages instead of the last")
    arg_parser.add_argument("--search-window", type=int, default=130, dest="search_window",
                            help="Number of pages (from front or back, see --front-index) to scan when "
                                 "auto-detecting the index range (default: 130)")
    arg_parser.add_argument("--index-anchor", action="append", default=None, dest="index_anchor",
                            help="Header phrase to match when auto-detecting the index start page (repeatable). "
                                 "Default: built-in 'table du recueil special des actes et documents relatifs "
                                 "aux societes' and 'table methodique/alfabetique des matieres' variants")
    arg_parser.add_argument("--queue-file", type=str, default=None, dest="queue_file",
                            help="Path to a text file listing one --folder path per line (blank lines and "
                                 "'#' comments are ignored). Each is processed in turn with auto-detected "
                                 "pages/output, and its line is removed from the file once done. The file is "
                                 "re-read before every folder, so you can append new paths to it while this is "
                                 "running and they'll be picked up next. Stops as soon as the file is empty — "
                                 "it does not wait around for more. Overrides --folder.")
    arg_parser.add_argument("--modern-format", action="store_true", dest="modern_format",
                            help="Use the 1903-onward entry format, where the record id + quarter marker "
                                 "trails the entry (e.g. '...— Stichting. — S. an., 4767.4.') instead of "
                                 "leading it. Changes both the line-merge rules (entries close on the id.quarter "
                                 "marker instead of on periods) and how the id/quarter are split out before CRF. "
                                 "If the year found in --folder (or --test-image's path) is 1912 or later, the "
                                 "1912+ layout variant is used automatically instead (no dashes, company name "
                                 "stated once per group of acts instead of repeated per act) — no extra flag needed.")
    arg_parser.add_argument("--test-image", type=str, default=None, dest="test_image",
                            help="Path to a single .tif image. Runs the full pipeline on just that one page "
                                 "for quick validation (e.g. of --modern-format changes) instead of scanning "
                                 "a whole folder. Output goes to a '<derived-output>_test' folder so it's kept "
                                 "separate from real runs. Overrides --folder/--start-page/--end-page/--dry.")
    args = arg_parser.parse_args()

    index_parser = IndexParser(model_path=args.model, debug_mode=args.debug, binarize=args.binarize)

    if args.queue_file:
        _process_queue_file(index_parser, args.queue_file,
                             index_start_page=args.start_page, index_end_page=args.end_page, output_dir=args.output,
                             dry=args.dry, front_index=args.front_index, search_window=args.search_window,
                             anchors=args.index_anchor, modern_format=args.modern_format)
    else:
        index_parser.run(args.folder, index_start_page=args.start_page, index_end_page=args.end_page, output_dir=args.output,
                          dry=args.dry, front_index=args.front_index, search_window=args.search_window,
                          anchors=args.index_anchor, modern_format=args.modern_format, test_image=args.test_image)

    # uv run src/index_parser/workflow2.py --folder /mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O/20251127/TIF/1887 --start-page 1707 --end-page 1730
    # --output is now auto-derived as /mnt/UGent_Share/ghentcdh_belhisfirm/Index_Output/1887 unless overridden
    #
    # Or let it find the index itself:
    # uv run src/index_parser/workflow2.py --folder /mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O/20251127/TIF/1887 --dry
    #
    # Or queue up multiple folders (append more paths to queue.txt while this runs):
    # uv run src/index_parser/workflow2.py --queue-file queue.txt
    # find ./ -maxdepth 3 -name "1887"