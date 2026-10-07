import csv
import re
import unicodedata
import io
import json

from pathlib import Path
from PIL import Image
from logging import getLogger

from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas

from src.processors.record_processor.data import Record

logger = getLogger(__name__)

class IOManager:

    LIGATURES = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss", "ø": "o", "Ø": "O"})

    @staticmethod
    def generate_folder_name(record: Record) -> str:
        """ Generate a folder name based on record information.

        Args: record (Record): Record to be stored in the CSV file.

        Return: Folder name for given record.
        """

        # Normalize folder name - remove/replace problematic characters
        title = record.record_title
        title = title.translate(IOManager.LIGATURES)  # These have no accent to strip, so spell them out
        title = unicodedata.normalize('NFKD', title)  # Split accented letters into letter + accent (é -> e + ´)
        title = title.encode('ascii', errors='ignore').decode('ascii')  # Drop the accents and any other non-ASCII
        title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid filename chars
        title = re.sub(r'\s+', '_', title)  # Replace whitespace with underscore
        title = re.sub(r'_+', '_', title)  # Collapse multiple underscores
        title = title.strip('_')  # Remove leading/trailing underscores
        title = title[:30] if len(title) > 30 else title  # Limit length
        folder_name = f"{int(record.record_id):03d}-{title}"

        return folder_name

    @staticmethod
    def collect_image_files(images_folder_path: Path) -> list[Path]:
        """ Collect all image files inside a folder path and return a list of paths.

            Args: folder_path (Path): Folder path with image files.

            Returns: A list of paths pointing to each image file.
        """

        logger.info(f"Collecting image files from {images_folder_path}...")
        image_files = sorted(list(Path(images_folder_path).glob("*.jpg")) + list(Path(images_folder_path).glob("*.jpeg")) + list(Path(images_folder_path).glob("*.tif")) + list(Path(images_folder_path).glob("*.jp2")))
        return image_files

    CSV_FIELDNAMES = ['record_id', 'internal_record_number', 'record_title', 'folder_name', 'num_pages',
                      'start_page', 'end_page', 'start_bbox', 'end_bbox', 'num_tables', 'num_tables_failed']

    @staticmethod
    def update_records_csv(record: Record, record_folder_path: Path, output_folder_path: Path, tables_summary: list[dict]) -> None:
        """ Insert or replace this record's row in records_index.csv, keyed by folder name,
            so re-running a stage never duplicates a record's row.

            Args: record (Record): Record to be stored in the CSV file.
            Args: record_folder_path (Path): Folder path to the record.
            Args: output_folder_path (Path): Folder path to save the CSV file in.
            Args: tables_summary (list[dict]): Table entries for this record, from load_tables_index.

            Returns: None
        """

        csv_path = output_folder_path / "records_index.csv"
        folder_name = record_folder_path.name

        record_data = {
            'record_id': record.record_id,
            'internal_record_number': record.internal_record_number,
            'record_title': record.record_title,
            'folder_name': folder_name,
            'num_pages': len(record.images),
            'start_page': record.start_header_bbox_page,
            'end_page': record.end_header_bbox_page,
            'start_bbox': str(record.start_header_bbox),
            'end_bbox': str(record.end_header_bbox),
            'num_tables': len(tables_summary),
            'num_tables_failed': sum(1 for t in tables_summary if t["error"] is not None),
        }

        rows = []
        if csv_path.exists():
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                rows = [row for row in csv.DictReader(f) if row.get('folder_name') != folder_name]
        rows.append(record_data)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=IOManager.CSV_FIELDNAMES, restval='')
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"CSV updated: {csv_path}")

    @staticmethod
    def find_record_folders(output_folder_path: Path) -> list[Path]:
        """ Record folders are the ones written by save_record_meta. """
        return sorted(meta.parent for meta in Path(output_folder_path).glob("*/record_meta.json"))

    @staticmethod
    def save_record_meta(record: Record, record_folder: Path) -> None:
        """ Writes the non-image fields of a record so later stages can rebuild it from disk. """

        meta = {
            "record_id": record.record_id,
            "record_title": record.record_title,
            "internal_record_number": record.internal_record_number,
            "start_header_bbox": record.start_header_bbox,
            "start_header_bbox_meta": record.start_header_bbox_meta,
            "start_header_bbox_page": record.start_header_bbox_page,
            "end_header_bbox": record.end_header_bbox,
            "end_header_bbox_meta": record.end_header_bbox_meta,
            "end_header_bbox_page": record.end_header_bbox_page,
        }
        with open(record_folder / "record_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=int)

    @staticmethod
    def load_record(record_folder: Path) -> Record:
        """ Rebuilds a Record from a folder written by save_record_images + save_record_meta. """

        meta = json.loads((record_folder / "record_meta.json").read_text(encoding="utf-8"))
        page_pattern = re.compile(r'^page_\d{3}$')
        page_paths = sorted(f for f in record_folder.glob("page_*.jpg") if page_pattern.match(f.stem))
        images = [Image.open(p).convert('RGB') for p in page_paths]

        return Record(
            images=images,
            record_id=meta["record_id"],
            record_title=meta["record_title"],
            internal_record_number=meta["internal_record_number"],
            start_header_bbox=meta["start_header_bbox"],
            start_header_bbox_meta=meta["start_header_bbox_meta"],
            start_header_bbox_page=meta["start_header_bbox_page"],
            end_header_bbox=meta["end_header_bbox"],
            end_header_bbox_meta=meta["end_header_bbox_meta"],
            end_header_bbox_page=meta["end_header_bbox_page"],
        )

    @staticmethod
    def weights_fingerprint(weights_path: Path) -> dict:
        """ Identifies a weights file, so a layout cached with other (e.g. retrained) weights is not reused. """
        stat = Path(weights_path).stat()
        return {"name": Path(weights_path).name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    @staticmethod
    def save_layout(record_folder: Path, layout: list[list[dict]], weights_fingerprint: dict) -> None:
        """ Writes the layout regions of every page of a record, so later stages do not run layout detection again. """
        with open(record_folder / "layout.json", 'w', encoding='utf-8') as f:
            json.dump({"weights": weights_fingerprint, "pages": layout}, f, ensure_ascii=False)

    @staticmethod
    def load_layout(record_folder: Path, weights_fingerprint: dict, num_pages: int) -> list[list[dict]] | None:
        """ Layout written by save_layout, or None when there is none for these weights and pages. """

        layout_path = record_folder / "layout.json"
        if not layout_path.exists():
            return None

        try:
            data = json.loads(layout_path.read_text(encoding="utf-8"))
            pages = data["pages"]
            if data["weights"] != weights_fingerprint or len(pages) != num_pages:
                return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        return pages

    @staticmethod
    def load_tables_index(record_folder: Path) -> list[dict]:
        index_path = record_folder / "tables_index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text(encoding="utf-8"))

    @staticmethod
    def generate_pdf_from_record(record_path: Path, ocr_data: list[dict]) -> None:
        """ Convert all images in a record folder to a searchable PDF with OCR text layer.

            Args: record_path (Path): Path to the record folder.
            Args: ocr_data (list[dict]): Parsed OCR data.

            Returns: None

            This method will create a PDF in the same folder as the given record folder.
        """
        # Match only page_XXX.jpg with exactly 3 digits (the format we use)
        page_pattern = re.compile(r'^page_\d{3}$')
        image_files = sorted([f for f in record_path.glob("page_*.jpg") if page_pattern.match(f.stem)])
        if not image_files:
            logger.warning(f"No images found in {record_path} to create PDF")
            return

        pdf_path = record_path / f"{record_path.name}.pdf"
        pdf_writer = PdfWriter()

        for page_idx, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_width, img_height = img.size

                # Create image-only PDF page
                img_pdf_buffer = io.BytesIO()
                img.save(img_pdf_buffer, format='PDF')
                img_pdf_buffer.seek(0)
                img_pdf_reader = PdfReader(img_pdf_buffer)
                img_page = img_pdf_reader.pages[0]

                # Create text layer PDF
                text_pdf_buffer = io.BytesIO()
                c = canvas.Canvas(text_pdf_buffer, pagesize=(img_width, img_height))

                # Add invisible text at each block's position, in reading order
                if page_idx < len(ocr_data):
                    page_blocks = ocr_data[page_idx].get("blocks", [])
                    column_order = {"single": 0, "left": 0, "spanning": 0, "right": 1}
                    sorted_blocks = sorted(page_blocks, key=lambda b: (column_order.get(b.get("column"), 0), b["bbox"][1]))

                    for block in sorted_blocks:
                        x1, y1, x2, y2 = block["bbox"]
                        block_lines = [line.strip() for line in block["text"].splitlines() if line.strip()]
                        if not block_lines:
                            continue

                        line_height = (y2 - y1) / len(block_lines)
                        for line_idx, line_text in enumerate(block_lines):
                            try:
                                # Sanitize text - keep only latin-1 characters
                                text = line_text.encode('latin-1', errors='ignore').decode('latin-1')
                                if not text.strip():
                                    continue

                                # Image coordinates have Y=0 at the top, PDF coordinates have Y=0 at the bottom.
                                # Each line gets an equal slice of the block, with the slice's bottom as the baseline.
                                baseline_y = y1 + (line_idx + 1) * line_height
                                pdf_y = img_height - baseline_y

                                font_size = max(6, min(line_height * 0.85, 72))
                                c.setFont("Helvetica", font_size)

                                text_width = c.stringWidth(text, "Helvetica", font_size)
                                h_scale = (x2 - x1) / text_width if text_width > 0 else 1

                                c.saveState()
                                c.setFillAlpha(0)  # Invisible text
                                c.translate(x1, pdf_y)
                                c.scale(h_scale, 1)
                                c.drawString(0, 0, text)
                                c.restoreState()
                            except Exception as text_err:
                                logger.warning(f"Failed to add text '{line_text[:50]}...' to PDF: {text_err}")

                # Without this a page with no text gets no page at all in the text layer
                c.showPage()
                c.save()
                text_pdf_buffer.seek(0)
                text_pdf_reader = PdfReader(text_pdf_buffer)
                text_page = text_pdf_reader.pages[0]

                # Merge text layer under image
                text_page.merge_page(img_page)
                pdf_writer.add_page(text_page)

            except Exception as e:
                logger.error(f"Failed to process {img_path.name} for PDF: {e}")

        with open(pdf_path, 'wb') as f:
            pdf_writer.write(f)

        # Check if OCR data was included
        has_ocr_text = any(
            page_data.get("blocks", [])
            for page_data in ocr_data if isinstance(page_data, dict)
        )
        pdf_type = "Searchable PDF" if has_ocr_text else "PDF (image-only)"

        logger.info(f"{pdf_type} created: {pdf_path}")



    @staticmethod
    def save_record_to_json(record: Record, record_folder: Path, ocr_data: list, tables_summary: list[dict]) -> None:
        ocr_json_path = record_folder / "ocr_data.json"

        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "record_id": record.record_id,
                "record_title": record.record_title,
                "pages": ocr_data,
                "tables": tables_summary,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"OCR data saved: {ocr_json_path}")

    @staticmethod
    def save_record_images(record: Record, record_folder: Path) -> None:
        """ Writes all images within a record to the record folder.

            Args: record (Record): The record to sample the images from.
            Args: record_folder (Path): The folder to save the images to.

            Returns: None
        """


        record_folder.mkdir(parents=True, exist_ok=True)

        # The cached layout belongs to the previous page images
        (record_folder / "layout.json").unlink(missing_ok=True)

        for idx, image in enumerate(record.images):
            image_filename = record_folder / f"page_{idx + 1:03d}.jpg"
            image.save(image_filename)
