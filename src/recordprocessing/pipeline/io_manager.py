import csv
import re
import io
import json

from pathlib import Path
from PIL import Image
from logging import getLogger

from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas

from src.recordprocessing.data import Record

logger = getLogger(__name__)

class IOManager:

    @staticmethod
    def generate_folder_name(record: Record) -> str:
        """ Generate a folder name based on record information.

        Args: record (Record): Record to be stored in the CSV file.

        Return: Folder name for given record.
        """

        # Normalize folder name - remove/replace problematic characters
        title = record.record_title
        title = title.encode('ascii', errors='ignore').decode('ascii')  # Remove non-ASCII
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

    @staticmethod
    def update_records_csv(record: Record, record_folder_path: Path, output_folder_path: Path) -> None:
        """ Update CSV file with current record information.

            Args: record (Record): Record to be stored in the CSV file.
            Args: record_folder (Path): Folder path to the record.
            Args: output_folder_path (Path): Folder path to save the CSV file in.

            Returns: None
        """

        csv_path = output_folder_path / "records_index.csv"
        file_exists = csv_path.exists()

        # Prepare record data
        record_data = {
            'record_id': record.record_id,
            'internal_record_number': record.internal_record_number,
            'record_title': record.record_title,
            'folder_name': record_folder_path.name,
            'num_pages': len(record.images),
            'start_page': record.start_header_bbox_page,
            'end_page': record.end_header_bbox_page,
            'start_bbox': str(record.start_header_bbox),
            'end_bbox': str(record.end_header_bbox),
        }

        # Write or append to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['record_id', 'internal_record_number', 'record_title', 'folder_name', 'num_pages',
                          'start_page', 'end_page', 'start_bbox', 'end_bbox']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(record_data)

        logger.info(f"CSV updated: {csv_path}")

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

                # Add invisible text at bbox positions, sorted by reading order
                if page_idx < len(ocr_data):
                    # Sort lines by reading order: left column top-to-bottom, then right column top-to-bottom
                    page_data = ocr_data[page_idx]
                    page_lines = page_data.get("lines", [])

                    def reading_order_key(_line):
                        # Use saved column assignment from OCR phase
                        column_name = _line.get("column", "unknown")
                        # Map column names to sort order: left=0 (includes spanning/titles), right=1
                        # Spanning lines are grouped with left column to maintain column separation
                        column_order = {"single": 0, "left": 0, "spanning": 0, "right": 1, "unknown": 0}
                        column = column_order.get(column_name, 0)

                        _y1 = _line["bbox"][1]
                        # Sort by column first, then by y position (top to bottom)
                        return column, _y1

                    sorted_lines = sorted(page_lines, key=reading_order_key)

                    for line_idx, line in enumerate(sorted_lines):
                        x1, y1, x2, y2 = line["bbox"]
                        text = line["text"]
                        if text.strip():
                            try:
                                # Sanitize text - keep only ASCII and common extended chars
                                text = text.encode('latin-1', errors='ignore').decode('latin-1')
                                if not text.strip():
                                    continue

                                # Convert from image coordinates (Y=0 at top) to PDF coordinates (Y=0 at bottom)
                                # Use y2 (bottom of bbox) as the baseline for text positioning
                                pdf_y = img_height - y2
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1

                                # Scale font size to match bbox height
                                font_size = max(6, min(bbox_height * 0.85, 72))  # Clamp between 6 and 72
                                c.setFont("Helvetica", font_size)

                                # Calculate text width and scale horizontally to fit bbox
                                text_width = c.stringWidth(text, "Helvetica", font_size)
                                if text_width > 0:
                                    h_scale = bbox_width / text_width
                                else:
                                    h_scale = 1

                                c.saveState()
                                c.setFillAlpha(0)  # Invisible text
                                c.translate(x1, pdf_y)
                                c.scale(h_scale, 1)  # Scale horizontally to fit bbox
                                c.drawString(0, 0, text)
                                c.restoreState()
                            except Exception as text_err:
                                logger.warning(f"Failed to add text '{text[:50]}...' to PDF: {text_err}")

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
            page_data.get("lines", [])
            for page_data in ocr_data if isinstance(page_data, dict)
        )
        pdf_type = "Searchable PDF" if has_ocr_text else "PDF (image-only)"

        logger.info(f"{pdf_type} created: {pdf_path}")



    @staticmethod
    def save_record_to_json(record: Record, record_folder: Path, ocr_data: list) -> None:
        ocr_json_path = record_folder / "ocr_data.json"

        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "record_id": record.record_id,
                "record_title": record.record_title,
                "pages": ocr_data
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

        for idx, image in enumerate(record.images):
            image_filename = record_folder / f"page_{idx + 1:03d}.jpg"
            image.save(image_filename)
