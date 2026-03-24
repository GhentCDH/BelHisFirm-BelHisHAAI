import io
import re

from pathlib import Path
from logging import getLogger
from reportlab.pdfgen import canvas
from PIL import Image
from PyPDF2 import PdfWriter, PdfReader

logger = getLogger(__name__)

class PDFExporter:

    @staticmethod
    def generate_pdf_from_images(folder_path: Path, ocr_data: list[dict]) -> None:
        """Convert all images in a folder to a searchable PDF with OCR text layer."""
        # Match only page_XXX.jpg with exactly 3 digits (the format we use)
        page_pattern = re.compile(r'^page_\d{3}$')
        image_files = sorted([f for f in folder_path.glob("page_*.jpg") if page_pattern.match(f.stem)])
        if not image_files:
            logger.warning(f"No images found in {folder_path} to create PDF")
            return

        pdf_path = folder_path / f"{folder_path.name}.pdf"
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

                    def reading_order_key(line):
                        # Use saved column assignment from OCR phase
                        column_name = line.get("column", "unknown")
                        # Map column names to sort order: left=0 (includes spanning/titles), right=1
                        # Spanning lines are grouped with left column to maintain column separation
                        column_order = {"single": 0, "left": 0, "spanning": 0, "right": 1, "unknown": 0}
                        column = column_order.get(column_name, 0)

                        y1 = line["bbox"][1]
                        # Sort by column first, then by y position (top to bottom)
                        return (column, y1)

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