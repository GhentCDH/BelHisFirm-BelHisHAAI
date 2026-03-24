import csv
import re

from pathlib import Path

from logging import getLogger

from recordprocessing.data import Record

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