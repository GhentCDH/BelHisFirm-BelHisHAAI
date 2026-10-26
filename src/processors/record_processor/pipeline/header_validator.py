import re

from src.processors.record_processor.data import MappedPrediction

class HeaderValidator:
    @staticmethod
    def is_valid_section_header(text: str) -> bool:
        """
        Checks if text matches a section header pattern (e.g. "1. - ...") and contains a comma.

        Args: text (str): Input text.

        Returns: True if valid section header, else False.
        """

        cleaned = text.strip().replace("\n", " ")

        section_header_pattern = re.compile(
            r"(?:^|\n)"  # Start of string or a new line
            r"[^\w]*"  # Optional leading noise (dots, commas, symbols)
            r"(\d{3,6})"  # The ID number (3 to 6 digits)
            r"[\s\.,]*"  # Optional separator after number
            r"[—–\-−]"  # The mandatory dash
            , re.MULTILINE
        )

        return bool(section_header_pattern.match(cleaned))

    @staticmethod
    def is_record_header_candidate(prediction: MappedPrediction, candidate_labels: set[str]) -> bool:
        """ Checks if a prediction can be considered a record header based on its label.

        Args: prediction (MappedPrediction): Model prediction.
        Args: candidate_labels (set[str]): Labels that qualify as header candidates.

        Returns: True if candidate header, else False.
        """

        return prediction.label in candidate_labels