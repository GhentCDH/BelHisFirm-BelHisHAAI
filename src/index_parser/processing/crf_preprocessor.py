class CRFPreProcessor:
    """
    Rules applied to lines before they are sent to the CRF model.

    - should_ignore(line) : return True to skip a line entirely
    """

    def should_ignore(self, line: str) -> bool:
        if self._is_header(line):
            return True
        if self._is_note(line):
            return True
        return False

    # --- ignore rules ---

    def _is_header(self, line: str) -> bool:
        return "TABLE DU RECUIL" in line or "N°" in line

    def _is_note(self, line: str) -> bool:
        return "d'ordre" in line
