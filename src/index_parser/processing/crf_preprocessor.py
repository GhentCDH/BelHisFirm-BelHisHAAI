from difflib import SequenceMatcher

_MONITEUR_TEMPLATE = "ANNEXE AU MONITEUR BELGE"
_MONITEUR_THRESHOLD = 0.75


class CRFPreProcessor:
    """
    Rules applied to lines before they are sent to the CRF model.

    - should_ignore(line) : return True to skip a line entirely
    """

    def clean_before_merge(self, line: str) -> str:
        """Remove known noise phrases from the OCR string before it is merged or added to page lines."""
        return self._remove_moniteur_belge(line)

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

    # --- cleaning rules ---

    def _remove_moniteur_belge(self, line: str) -> str:
        tlen = len(_MONITEUR_TEMPLATE)
        upper = line.upper()
        if len(upper) < tlen:
            return line

        best_ratio, best_start = 0.0, -1
        for i in range(len(upper) - tlen + 1):
            ratio = SequenceMatcher(None, upper[i:i + tlen], _MONITEUR_TEMPLATE).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio >= _MONITEUR_THRESHOLD:
            return (line[:best_start] + line[best_start + tlen:]).strip()

        return line
