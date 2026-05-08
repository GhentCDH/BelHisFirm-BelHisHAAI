class OCRPostProcessor:
    """
    Rules for cleaning OCR output. Two entry points:

    - combine(left, right)  : called when joining a continuation line to the previous one
    - process(lines)        : called on the full list of lines for a page
    """

    def combine(self, left: str, right: str) -> str:
        # Hyphenation: word split across lines, hyphen at end of first or start of second
        if left.endswith("-"):
            return left[:-1] + right
        if right.startswith("-"):
            return left + right[1:]
        return left + " " + right

    def process(self, lines: list) -> list:
        return [cleaned for line in lines if (cleaned := self._clean(line))]

    # --- individual line rules ---

    def _clean(self, line: str) -> str:
        line = line.strip()
        return line
