import re
from collections import Counter
from difflib import SequenceMatcher
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

_RED    = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
_YELLOW = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
_GREEN  = PatternFill(start_color="99FF99", end_color="99FF99", fill_type="solid")
_ORANGE = PatternFill(start_color="FFD099", end_color="FFD099", fill_type="solid")

_NAME_SIMILARITY_THRESHOLD = 0.9


class ExcelFormatter:
    """
    Formatting rules applied to the saved Excel file.

    - format(path) : open the file, apply rules, save in place
    """

    def format(self, path):
        wb = load_workbook(path)
        ws = wb.active
        self._apply_name_similarity_rules(ws)
        self._apply_record_id_rules(ws)
        wb.save(path)

    # --- row rules ---

    def _apply_name_similarity_rules(self, ws):
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        if "Name" not in header:
            return

        col = header.index("Name")
        rows = list(ws.iter_rows(min_row=2))
        names = [str(row[col].value or "") for row in rows]

        similar = set()
        for i, name_a in enumerate(names):
            if not name_a:
                continue
            for j, name_b in enumerate(names):
                if i >= j or not name_b:
                    continue
                if SequenceMatcher(None, name_a, name_b).ratio() >= _NAME_SIMILARITY_THRESHOLD:
                    similar.add(i)
                    similar.add(j)

        for i in similar:
            for cell in rows[i]:
                cell.fill = _GREEN

    def _apply_record_id_rules(self, ws):
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        if "RecordID" not in header:
            return

        col = header.index("RecordID")

        # Count occurrences of each non-empty RecordID to detect duplicates
        values = [str(row[col].value or "") for row in ws.iter_rows(min_row=2)]
        counts = Counter(v for v in values if v)

        for row in ws.iter_rows(min_row=2):
            value = str(row[col].value or "")
            numbers = re.findall(r'\d+', value)

            if not value:
                fill = _YELLOW
            elif len(numbers) == 2:
                fill = _RED
            elif counts[value] > 1:
                fill = _YELLOW
            elif len(numbers) == 0:
                fill = _ORANGE
            else:
                continue

            for cell in row:
                cell.fill = fill
