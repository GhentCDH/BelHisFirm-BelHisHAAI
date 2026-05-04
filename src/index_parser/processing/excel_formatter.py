import re
from collections import Counter
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

_RED    = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
_YELLOW = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
_GREEN  = PatternFill(start_color="99FF99", end_color="99FF99", fill_type="solid")


class ExcelFormatter:
    """
    Formatting rules applied to the saved Excel file.

    - format(path) : open the file, apply rules, save in place
    """

    def format(self, path):
        wb = load_workbook(path)
        ws = wb.active
        self._apply_record_id_rules(ws)
        wb.save(path)

    # --- row rules ---

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
            elif counts[value] > 1:
                fill = _GREEN
            elif len(numbers) == 2:
                fill = _RED
            elif len(numbers) == 0:
                fill = _YELLOW
            else:
                continue

            for cell in row:
                cell.fill = fill
