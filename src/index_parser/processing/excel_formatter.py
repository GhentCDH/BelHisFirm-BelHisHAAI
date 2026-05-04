import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

_RED    = PatternFill(start_color="FF746C", end_color="FF746C", fill_type="solid")
_YELLOW = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
_GREEN  = PatternFill(start_color="99FF99", end_color="99FF99", fill_type="solid")
_ORANGE = PatternFill(start_color="FFD099", end_color="FFD099", fill_type="solid")

_NAME_SIMILARITY_THRESHOLD = 0.9

_LEGEND = """\
LEGEND
------
RED    = No RecordID
YELLOW = Possible duplication based on RecordID
ORANGE = Something unexpected in RecordID
GREEN  = Possible duplication based on name

Note: if a row is flagged by both name similarity and a RecordID rule,
      the RecordID color takes precedence in the Excel file.

"""


class ExcelFormatter:
    """
    Formatting rules applied to the saved Excel file.

    - format(path, explain) : open the file, apply rules, save in place.
      If explain=True and any rows were colored, a <stem>_explain.txt file
      is written next to the Excel file with the exact reason per colored row.
    """

    def format(self, path, explain=False):
        wb = load_workbook(path)
        ws = wb.active
        log = defaultdict(lambda: {'color': None, 'reasons': []})
        self._apply_name_similarity_rules(ws, log)
        self._apply_record_id_rules(ws, log)
        wb.save(path)
        if explain and log:
            self._write_explain_log(path, log)

    # --- row rules ---

    def _apply_name_similarity_rules(self, ws, log):
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        if "Name" not in header:
            return

        col = header.index("Name")
        rows = list(ws.iter_rows(min_row=2))
        names = [str(row[col].value or "") for row in rows]

        similar = defaultdict(list)
        for i, name_a in enumerate(names):
            if not name_a:
                continue
            for j, name_b in enumerate(names):
                if i >= j or not name_b:
                    continue
                ratio = SequenceMatcher(None, name_a, name_b).ratio()
                if ratio >= _NAME_SIMILARITY_THRESHOLD:
                    similar[i].append((j, ratio))
                    similar[j].append((i, ratio))

        for i, matches in similar.items():
            for cell in rows[i]:
                cell.fill = _GREEN
            row_num = i + 2
            match_strs = [f"row {j + 2} \"{names[j]}\" ({ratio:.0%})" for j, ratio in matches]
            log[row_num]['color'] = 'GREEN'
            log[row_num]['reasons'].append(
                f"Name \"{names[i]}\" is similar to: {', '.join(match_strs)}"
            )

    def _apply_record_id_rules(self, ws, log):
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        if "RecordID" not in header:
            return

        col = header.index("RecordID")
        values = [str(row[col].value or "") for row in ws.iter_rows(min_row=2)]
        counts = Counter(v for v in values if v)

        for row_idx, row in enumerate(ws.iter_rows(min_row=2)):
            value = str(row[col].value or "")
            numbers = re.findall(r'\d+', value)

            if not value:
                fill, color = _RED, 'RED'
                reason = "RecordID is empty"
            elif len(numbers) == 2:
                fill, color = _RED, 'RED'
                reason = f"RecordID \"{value}\" contains two number groups (possible merged entry)"
            elif counts[value] > 1:
                fill, color = _YELLOW, 'YELLOW'
                reason = f"RecordID \"{value}\" appears {counts[value]} times on this page"
            elif len(numbers) == 0:
                fill, color = _ORANGE, 'ORANGE'
                reason = f"RecordID \"{value}\" contains no numbers"
            else:
                continue

            for cell in row:
                cell.fill = fill
            row_num = row_idx + 2
            log[row_num]['color'] = color  # overwrites GREEN when a RecordID rule fires
            log[row_num]['reasons'].append(reason)

    # --- explain log ---

    def _write_explain_log(self, excel_path, log):
        log_path = Path(excel_path).with_name(Path(excel_path).stem + "_explain.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(_LEGEND)
            f.write("COLORED ROWS\n")
            f.write("------------\n")
            for row_num in sorted(log):
                entry = log[row_num]
                color = entry['color']
                for reason in entry['reasons']:
                    f.write(f"Row {row_num:4d}  [{color:6}]  {reason}\n")
