import re
from difflib import SequenceMatcher

_MONITEUR_TEMPLATE = "ANNEXE AU MONITEUR BELGE"
_MONITEUR_THRESHOLD = 0.75

# From 1897 onward, the record id is followed by ".<quarter>" (e.g. "2412.3" = record
# 2412, 3rd quarter). Quarter should always be 1-4, but any single digit is split off
# here so a bad OCR read (e.g. ".5") still doesn't reach the CRF model — it's flagged
# as invalid downstream (see VALID_QUARTERS) instead of silently confusing the model,
# which was trained on the plain "<id>." shape and would have to be retrained to
# handle the suffix at all.
_QUARTER_PREFIX = re.compile(r'^(\d+)\.(\d)(?!\d)')
VALID_QUARTERS = {"1", "2", "3", "4"}

# 1903 onward: the record id + quarter marker moves to the END of the entry, often
# preceded by a company-type abbreviation ("S. an.", "S. c.", "S. coop.", "S. c. a.").
# _MODERN_ID_WITH_TYPE anchors on the literal "S." rather than "the nearest preceding
# em-dash" — the em-dash before the type is sometimes dropped by OCR (e.g. "1903.
# S. an., 5542.4." instead of "1903. — S. an., 5542.4."), and anchoring on a dash
# that isn't there makes the match jump back to an unrelated, earlier dash instead,
# swallowing most of the sentence as the "type". _MODERN_ID_PLAIN is the fallback for
# entries with no type marker at all, just a trailing ", <id>.<quarter>.".
_MODERN_ID_WITH_TYPE = re.compile(r'(S\.\s*[^,]+?)\s*,\s*(\d+)\.(\d)\.?\s*$')
_MODERN_ID_PLAIN = re.compile(r'(?:[—,]\s*)?(\d+)\.(\d)\.?\s*$')
_TRAILING_DASH = re.compile(r'—\s*$')

# Layout detection sometimes merges two physical entries' bounding boxes into one OCR
# line, so an id.quarter marker ends up mid-line with the next entry's name trailing
# right after it (e.g. "au 30 juin 1902. — S. an., 737.1. Industrie ..."). Detected by
# a capitalized word immediately following the marker — the start of a new company name.
_MIDLINE_ID_QUARTER = re.compile(r'(\d+\.\d\.?)\s+([A-ZÀ-Ý].*)$')

# 1912 onward: no em-dashes at all, and the company name/address/type is stated ONCE
# as a "header" line, followed by one or more "act" lines (event + id.quarter, comma-
# separated, no name repeat, no "Id." marker) that all belong to that header until the
# next header line appears. _ACT_LINE recognizes an act line by its trailing id.quarter;
# anything else is (part of) a header. _TYPE_MARKER finds the "S. type" abbreviation
# wherever it falls — on the header line normally, but also mid-act-text when layout
# detection has glued a new header+act onto what would otherwise look like an act's
# leftover text (see link_grouped_entries).
_ACT_LINE = re.compile(r'^(.*),\s*(\d+)\.(\d)\.?\s*$')
_TYPE_MARKER = re.compile(r'\bS\.\s*[a-zà-ÿ.]+\.?')


class CRFPreProcessor:
    """
    Rules applied to lines before they are sent to the CRF model.

    - should_ignore(line) : return True to skip a line entirely
    - split_trailing_marker(line) : split a line where an id.quarter marker has the
      next entry's name glued on after it (1903-onward format)
    - extract_quarter(line) : split off the ".<quarter>" record-id suffix, if present
      (1897-1902 format: id.quarter leads the entry)
    - reorder_modern_line(line) : move a trailing id.quarter marker to the front
      (1903-onward format: id.quarter trails the entry)
    - split_all_markers(line) : like split_trailing_marker but repeated, for lines
      where layout detection glued many entries together (1912-onward format)
    - link_grouped_entries(lines) : link header/act lines to reconstruct full
      entries (1912-onward format)
    """

    def split_trailing_marker(self, line: str):
        """Splits a line at an id.quarter marker that has more text (a new entry's
        name) trailing after it on the same line. Returns a list of 1 or 2 lines —
        [line] unchanged if there's nothing trailing the marker."""
        match = _MIDLINE_ID_QUARTER.search(line)
        if not match:
            return [line]
        return [line[:match.end(1)], match.group(2)]

    def clean_before_merge(self, line: str) -> str:
        """Remove known noise phrases from the OCR string before it is merged or added to page lines."""
        return self._remove_moniteur_belge(line)

    def extract_quarter(self, line: str):
        """Splits a "<id>.<digit>" prefix (e.g. "2412.3") into (cleaned_line, quarter,
        company_type). cleaned_line has the ".<digit>" removed — "2412.3. Tramways..."
        becomes "2412. Tramways...", the same shape the CRF model was trained on.
        quarter is split off even when it isn't a valid 1-4 quarter (check against
        VALID_QUARTERS) so a bad value still doesn't reach the CRF model. company_type
        is always None here — this format (1897-1902) has no type marker. Returns
        (line, None, None) unchanged if the line doesn't start with this pattern."""
        match = _QUARTER_PREFIX.match(line)
        if not match:
            return line, None, None
        record_id, quarter = match.group(1), match.group(2)
        cleaned = record_id + line[match.end():]
        return cleaned, quarter, None

    def reorder_modern_line(self, line: str):
        """Moves a trailing "<id>.<quarter>" marker (1903-onward format) to the
        front, restoring the "<id>. <name>..." shape the CRF model was trained on.
        A company-type abbreviation before the marker (e.g. "S. an.") is extracted
        entirely rather than fed to the model at all — it's returned separately for
        its own output column, since re-inserting it into the sentence is fragile
        (company names with their own internal commas make the true insertion point
        ambiguous). "Antwerpsche Overdekte Markt, te Antwerpen. — Stichting. —
        S. an., 4767.4." becomes ("4767. Antwerpsche Overdekte Markt, te Antwerpen.
        — Stichting.", "4", "S. an."). Returns (line, None, None) unchanged if the
        line has no trailing id.quarter marker at all."""
        match = _MODERN_ID_WITH_TYPE.search(line)
        if match:
            company_type = match.group(1).strip()
            record_id, quarter = match.group(2), match.group(3)
            remainder = _TRAILING_DASH.sub('', line[:match.start()].rstrip()).rstrip()
            return f"{record_id}. {remainder}", quarter, company_type

        match = _MODERN_ID_PLAIN.search(line)
        if match:
            record_id, quarter = match.group(1), match.group(2)
            remainder = line[:match.start()].rstrip()
            return (f"{record_id}. {remainder}" if remainder else f"{record_id}."), quarter, None

        return line, None, None

    def split_all_markers(self, line: str):
        """Like split_trailing_marker, but keeps splitting at every subsequent
        id.quarter + capitalized-word boundary — 1912-onward pages sometimes have
        many entries glued onto a single OCR line with no line breaks at all.
        Returns a list of 1+ fragments, each still needing header/act classification
        (see link_grouped_entries)."""
        pieces = []
        remaining = line
        while True:
            match = _MIDLINE_ID_QUARTER.search(remaining)
            if not match:
                pieces.append(remaining)
                break
            pieces.append(remaining[:match.end(1)])
            remaining = match.group(2)
        return pieces

    def link_grouped_entries(self, lines):
        """1912-onward format: a header line (name, address, optional "S. type.")
        appears once, followed by one or more act lines ("event, id.quarter.", no
        dash, no name repeat, no "Id." marker) belonging to whichever header
        preceded them. Returns a list of (reconstructed_line, quarter,
        company_type) tuples, one per act, in the "<id>. <name>, <address>. —
        <event>." shape the CRF model was trained on.

        Lines are expected to already be run through split_all_markers — layout
        detection sometimes glues a new header+act onto what looks like trailing
        event text (e.g. "Halevy Brothers, à Anvers. S. c. Constitution, 6707.4."
        instead of separate "Halevy Brothers, à Anvers. S. c." / "Constitution,
        6707.4." lines). An "S. type" marker found within what would otherwise be
        treated as event text is recognized as exactly this case: everything
        before it becomes a new header, everything after is the event."""
        results = []
        header_parts = []
        header_type = None
        expect_new_header = True

        for line in lines:
            match = _ACT_LINE.match(line)
            if not match:
                if expect_new_header:
                    header_parts = [line]
                    header_type = None
                    expect_new_header = False
                else:
                    header_parts.append(line)
                continue

            pre_marker, record_id, quarter = match.groups()
            pre_marker = pre_marker.strip()

            type_match = _TYPE_MARKER.search(pre_marker)
            if type_match:
                before = pre_marker[:type_match.start()].strip()
                event_text = pre_marker[type_match.end():].strip()
                embedded_type = type_match.group().strip()
                if before:
                    # a new header (+ its type) was glued onto this act's text
                    header_parts = [before]
                    header_type = embedded_type
                else:
                    header_type = embedded_type
            else:
                event_text = pre_marker

            header_text = " ".join(header_parts).strip()
            trailing_type = _TYPE_MARKER.search(header_text)
            if trailing_type:
                if header_type is None:
                    header_type = trailing_type.group().strip()
                header_text = header_text[:trailing_type.start()].strip()

            reconstructed = f"{record_id}. {header_text} — {event_text}."
            results.append((reconstructed, quarter, header_type))
            expect_new_header = True

        return results

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
