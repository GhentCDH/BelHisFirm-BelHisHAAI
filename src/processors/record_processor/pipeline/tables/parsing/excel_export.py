"""Flatten each transcribed table JSON into its own Excel workbook and one
flat CSV, written right next to it.

For each structured table JSON in a record's "tables" folder, writes a matching
"<name>.xlsx" and "<name>.csv" in the same folder - so each page folder ends
up holding its JSON, its table crop(s), its full-page preview image, its
workbook, and now one flat CSV per table, all together.

The .xlsx has a sheet per data shape (deliberately not one flat sheet there,
since a single table's JSON can still have several independent one-to-many
relationships - an entry can have several share types, several
representatives, several votes - that don't all fit one row without either
losing data or repeating it awkwardly):

  - "Info": the table's own title/footnotes and whether it's a shareholder
    register (one row).
  - "Aandeelhouders": one row per shareholder entry - multi-valued fields
    (representatives, share types, votes) are joined into readable summary
    text here; see "Aandelen" for those broken out one-per-row instead.
    Also adds "totaal_aandelen_berekend", the SUM of the entry's own share
    counts, computed here for convenience - not something the model itself
    is asked to compute (see prompt.py's "totalen" guidance for why).
  - "Aandelen": one row per share-type item (exploded from each entry's
    "type" list).
  - "Totalen": one row per table-level grand-total item ("totalen").
  ("Aandeelhouders"/"Aandelen"/"Totalen" only apply to a shareholder
  register; a non-shareholder table gets "Tabel" instead:)
  - "Tabel": one row per row of the table ("generic_table.rows").

The .csv is genuinely flat - one file, no separate sheets/pages - using the
same row-per-entry (or row-per-generic-table-row) shape as "Aandeelhouders"/
"Tabel" above, with the "Info" fields (title/footnotes/register flag) and a
"totalen_samenvatting" summary repeated as extra columns on every row
instead of living on their own page. The per-share-type breakdown ("Aandelen")
isn't duplicated as extra rows here - "aandelen_samenvatting" already carries
that same information as text, and exploding it would mean two different row
counts (entries vs. share-type items) fighting for the same flat file.

Ported from BelHisFirm---HisTableFinder's parsing/frans_data.py, without its CLI
(TablePipeline walks the record folders here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .numerical_rules import _parse_aantal

INFO_COLUMNS = ["titel", "is_aandeelhoudersregister", "voetnoten"]
AANDEELHOUDERS_COLUMNS = [
    "volgnummer", "type", "naam", "gender", "adres", "beroep", "vertegenwoordiger",
    "aandelen_samenvatting", "totaal_aandelen_berekend", "stemmen_samenvatting",
]
AANDELEN_COLUMNS = ["volgnummer", "naam", "type", "aantal", "aantal_numeriek", "aantal_berekend", "waarde"]
TOTALEN_COLUMNS = ["type", "aantal", "aantal_numeriek", "waarde"]
TABEL_COLUMNS = ["rij_type", "sectie", "label", "waarde_1", "waarde_2", "waarden", "kolommen"]
FLAT_AANDEELHOUDERS_COLUMNS = INFO_COLUMNS + AANDEELHOUDERS_COLUMNS + ["totalen_samenvatting"]
FLAT_TABEL_COLUMNS = INFO_COLUMNS + TABEL_COLUMNS


def _join(items: list[str]) -> str:
    return "; ".join(item for item in items if item)


def _format_adres(adres: Any) -> str:
    if isinstance(adres, dict):
        return ", ".join(str(v) for v in adres.values() if v)
    return adres or ""


def _format_vertegenwoordiger(items: list[dict[str, Any]]) -> str:
    parts = []
    for v in items:
        naam = v.get("vertegenwoordiger", "")
        detail = " - ".join(x for x in (_join(v.get("beroep", [])), _format_adres(v.get("adres"))) if x)
        parts.append(f"{naam} ({detail})" if detail else naam)
    return _join(parts)


def _format_types(items: list[dict[str, Any]]) -> str:
    return _join(f"{t.get('type') or '?'}: {t.get('aantal_numeriek', '')}" for t in items)


def _entry_rows(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(aandeelhouders_rows, aandelen_rows) - one shareholder row per entry, plus one exploded row per share-type item."""
    aandeelhouders: list[dict[str, Any]] = []
    aandelen: list[dict[str, Any]] = []

    for i, entry in enumerate(entries, start=1):
        aandeelhouder = entry.get("aandeelhouder") or {}
        naam = aandeelhouder.get("aandeelhouder", "")
        types = entry.get("type", [])
        # Derived from the WRITTEN "aantal" text (e.g. "dix-neuf cents"), not
        # "aantal_numeriek" - that field is a literal, uncorrected
        # transcription (schema.py) that can use "," as either a thousands
        # separator or a decimal point depending on the source, so it can't
        # be reliably reinterpreted as a plain float; the written-out form
        # parses unambiguously instead.
        berekende_waarden = [v for v in (_parse_aantal(t.get("aantal", "")) for t in types) if v is not None]

        aandeelhouders.append({
            "volgnummer": i,
            "type": aandeelhouder.get("type", ""),
            "naam": naam,
            "gender": aandeelhouder.get("gender") or "",
            "adres": _format_adres(entry.get("adres")),
            "beroep": _join(entry.get("beroep", [])),
            "vertegenwoordiger": _format_vertegenwoordiger(entry.get("vertegenwoordiger", [])),
            "aandelen_samenvatting": _format_types(types),
            "totaal_aandelen_berekend": sum(berekende_waarden) if berekende_waarden else None,
            "stemmen_samenvatting": _format_types(entry.get("stemmen", [])),
        })

        for t in types:
            aandelen.append({
                "volgnummer": i,
                "naam": naam,
                "type": t.get("type", ""),
                "aantal": t.get("aantal", ""),
                "aantal_numeriek": t.get("aantal_numeriek", ""),
                "aantal_berekend": _parse_aantal(t.get("aantal", "")),
                "waarde": t.get("waarde", ""),
            })

    return aandeelhouders, aandelen


def _totalen_rows(totalen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": t.get("type", ""),
            "aantal": t.get("aantal", ""),
            "aantal_numeriek": t.get("aantal_numeriek", ""),
            "waarde": t.get("waarde", ""),
        }
        for t in totalen
    ]


def _tabel_rows(gt: dict[str, Any]) -> list[dict[str, Any]]:
    kolommen = _join(gt.get("columns", []))
    rows = []
    for row in gt.get("rows", []):
        waarden = row.get("values", [])
        rows.append({
            "rij_type": row.get("row_type", ""),
            "sectie": row.get("section", ""),
            "label": row.get("label", ""),
            "waarde_1": waarden[0] if len(waarden) > 0 else "",
            "waarde_2": waarden[1] if len(waarden) > 1 else "",
            "waarden": _join(waarden),
            "kolommen": kolommen,
        })
    return rows


def _info(table: dict[str, Any]) -> tuple[bool, str, str]:
    """(is_shareholder_register, titel, voetnoten)."""
    sr = table.get("shareholder_register") or {}
    gt = table.get("generic_table") or {}
    return (
        bool(table.get("is_shareholder_register")),
        sr.get("title") or gt.get("title") or "",
        sr.get("footnotes") or gt.get("footnotes") or "",
    )


def build_sheets(table: dict[str, Any]) -> dict[str, tuple[list[str], list[dict[str, Any]]]]:
    """Build this one table's sheets (for the .xlsx) as {name: (columns, rows)}."""
    is_sh, titel, voetnoten = _info(table)
    sr = table.get("shareholder_register") or {}
    gt = table.get("generic_table") or {}

    sheets: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "Info": (INFO_COLUMNS, [{"titel": titel, "is_aandeelhoudersregister": is_sh, "voetnoten": voetnoten}]),
    }

    if is_sh:
        aandeelhouders, aandelen = _entry_rows(sr.get("entries", []))
        sheets["Aandeelhouders"] = (AANDEELHOUDERS_COLUMNS, aandeelhouders)
        sheets["Aandelen"] = (AANDELEN_COLUMNS, aandelen)
        sheets["Totalen"] = (TOTALEN_COLUMNS, _totalen_rows(sr.get("totalen", [])))
    else:
        sheets["Tabel"] = (TABEL_COLUMNS, _tabel_rows(gt))

    return sheets


def build_flat_rows(table: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Build this one table's single flat row set (for the .csv) as (columns, rows) -
    the "Info" fields repeated onto every row instead of living on their own page/file."""
    is_sh, titel, voetnoten = _info(table)
    sr = table.get("shareholder_register") or {}
    gt = table.get("generic_table") or {}
    info = {"titel": titel, "is_aandeelhoudersregister": is_sh, "voetnoten": voetnoten}

    if is_sh:
        aandeelhouders, _ = _entry_rows(sr.get("entries", []))
        totalen_samenvatting = _format_types(sr.get("totalen", []))
        rows = [{**info, **row, "totalen_samenvatting": totalen_samenvatting} for row in aandeelhouders]
        return FLAT_AANDEELHOUDERS_COLUMNS, rows

    rows = [{**info, **row} for row in _tabel_rows(gt)]
    return FLAT_TABEL_COLUMNS, rows


def convert_table(table: dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    """Convert one structured table into a matching .xlsx (one sheet per data
    shape) and .csv (one flat file, no separate pages) next to json_path.
    Returns (xlsx_path, csv_path)."""
    if "is_shareholder_register" not in table:
        raise ValueError("not a transcribed table JSON (missing 'is_shareholder_register')")

    excel_path = json_path.with_suffix(".xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, (columns, rows) in build_sheets(table).items():
            pd.DataFrame(rows, columns=columns).to_excel(writer, sheet_name=name, index=False)

    csv_path = json_path.with_suffix(".csv")
    columns, rows = build_flat_rows(table)
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)

    return excel_path, csv_path
