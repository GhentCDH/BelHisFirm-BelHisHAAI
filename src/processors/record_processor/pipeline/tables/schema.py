"""JSON schema a table crop's transcription is structured into.

Ported verbatim from BelHisFirm---HisTableFinder's transcribe_tables/schema.py.
Tuned for Belgian shareholder-register tables specifically, with a generic
"generic_table" fallback for anything else (financial statements, plain
grids) via the "is_shareholder_register" classification field. Validate
against real BelHisHAAI table samples before assuming this fits as-is.
"""

from __future__ import annotations

from typing import Any

ROW_TYPES = ["header", "section", "item", "subtotal", "total"]
TABLE_TYPES = ["financial_statement", "grid_table", "other"]
SHAREHOLDER_ENTRY_TYPES = ["persoon", "groep", "bedrijf", "onbekend"]

AANDEEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "The share type/class/denomination as printed. empty string if none is printed.",
        },
        "aantal": {
            "type": "string",
            "description": (
                "Number of shares of this type/class, in the full written form "
                "exactly as printed, including any fractional wording (e.g. 'cent "
                "nonante-cinq', or 'douze et quatre cinquiemes' when a fraction is "
                "stated). The fraction can be separated from its whole number by "
                "other words (e.g. 'cent nonante-cinq actions et quatre cinquiemes "
                "d'actions entierement liberees') - it still belongs to that same "
                "count; do not drop it."
            ),
        },
        "aantal_numeriek": {
            "type": "string",
            "description": (
                "The same count in numeral form, mirroring the numeral notation "
                "used in the source AS PRINTED - including any fraction shorthand "
                "the source itself uses (e.g. '12.4' if the source prints a bare "
                "decimal-look mark after the whole number for a fraction, even "
                "where the true value implied by 'cinquiemes' would be .8). Do not "
                "correct or reinterpret anything here; this is the literal digit "
                "transcription."
            ),
        },
        "waarde": {
            "type": "string",
            "description": "Value of the shares of this type/class, always as a plain number but must include the currency if printed.",
        },
    },
    "required": [
        "type",
        "aantal",
        "aantal_numeriek",
    ],
    "additionalProperties": False,
}

STEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "The vote type as printed (e.g. 'stem', 'voix'). Empty string if none is printed.",
        },
        "aantal": {
            "type": "string",
            "description": (
                "Number of votes attached to this entry, in the full written form "
                "exactly as printed, including any fractional wording."
            ),
        },
        "aantal_numeriek": {
            "type": "string",
            "description": (
                "The same count in numeral form, mirroring the numeral notation "
                "used in the source AS PRINTED. Do not correct or reinterpret "
                "anything here; this is the literal digit transcription."
            ),
        },
    },
    "required": [
        "type",
        "aantal",
        "aantal_numeriek",
    ],
    "additionalProperties": False,
}

AANDEELHOUDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": SHAREHOLDER_ENTRY_TYPES,
            "description": "'persoon' for a natural person, "
            "'groep' for an entry with multiple persons that are the shareholders, "
            "'bedrijf' for a company/legal entity, "
            "'onbekend' for a block of shares held by unspecified/unnamed parties "
            "(no individual name is printed for it) that is not itself a summary of "
            "the other entries in this table - see prompt.py 2.2 for the distinction "
            "from 'totalen'.",
        },
        "aandeelhouder": {
            "type": "string",
            "description": (
                "The full name and titles of the person, group or company, exactly "
                "as printed. For 'onbekend', the descriptive phrase identifying the "
                "block of shares, exactly as printed (e.g. the text before the share "
                "count itself)."
            ),
        },
    },
    "required": ["type", "aandeelhouder"],
    "additionalProperties": False,
}

REPRESENTATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vertegenwoordiger": {
            "type": "string",
            "description": (
                "Full name of the person, exactly as printed."
            ),
        },
        "adres": {
            "type": "string",
            "description": "The address exactly as printed, in full.",
        },
        "beroep": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General profession/occupation(s) printed for this person possibly unrelated to their role in the shareholder context.",
        },

    },
    "required": [
        "vertegenwoordiger"
    ],
    "additionalProperties": False,
}

SHAREHOLDER_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aandeelhouder": AANDEELHOUDER_SCHEMA,
        "adres": {
            "type": "string",
            "description": "The address exactly as printed, in full.",
        },
        "beroep": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General profession/occupation(s) printed for this person possibly unrelated to their role as representative.",
        },
        "vertegenwoordiger": {
            "type": "array",
            "items": REPRESENTATIVE_SCHEMA,
            "description": (
                "This entry if for the representative ('vertegenwoordiger') of the shareholder ('aandeelhouder'). This is always a person."
                "If the shareholder is a group of persons, this field is repeated for each representative of the group."
                "This can be indicated by a phrase like 'pour', representing the shareholder or 'représenté par', represented by the representative."
            ),
        },
        "type": {
            "type": "array",
            "items": AANDEEL_SCHEMA,
            "description": "Every share block held by this entry, one item per distinct share type/class printed on the row.",
        },
        "stemmen": {
            "type": "array",
            "items": STEM_SCHEMA,
            "description": "Every vote entry printed for this row, if any.",
        },
    },
    "required": [
        "aandeelhouder",
        "vertegenwoordiger",
        "type",
        "stemmen",
    ],
    "additionalProperties": False,
}




ROW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "row_type": {
            "type": "string",
            "enum": ROW_TYPES,
            "description": (
                "'header': a printed column-header row inside the table. "
                "'section': a label-only row with no values that introduces "
                "the rows below it (e.g. 'Envers la societe :') - also the "
                "right type for a secondary heading that divides one table "
                "into named parts (e.g. 'Actif'/'Passif', 'Débit'/'Crédit') - "
                "see GENERIC_TABLE_SCHEMA's 'title' for why that is NOT the "
                "table's own title. "
                "'item': a row with a label and one or more values. "
                "'subtotal': a row ruled off with a single line above its "
                "values. 'total': a final row ruled off with a double line "
                "or introduced by a word like 'Ensemble' or 'Total'."
            ),
        },
        "label": {
            "type": "string",
            "description": (
                "The row's text, transcribed exactly as printed (French/Dutch, "
                "original spelling, abbreviations, and punctuation kept as-is). "
                "Use an empty string if the row has no label text of its own."
            ),
        },
        "values": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The row's numeric/amount cells, left to right, in the same "
                "order as the table's 'columns'. Transcribe each value "
                "exactly as printed, including thousands separators, decimal "
                "or centime parts, and placeholder marks such as '»' or '-' "
                "used to mean nil/ditto. Leave empty for label-only rows."
            ),
        },
        "section": {
            "type": "string",
            "description": (
                "The exact 'label' of the 'section' row this row falls under - "
                "makes each row's context explicit instead of something a "
                "downstream reader has to re-derive from row order, which "
                "breaks once a table has more than one section or a "
                "subtotal/total row closes one section while others follow. "
                "Copy the governing section's own 'label' verbatim. Empty "
                "string for a 'header' row, for a 'section' row itself, or "
                "for any row printed before the table's first section header."
            ),
        },
    },
    "required": ["row_type", "label", "values", "section"],
    "additionalProperties": False,
}

GENERIC_TABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": ["string", "null"],
            "description": (
                "The OUTER heading that names the whole statement, printed "
                "directly above the table, if present; null if there is "
                "none (e.g. 'Bilan au 31 décembre 1914', 'Profits et "
                "pertes'). A secondary heading that introduces just PART of "
                "this same table's rows (e.g. 'Actif'/'Passif' under a "
                "'Bilan' title, or 'Débit'/'Crédit' under a 'Profits et "
                "pertes' title) is NOT this table's title - it belongs in "
                "'rows' instead, as its own 'section' row."
            ),
        },
        "table_type": {
            "type": "string",
            "enum": TABLE_TYPES,
            "description": "Best-fit category for this table's layout.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Column headers, left to right, matching the order of each "
                "row's 'values'. Use an empty array if the table has no "
                "printed column headers (common for financial statements, "
                "which usually just have one or two unlabeled amount "
                "columns)."
            ),
        },
        "rows": {
            "type": "array",
            "items": ROW_SCHEMA,
            "minItems": 1,
            "description": "Every row in the table, top to bottom, in reading order.",
        },
        "footnotes": {
            "type": ["string", "null"],
            "description": "Footnote text printed below the table in the source image, if any; null if there is none.",
        },
    },
    "required": ["title", "table_type", "columns", "rows", "footnotes"],
    "additionalProperties": False,
}

SHAREHOLDER_REGISTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": ["string", "null"],
            "description": "Caption or heading text printed directly above the table, if present in the image; null if there is none.",
        },
        "entries": {
            "type": "array",
            "items": SHAREHOLDER_ENTRY_SCHEMA,
            "minItems": 1,
            "description": (
                "Every shareholder/member entry, top to bottom, in reading order"
            ),
        },
        "totalen": {
            "type": "array",
            "items": AANDEEL_SCHEMA,
            "description": (
                "Grand/running totals for the WHOLE table, ONLY when an explicit "
                "total figure is actually printed - before all entries (e.g. 'the "
                "remaining N shares are subscribed by the following:', introducing "
                "the entries below), after all entries (e.g. below a closing rule), "
                "or both. Transcribe the printed figure(s) only - NEVER compute, "
                "sum, or derive a total yourself from the individual entries above. "
                "One item per distinct share type totalled, in the same shape as an "
                "entry's 'type' items. Empty array if no such total is printed - "
                "this will be the common case. See prompt.py 2.1/2.2 for how this "
                "differs from an 'onbekend' entry."
            ),
        },
        "footnotes": {
            "type": ["string", "null"],
            "description": (
                "Any other extra text printed around the table, if any (other "
                "than the grand totals already captured in 'totalen'); "
                "otherwise null."
            ),
        },
    },
    "required": ["title", "entries", "totalen", "footnotes"],
    "additionalProperties": False,
}

TABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_shareholder_register": {
            "type": "boolean",
            "description": (
                "true if this table is a shareholder/member register: a numbered list of "
                "individual or corporate shareholders with their address, function/profession, "
                "and the shares (and optionally dividends/votes) they hold. false for financial "
                "statements, plain grid tables, or anything else."
            ),
        },
        "generic_table": {
            "anyOf": [{"type": "null"}, GENERIC_TABLE_SCHEMA],
            "description": "Populated with the generic transcription when is_shareholder_register is false; null when it is true.",
        },
        "shareholder_register": {
            "anyOf": [{"type": "null"}, SHAREHOLDER_REGISTER_SCHEMA],
            "description": "Populated with the structured shareholder entries when is_shareholder_register is true; null when it is false.",
        },
    },
    "required": ["is_shareholder_register", "generic_table", "shareholder_register"],
    "additionalProperties": False,
}
