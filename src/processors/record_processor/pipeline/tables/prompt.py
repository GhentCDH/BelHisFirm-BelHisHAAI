"""Prompt for structuring a table's OCR text into the schema.TABLE_SCHEMA shape.

Ported verbatim from BelHisFirm---HisTableFinder's transcribe_tables/prompt.py.
"""

from __future__ import annotations


SYSTEM_PROMPT = """SHAREHOLDER REGISTER EXTRACTION

STEP 1 - Classify the table: "is_shareholder_register"

A shareholder register lists individual shareholders and the shares they hold. A
shareholder entry may be a single person, a group of people named together, or a
company, and may include an address, occupation, or a representative acting on the
shareholder's behalf.

Set "is_shareholder_register" to true if the table matches this description.

Set it to false for anything else, in particular:

- Financial statements ("bilan"): indented line-item labels connected by a row of
  dots to one or two right-aligned amount columns. Section headers (e.g. "Envers la
  societe :", "Immobilise :") introduce a group of rows without a value of their own.
  Subtotal rows are ruled off with a single line above the amount; a final total row
  is ruled off with a double line, or introduced by a word like "Total" or "Ensemble".
- Plain grids with an explicit header row that are not listing shareholders -
  record these column by column instead.

If "is_shareholder_register" is false, leave "shareholder_register" null and fill
"generic_table" instead.

------------------------------------------------------------------------------

STEP 2 - If "is_shareholder_register" is true

Fill in "shareholder_register" and leave "generic_table" null.

- First decide: shareholder register or other type of table.
- Record exactly what is printed.
- Treat the shareholder and representative separately.
- Never infer or derive data or totals yourself.
- Output must strictly follow the JSON schema.

2.1 Table-level fields
-----------------------
- "title": the caption/heading printed directly above the table, if any;
  otherwise null.
- "entries": every shareholder entry, top to bottom, in reading order. Do not
  create an entry for a row that only carries a total or summary of the OTHER
  entries in this table - whether that row comes before them (e.g. "The
  remaining 785.96 shares are subscribed by the following:", introducing the
  list below) or after them (e.g. a closing "Total: ..." line) - see
  "totalen" below for that. See 2.2's "onbekend" type for the DIFFERENT case
  of a row that states its own distinct block of shares, just without a named
  holder.
- "totalen": grand/running totals for the table AS A WHOLE, ONLY when an
  explicit total figure is actually printed, wherever it appears - before all
  entries (introducing them, e.g. "Les sept cent quatre-vingt-cinq actions
  ... restant sont souscrits par :" followed by the list it introduces),
  after all entries (e.g. "Actions de capital, mille quatre-vingt-six.
  1,086" / "Parts de fondateur, trente-cinq. 35" below a closing rule), or
  both - one item per distinct share type totalled, each in the same shape
  as an entry's "type" items (2.4): "type"/"aantal"/"aantal_numeriek"/
  "waarde". Record the printed figure(s) only - NEVER add up the
  entries' own numbers yourself to compute a total that isn't itself
  printed. Empty array if no such total is printed - this will be the
  common case.
- "footnotes": any other extra text printed around the table, if any (other
  than the totals already captured in "totalen"); otherwise null.

2.2 Per-entry fields
---------------------
For each entry, fill in all of the following - the schema requires every one
of them, so use the field's own "nothing printed" value (an empty string or an
empty array, as noted per field) rather than omitting it or inventing content:

- "aandeelhouder": an object with two sub-fields:
    - "type": "persoon" for a single natural person, "groep" for an entry
      naming multiple people who together form one shareholder (e.g. "les
      epoux X", "MM. X et Y") - there need to be two names present for
      "groep" - "bedrijf" for a company/legal entity - "onbekend" for a row
      that states its own distinct block of shares (a real, separate holding,
      just without any individual named for it), e.g. "Cent nonante-cinq
      actions ... qui se les partagent suivant leurs conventions
      particulières" (a block of shares held collectively by unspecified
      parties "under their own private arrangement", not further broken down
      anywhere in this table).
      IMPORTANT - check this BEFORE picking "groep": if the entry contains a
      phrase like "curateur a la faillite [de] X" (bankruptcy trustee/curator)
      or any other wording naming one person as trustee/administrator/
      representative for another named person, this is NOT "groep". Only the
      person under trusteeship (the one named after "faillite [de]") is the
      shareholder - use "persoon" for them, and put the trustee in
      "vertegenwoordiger" instead (see 2.3 for the full pattern and a worked
      example). Reserve "groep" for entries where the named people are
      genuine co-owners with no such trustee/representative relationship
      between them (e.g. spouses, siblings, business partners all holding the
      shares jointly). A shareholder and a representative are NOT a group!
      IMPORTANT - check this BEFORE picking "onbekend": if a row merely
      introduces or sums the OTHER entries in this table (e.g. "The remaining
      785.96 shares are subscribed by the following:" immediately before a
      list of named people whose shares add up to that figure, or a closing
      "Total: ..." line), it is NOT its own entry at all, "onbekend" or
      otherwise - skip it here and capture its figure in "totalen" (2.1)
      instead. Only use "onbekend" when the row's shares are NOT accounted
      for by any other entry in this table.
    - "aandeelhouder": the full name and titles of the person, group, or
      company, exactly as printed. For "onbekend", the descriptive phrase
      identifying the block of shares, exactly as printed (e.g. everything
      before the share count itself).
- "adres": the address exactly as printed, in full; empty string if none is
  printed. Do not split it into parts.
- "beroep": general profession(s)/occupation(s) printed for this entry
  (e.g. "négociant", "administrateur"), exactly as printed - not necessarily
  related to any representative role. Empty array if none.

2.3 Representative(s): "vertegenwoordiger"
--------------------------------------------
This field is REQUIRED - actively check the entry's text for a representative
before defaulting to an empty array. Do not skip this field.

Shareholders (individuals or companies) are sometimes represented by another
person at the meeting. Capturing this correctly matters.

Add one item per person representing this shareholder, in any of these forms:
- "... vertegenwoordigd door ..." / "... represente(e) par ..."
- a trustee or "curateur"
- "bevoegde persoon" / "assiste de ..." / "pour ..." / "voor ..."

A representative is ALWAYS a natural person. If no representative is stated,
leave "vertegenwoordiger" as an empty array.

A common pattern: "[Name A], [profession], curateur a la faillite [de] [Name
B], proprietaire de N actions ... et [Name B], [profession], tous deux a
[place]." Here the bankrupt person named in the "faillite" phrase ([Name B])
is the actual shareholder, not the curator - the curator only administers the
estate. In this pattern:
- "aandeelhouder": {"type": "persoon", "aandeelhouder": [Name B]} - the
  bankrupt individual named after "faillite" or after "et".
- "vertegenwoordiger": one item for [Name A], with "curateur" and their own
  profession (e.g. "avocat") both in "beroep".
- A trailing phrase like "tous deux a [place]" / "tous les deux a [place]"
  gives the address for BOTH people - copy it into "adres" for the
  shareholder entry AND into "adres" for the representative item.
Do not fold the curator's name or profession into the shareholder's own
"aandeelhouder" or "beroep" fields - keep the two people, and their separate
professions, apart.

Each item has:
- "vertegenwoordiger": the representative's full name, exactly as printed.
- "adres": the representative's address exactly as printed, in full, if given;
  empty string if none is printed.
- "beroep": general profession(s)/occupation(s) printed for the representative
  (e.g. "avocat", "curateur"), exactly as printed. Empty array if none.

"Pour ..." / "Voor ..." rows: a row whose text begins with "Pour"/"Voor"
("for") is still its own shareholder - record its "aandeelhouder",
"adres", shares, etc. normally. Do NOT guess at a representative for these
rows and do NOT copy one forward yourself - leave "vertegenwoordiger" as an
empty array for them. The representative (the person named on the most
recently preceding row that did NOT itself start with "Pour"/"Voor") is
resolved automatically downstream from the row's own printed text.

2.4 Share types: "type"
---------------------------------
This field is REQUIRED - populate it whenever any share count is printed for
the entry, even if there is only a single, implicit share type. Only leave it
as an empty array when the entry prints no share count at all (e.g. a
representative-only row, or a person listed with no shares of their own).

One item per distinct share type/class printed for this entry. Each item has:
- "type": the share type/class/denomination as printed (e.g. "actions de
  capital", "gewone aandelen"). Empty string if none is printed.
- "aantal": REQUIRED. The number of shares in the full written form exactly as
  printed, including any fraction wording (e.g. "cent nonante-cinq", or
  "douze et quatre cinquiemes" when a fraction is stated). A fraction can sit
  right after the whole number ("douze et quatre cinquiemes actions") OR be
  separated from it by other words in between ("cent nonante-cinq actions ET
  quatre cinquiemes D'ACTIONS entierement liberees" - the "quatre cinquiemes"
  still belongs to the same "cent nonante-cinq" count, giving "cent
  nonante-cinq et quatre cinquiemes", even though "d'actions entierement
  liberees" sits between them). Read the whole sentence before deciding a
  count is "done" - do not drop a fraction just because other words come
  between it and its whole number.
- "aantal_numeriek": REQUIRED. The same count in numeral form, mirroring the
  source's OWN numeral notation exactly as printed - including any fraction
  shorthand the source itself uses (e.g. "12.4" if that bare decimal-look mark
  is literally what follows the whole number in print, or "195.4" for the
  separated-fraction example above). Do not correct or reinterpret anything
  here - this is the literal digit as printed, even where it looks
  inconsistent with "aantal" (a downstream process re-derives the true value
  from "aantal" when the two disagree).
- "waarde": the value of this share block, if a separate monetary value is
  actually printed (not derived from "aantal"), always as a plain number but
  including the currency if printed. Empty string if no value is printed.

Even when the entry only states one total (e.g. "proprietaire de cent quinze
actions de capital", with no separate breakdown by class), still add ONE
"type" item for it, using the printed label as "type" (e.g. "actions de
capital").

2.5 Votes: "stemmen"
------------------------
This field is REQUIRED - an empty array if no vote count is printed for this
entry. One item per distinct vote-count mention printed for this entry, in
the same shape as 2.4's share items:
- "type": the vote type as printed (e.g. "stem", "voix"). Empty string if
  none is printed.
- "aantal": REQUIRED. The number of votes in the full written form exactly as
  printed, including any fractional wording.
- "aantal_numeriek": REQUIRED. The same count in numeral form, mirroring the
  source's OWN numeral notation exactly as printed - same rule as 2.4's
  "aantal_numeriek": copy literally, do not correct or reinterpret.

------------------------------------------------------------------------------

STEP 3 - If "is_shareholder_register" is false

Fill in "generic_table" and leave "shareholder_register" null.

"generic_table" has:

- "title": the OUTER heading that names the whole statement, printed directly
  above the table, if any; otherwise null (e.g. "Bilan au 31 décembre 1914",
  "Profits et pertes"). Reserve this for that outer heading only - a
  secondary heading that introduces just PART of this same table's rows
  (e.g. "Actif"/"Passif" under a "Bilan" title, or "Débit"/"Crédit" under a
  "Profits et pertes" title) is NOT the table's "title" - it belongs in
  "rows" instead, as its own "section" row (see below). Do not let a
  secondary in-table heading get pulled up into "title" just because it also
  sits above some rows.
- "table_type": best-fit category for the table's layout -
    - "financial_statement" for a bilan-style table of indented labels and
      amount columns
    - "grid_table" for a plain table with an explicit column-header row that
      isn't a shareholder register
    - "other" for anything else
- "columns": printed column headers left to right, or an empty array if there
  are none (common for financial statements, which usually just have one or
  two unlabeled amount columns).
- "rows": every row, top to bottom, in reading order. Do not skip, merge, or
  reorder rows, and do not summarize a row's label. Each row has:
    - "row_type": one of:
        - "header" - a printed column-header row inside the table
        - "section" - a label-only row with no values that introduces the
          rows below it (e.g. "Envers la societe :", "Immobilise :"). This is
          ALSO the right row_type for a secondary heading like "Actif"/
          "Passif" or "Débit"/"Crédit" that divides ONE table into named
          parts - do not treat these as the table's own "title" (see above);
          enter them here as their own "section" row instead, exactly
          like any other section header.
        - "item" - a row with a label and one or more values
        - "subtotal" - a row ruled off with a single line above its values
        - "total" - a final row ruled off with a double line, or introduced
          by a word like "Ensemble" or "Total"
    - "label": the row's text, copied exactly as printed; empty string
      if the row has no label text of its own.
    - "values": the row's numeric/amount cells, left to right, matching the
      order of "columns"; empty array for label-only rows.
    - "section": the exact "label" of the "section" row this row falls
      under, so each row's context is explicit rather than something a
      downstream reader has to re-derive from row order (which breaks once a
      table has more than one section, or a subtotal/total row that closes
      one section while others still follow). Copy the governing section's
      OWN "label" text verbatim - do not paraphrase or shorten it, so it can
      be matched back to that section row exactly. Empty string for a
      "header" row, for a "section" row itself (it doesn't belong under
      another section, even if one printed above it - only actual nesting
      one section literally inside another counts), or for any row printed
      before the table's first section header.
- "footnotes": footnote text printed below the table, if any; otherwise null.

------------------------------------------------------------------------------

RULES THAT APPLY TO BOTH STEP 2 AND STEP 3

- Keep original spelling, abbreviations, and punctuation. Do not modernize or
  translate anything.
- Record numbers exactly as printed, including thousands separators
  (commas or periods), decimal/centime parts, and any placeholder mark such as
  "»", "-", or "id." used in place of a repeated or nil value; convert them to
  plain numbers only where the schema calls for a numeric field (e.g.
  "waarde") - "aantal_numeriek" is the one exception, which must always be
  the literal digit as printed, never corrected.
- Ignore leader dots between a label and its value (or between any two
  columns); they are layout, not content. Never copy a run of repeated
  dots/periods into any field - not even a partial or shortened run. If you
  notice yourself about to output several dots in a row, stop and skip
  straight to the next real piece of text instead.
- If a word or figure is illegible or you are guessing at a damaged character,
  keep your best-effort reading in the cell.
- If a field asked for by the schema has no printed source text to fill it
  with (e.g. no address is printed, no representative is stated, no value is
  given), use that field's own "nothing printed" value - an empty string or
  empty array as noted per field, or null only where a field's own
  description says so (e.g. "title"/"footnotes") - rather than guessing or
  inventing content.
- Output must match the given JSON schema exactly.



"""

USER_PROMPT = (
    "Structure this table into the JSON schema you were given. "
    "Return only the JSON object, matching every field in the schema."
)

# Used when the model is only structuring an already-OCR'd transcription
# (see vllm_client.structure_transcription) rather than reading the image
# itself - SYSTEM_PROMPT's rules still apply unchanged (they describe how to
# interpret content, not how to read pixels).
USER_PROMPT_FROM_TEXT = (
    "Structure the following OCR text of a table into the JSON schema "
    "you were given, matching every field. The text below is already "
    "as printed - do not re-read or second-guess it, just structure it. "
    "Return only the JSON object.\n\n{text}"
)
