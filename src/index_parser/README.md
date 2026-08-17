# Index Parser

Extracts structured records (company name, address, event, record ID, ...) from the
alphabetical/annual index ("table") pages of the *Annexe au Moniteur belge* — scanned
`.tif` pages in, an Excel spreadsheet per page out.

Pipeline per page: **Surya** (layout detection) crops each text line → **Qwen3-VL**
(OCR2.py) transcribes it → text is cleaned, merged into full entries, and reshaped to
match the CRF model's training format → **CRF model** (`sklearn-crfsuite`) labels each
token (RecordID / Name / Address / Event / ...) → written to `.xlsx` with problem rows
highlighted for review.

## Quick start

```bash
# Process a whole volume — page range and output location are auto-detected
uv run src/index_parser/workflow2.py --folder /path/to/TIF/1887

# Just check where the index would be found, without running the full pipeline
uv run src/index_parser/workflow2.py --folder /path/to/TIF/1887 --dry

# Volumes from 1903 onward need this flag (see "Entry format eras" below)
uv run src/index_parser/workflow2.py --folder /path/to/TIF/1903 --modern-format

# Validate a single page before committing to a full run
uv run src/index_parser/workflow2.py --test-image /path/to/TIF/1912/IV/..._1189.tif --modern-format

# Process many folders unattended — see "Queue processing" below
uv run src/index_parser/workflow2.py --queue-file src/index_parser/queue.txt
```

## What gets auto-detected (so you rarely need to type more than `--folder`)

- **Which pages are the index.** `IndexLocator` OCRs just the top header band of each
  page in a search window (last `--search-window` pages by default; `--front-index`
  searches the first N instead) and looks for known header phrases. Two priority
  tiers: a comprehensive **yearly** table ("table annuelle", "recueil ... sociétés
  commerciales", ...) is preferred over a **regular** single/quarterly table ("table du
  recueil...", "table méthodique/alfabétique des matières") — the yearly tier only
  falls back to the regular one if nothing yearly is found at all. A page whose header
  also mentions a quarter/trimester ("1er TRIM.") is rejected even if it otherwise
  matches — that's a running header *inside* the annual table, not its start. The
  detected end is wherever page content (via a lightweight per-page box count) drops
  off to blank/near-blank filler pages.
  Override with `--start-page`/`--end-page` (skips detection), or point it at custom
  phrases with `--index-anchor "phrase"` (repeatable).

- **Where output goes.** `Index_Output/<year>[_<subpart>]` next to the input's
  `.../<collection>/<date>/TIF/<year>[/<subpart>]` folder (e.g. `.../TIF/1892/III-IV`
  → `Index_Output/1892_III-IV`). Each run gets its own `<suffix>_N` subfolder
  (`1887_1`, `1887_2`, ...), so nothing is ever overwritten. Override with `--output`.

- **Which entry-format era to use.** See below — `--modern-format` switches on the
  1903+ handling, and within that, the year found in the folder path decides whether
  the 1903-1911 or 1912+ variant applies. No extra flag for that second switch.

## Entry format eras

The raw text layout of an index entry changed twice across the scanned period, and the
CRF model was only ever trained on the original (pre-1897) shape. Rather than retrain
it, each later era is rewritten back into that shape before it reaches the model.

| Era | Example (as printed) | Handling |
|---|---|---|
| pre-1897 | `2513. Tramways de Nice, à Bruxelles. — Constitution.` | fed to the CRF as-is |
| 1897–1902 | `2412.3. Tramways de Nice, à Bruxelles. — Constitution.` (id**.quarter** leads) | quarter split off → `2412. Tramways...`, quarter kept in its own column |
| 1903–1911 (`--modern-format`) | `Tramways de Nice, à Bruxelles. — Constitution. — S. an., 2412.3.` (id.quarter now **trails**, often after a company-type abbreviation) | id/quarter/type extracted and the id moved back to the front; merge rules also change, since these entries contain internal periods (after the address, after the event) that would otherwise close the entry too early |
| 1912+ (`--modern-format`, auto-detected from the year) | No dashes at all. A **header** line (name, address, optional type) is stated once, followed by **one or more act lines** (`event, id.quarter.`) with no name repeat and no "Id." marker | header/act linking reconstructs one full entry per act, replacing the merge-rules step entirely (see `CRFPreProcessor.link_grouped_entries`) |

Company type (`S. an.`, `S. c.`, `S. coop.`, `S. c. a.`, ...) is never fed to the CRF
model — it's extracted and written to its own `CompanyType` column instead of being
reinserted into the sentence, since there's no reliable, unambiguous place to put it
back for every company name shape.

A post-hoc sanity check catches the index-range auto-detector occasionally landing on
a quarterly-only page instead of the true annual table (the same header text can
appear on both): if 5+ consecutive entries on a `--modern-format` page have no
`id.quarter` marker at all, that page is skipped and its thumbnail removed from the
output folder rather than written with mostly-empty RecordID/Quarter columns.

## Output

Each processed page produces, in the run's output folder:

- `<page>.xlsx` — one row per entry. Columns: `RecordID`, `Name`, `Addres`, `Event`,
  `ExtraInformation`, `Delimiter` (dropped before saving), plus `Full Text` (the
  original OCR'd line, for cross-checking), `Quarter`, `CompanyType`.
  Rows are colour-coded for review (red = missing/invalid field, yellow = duplicate
  RecordID on the page, orange = RecordID has no digits, green = name looks like a
  near-duplicate of another row) — see `<page>_explain.txt` for the reasons per row.
- `<page>.tif` — a scaled-down thumbnail of the source page, for quickly checking a
  flagged row against the original.
- `processing.log` — full run log (also printed to the terminal).

## Queue processing

`--queue-file path/to/queue.txt` processes a list of folders (one `--folder` path per
line, `#` comments allowed) instead of a single one. Models are loaded once and reused
across folders (freed and reloaded between each, so one folder's memory footprint can't
compound into an out-of-memory error on the next). A line is only removed from the
queue file once its folder finishes successfully — a failure leaves it in place for a
retry, and is logged instead to a `processed.txt` file next to the queue file so
there's a durable record of what actually completed. You can append new lines to the
queue file while it's running; it's re-read before every folder. It stops as soon as
the file has no pending lines left — it does not wait around for more.

## Other flags

- `--start-page` / `--end-page` — manual page range, skips auto-detection entirely.
- `--front-index` — the index is at the front of this volume, not the back; searches
  the first `--search-window` pages instead of the last.
- `--search-window N` (default 130) — how many pages the index-range scan covers.
- `--index-anchor "phrase"` (repeatable) — override the built-in header phrases used
  for auto-detection with your own (used as a single flat list, no yearly/regular
  tiering).
- `--model path.pkg` — CRF model file (default: `model/19thcenturyMoniteurV7.pkg`).
- `--binarize` — Otsu-threshold line crops to black-and-white before OCR.
- `--debug bbox|ocr|crf` — `bbox` saves layout-detection debug images and stops;
  `ocr` prints/stops after OCR, before CRF; `crf` runs everything and also prints CRF
  labels to the terminal.
- `--test-image path.tif` — run the full pipeline on one page only, output kept in a
  separate `..._test_N` folder. Useful for validating a change against a real page
  without committing to a full-volume run.

## Known limitations

- **Scan quality / tight line spacing**: on some densely-set pages, Surya's own line
  detection merges two physical lines into one oversized box before OCR ever runs,
  garbling the text. Traced to the detector itself, not the merge post-processing —
  a targeted fix was attempted and reverted after it was found to undo ~90% of the
  (necessary) merges on an already-validated page. No code-level fix currently;
  affected rows get caught by the same red/orange Excel highlighting as any other
  missing/malformed field.
- **1903-1911 company-type placement**: the type abbreviation is inserted back into
  the sentence, right after the company name (found via the last `à`/`te` address
  marker, not just the first comma — company names with their own internal commas
  can otherwise fool a naive "first comma" split). Untested for exotic name shapes
  beyond what's been checked against real pages so far.
- **Index-range detection is a heuristic**, not a guarantee — always spot-check with
  `--dry` (or `--test-image` against the reported start/end page) before committing
  a full volume run, especially for a year/era not yet validated against this tool.
