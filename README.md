# BelHisFirm-BelHisHAAI

Human and agentic AI based data extraction workflow for BelHisFirm.

<div align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</div>

## What This Project Does

This repository contains:

- A Textual dashboard (BelHisFirm-TaPro style) that runs the pipeline one stage at a time, with live logs and ABORT.
- A record-processing pipeline that:
  - splits scanned pages into records based on detected headers (custom YOLO layout model),
  - detects and structures tables found within each record (YOLO detection → GLM-OCR transcription → vLLM-structured JSON),
  - runs OCR on each record page (Surya line detection + GLM-OCR transcription),
  - exports per-record images, JSON, searchable PDF, and an index CSV.
- Utility scripts for image conversion and validation.

## Quick Start

### 1. Prerequisites

- Linux (recommended for current setup)
- Python 3.13+
- NVIDIA GPU with CUDA support (required — YOLO, Surya, and GLM-OCR all run on GPU)
- The YOLO layout model weights must be placed manually at `model/best.pt` — this file is gitignored and not fetched automatically. It currently needs to be copied in from wherever the trained checkpoint lives (e.g. a sibling project's training run output).
- Optional: a fine-tuned GLM-OCR model (a merged model folder, not a LoRA adapter) placed or symlinked at `model/glm-ocr`. It is then used for the header, body and table transcription instead of the base `zai-org/GLM-OCR`.
- A running vLLM server (OpenAI-compatible API) hosting the table-structuring model (e.g. `qwen3.6-35b-nvfp4`), reachable at the URL passed to `RecordProcessor(vllm_base_url=..., vllm_model=...)` (default `http://localhost:8000/v1`). This is an external, separately-managed process — the pipeline does not start or manage it. Only needed for the table STRUCTURING sub-stage.
- For address parsing in the table RULE PARSING sub-stage: the geonames gazetteer, installed once machine-wide with `uv run python -m geoparser install geonames` (about 10 GB on disk, about 30 GB free space needed during install). Without it, addresses are left as plain text and the other rules still apply.

### 2. Install Python dependencies

This project includes a `pyproject.toml` and `uv.lock`, so `uv` is the easiest path.

```bash
uv sync
source .venv/bin/activate
```

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Basic Usage

### Run the dashboard

```bash
uv run python main.py
```

The dashboard works like the BelHisFirm-TaPro TUI:

- **PATHS**: input folder of scans (PAGES) and output folder (OUTPUT).
- **STAGES**: toggle `1/3 RECORD SPLITTING`, `2/3 TABLE PROCESSING` and `3/3 OCR + EXPORT`. Each stage shows PENDING, RUNNING, DONE, SKIPPED, FAILED or ABORTED.
- **Table sub-stages**: listed under `2/3 TABLE PROCESSING`, each with its own toggle and status: `2a TABLE DETECTION`, `2b OCR TRANSCRIPTION`, `2c STRUCTURING`, `2d RULE PARSING` and `2e EXCEL EXPORT` (the same five as BelHisFirm-TaPro). Untick one to skip it when its output is already on disk.
- **ADVANCED**: YOLO weights, vLLM model and base URL, TEST CONNECTION, a fine-tuned GLM-OCR checkpoint for the header, body and table transcription (blank = `model/glm-ocr` if present, else the base model), the structuring concurrency, and `Focus shareholders only` (keeps an image copy next to the JSON only for shareholder registers).
- **Errors**: any error turns the panel and log frames red and shows a banner with the error count above the log. Each error is logged with its full traceback, the stage it happened in shows `⚠ N ERR`, and all errors are repeated in a summary at the end of the log. The red state clears on the next run.
- **RUN PIPELINE** runs the enabled stages in order in one process, so the models are loaded once. **ABORT** stops the run. Press `q` to quit.

A stage can be skipped to resume a partial run, as long as its earlier outputs are already in the output folder. The STRUCTURING sub-stage needs a running vLLM server; without one its tables end up under `tables/FAILED/` and are retried on the next run.

### Run the pipeline from the command line

Each stage can run on its own. All stages share the same output folder:

```bash
uv run python -m src.processors.record_processor.cli --stage records --pages /path/to/images --output /path/to/output
uv run python -m src.processors.record_processor.cli --stage tables  --output /path/to/output
uv run python -m src.processors.record_processor.cli --stage ocr     --output /path/to/output
```

`--stage` also takes several stages (e.g. `--stage tables ocr`), which then share one process and load the models once. `--stage all` runs the three in order. Add `--weights`, `--vllm-url` and `--vllm-model` to override the defaults, and `--glm-checkpoint` to read headers, body text and tables with another fine-tuned GLM-OCR checkpoint than `model/glm-ocr`.

The tables stage runs its sub-stages `crop`, `transcribe`, `structure`, `parse` and `excel` in that order. Pick some with `--table-steps`, and tune them with `--concurrency` and `--focus-shareholders`:

```bash
uv run python -m src.processors.record_processor.cli --stage tables --table-steps parse excel --output /path/to/output
```

`transcribe` and `structure` skip tables that already have their output, so an interrupted or partly failed run can be continued.

### Run the record pipeline from Python

```python
from pathlib import Path
from src.processors.record_processor.record_processor import RecordProcessor

input_dir = Path("/path/to/images")
output_dir = Path("/path/to/output")

processor = RecordProcessor(
    skip_ocr=False,
    skip_tables=False,
    vllm_base_url="http://localhost:8000/v1",
    vllm_model="qwen3.6-35b-nvfp4",
)
processor.run(input_dir, output_dir)
```

### Input folder requirements

The pipeline scans one folder and processes files in sorted order. Supported extensions:

- `.jpg`
- `.jpeg`
- `.tif`
- `.jp2`

## Output Structure

For each detected record, the pipeline creates a folder:

```text
output/
  000-<record_title>/
    page_001.jpg
    page_002.jpg
    ocr_data.json
    000-<record_title>.pdf
    tables/
      _crops/
        crops.json
        page_001_table_00.png
      _transcriptions/
        page_001_table_00.txt
      page_001_table_00.png
      page_001_table_00.json
      page_001_table_00.xlsx
      page_001_table_00.csv
      FAILED/
        ...
  001-<record_title>/
    ...
  records_index.csv
```

Each record folder also contains `record_meta.json` (the record's header data, used by the later stages) `tables_index.json` (table pointers, written by the table stage) and `layout.json` (the YOLO layout regions per page, taken from the detection the record stage already does, so the table and OCR stages do not run YOLO again; they only re-detect for folders without a usable `layout.json`, e.g. from an older run or after the weights file changed).

Artifacts generated:

- `page_XXX.jpg`: cropped/split record pages
- `ocr_data.json`: OCR results and bounding boxes, plus a `tables` key with pointers to each table's crop/JSON (or error) under `tables/`
- `tables/_crops/`: every detected table's crop image and `crops.json` (page and bounding box per crop), written by TABLE DETECTION
- `tables/_transcriptions/page_NNN_table_NN.txt`: GLM-OCR plain text per crop, written by OCR TRANSCRIPTION
- `tables/page_NNN_table_NN.json` + `.png`: structured transcription (plain text + schema-structured JSON) and a copy of the crop, written by STRUCTURING; RULE PARSING then adds gender, parsed addresses and re-derived share counts to shareholder registers in place. Failures are mirrored under `tables/FAILED/` instead, with a `.txt` error message
- `tables/page_NNN_table_NN.xlsx` + `.csv`: workbook (one sheet per data shape) and flat CSV per table, written by EXCEL EXPORT
- `<record_folder>.pdf`: searchable PDF (image + text layer for body text; table regions intentionally have no text layer, since GLM-OCR's table transcription has no per-line bounding boxes to anchor it to)
- `records_index.csv`: global index of all extracted records, including `num_tables`/`num_tables_failed` per record. Re-running a stage replaces a record's row instead of adding a duplicate.

## Utility Scripts

### Convert JP2 to JPEG

```bash
python src/utils/jp2_to_jpeg.py /path/to/input_jp2 /path/to/output_jpeg
```

### Validate TIFF/JP2 files

```bash
python src/utils/validate_image.py /path/to/images
```

This writes a `validation_errors.log` in the scanned directory.

### Send text regions to Label Studio for OCR ground truth

Samples record headers, section headers, body text, captions and page headers/footers from scanned pages, adds a GLM-OCR guess to each crop, and imports them into a Label Studio project to be corrected into training data for GLM-OCR. Put `LS_URL`, `LS_TOKEN` and `LS_PROJECT_ID` in a `.env` file in the repo root (copy `.env.example`).

```bash
# Dry run: writes the crops to ocr_preanno_crops/ and the tasks to labelstudio_tasks.json
uv run python -m src.utils.labelstudio_preannotate.preannotate_regions --pages /path/to/images --per-label 20

# Import into Label Studio
uv run python -m src.utils.labelstudio_preannotate.preannotate_regions --pages /path/to/images --per-label 20 --commit
```

The project needs the same labeling config as the table transcription project of HisTableFinder (an image under `$captioning` with a `caption` text area), see the docstring of the script.

### Detect entities in record texts (DEBUG, GLiNER)

Temporary command line for a first entity pass, until it is wired into the dashboard. It runs GLiNER on every text block of each record's `ocr_data.json` (the training data of the adapters are paragraph-sized texts as well), with the 23 BelHisFirm labels of `src/processors/text_processor/gliner_schema_BelHisFirm.json` (label descriptions and thresholds, from the ghentcdh-glinerv2-tutorial). With `--print` the blocks are shown with the entities highlighted in the color of their domain (date, organisation, legal, person, share, location; underlined where entities overlap).

Each record gets an `entities.json` next to its `ocr_data.json`: every entity with its label, domain, text, confidence, character span (relative to its block) and the page, block index and bbox of the block.

```bash
uv run python -m src.processors.text_processor.cli --output /path/to/output --print
uv run python -m src.processors.text_processor.cli --output /path/to/output --records 055 019 --dry-run
uv run python -m src.processors.text_processor.cli --text "Landen, le 25 juin 1909." --print
# OCR page images first, then the entities (also temporary)
uv run python -m src.processors.text_processor.debug_ocr_cli --input page_001.jpg --print
```

The model is GLiNER2.5 (`fastino/gliner2.5-multi-v1`). The BelHisFirm LoRA adapters in `model/gliner/adapters/v25` are used when they exist (`--adapter-dir` for others, `--no-adapters` for the plain model): a folder with one adapter per domain (one pass per domain), or a single all-label adapter.

#### Annotate further with the model's pre-annotations

Once an adapter works, let it pre-annotate new records and only correct its mistakes in Label Studio, then retrain on the corrected data. `labelstudio_cli` takes a folder of page images (or the output folder of the record pipeline), runs the OCR (YOLO layout + GLM-OCR, as in the OCR stage) on what has no `ocr_data.json` yet, and makes every detection (YOLO text block with its OCR text, also those without entities) a Label Studio task of its own. By default only the `Text` and `Section-header` detections are taken (`--labels`, `all` for every text detection) and at most 10 per page (`--max-per-page`, 0 for all): the largest ones by surface, divided 20% section headers and 80% texts (`--split Section-header=20 Text=80`; a label with too few leaves its places to the other). `--max-pages N` takes at most N pages over the whole input, a random pick that is the same on every run (`--seed`). What is not picked is not read by the OCR at all. The task data keeps where it came from (`record_folder`, `page`, `block_index`, `bbox`, `yolo_label`) and a `source` key, so tasks that are already in the project are left out when you import again.

The tasks go to a Label Studio NER project (a `text` Text tag and a `label` Labels tag with the 23 labels, like project 120 `BelHisFirm NER #2`; `--print-config` prints the config for a new one). `LS_URL` and `LS_TOKEN` come from `.env`, give the project with `--project` or `LS_NER_PROJECT_ID`.

```bash
# 1. Dry run: OCR + entities, only writes labelstudio_ner_tasks.json (nothing is written into the input folder)
uv run python -m src.processors.text_processor.labelstudio_cli --input /path/to/pages --project 120
# 2. Import a few first, then everything (--from-tasks imports the file of the dry run without redoing the OCR)
uv run python -m src.processors.text_processor.labelstudio_cli --from-tasks labelstudio_ner_tasks.json --project 120 --commit --limit 20
uv run python -m src.processors.text_processor.labelstudio_cli --from-tasks labelstudio_ner_tasks.json --project 120 --commit
# 3. Correct the entities in Label Studio, export the project as JSON
# 4. Turn the export into training data, added to the existing data (that folder stays as it is)
uv run python -m src.processors.text_processor.gliner_convert --export ls_export.json --merge model/gliner/data --out model/gliner/data_round2
# 5. Retrain on it
uv run python -m src.processors.text_processor.gliner_train --data-dir model/gliner/data_round2 --out model/gliner/adapters/v25_round2
```

The corrected data only goes to the training set, `test.json` is copied unchanged so the rounds can be scored on the same texts. Review the pre-annotations rather than accepting them, otherwise the model's mistakes end up in its next training data.

#### Train the adapters

The training data (`train.json`, `train_<domain>.json`, `test.json`) comes from the tutorial repo and is copied to `model/gliner/data` (gitignored). The adapters are trained on GLiNER2.5 and saved to the folder the CLI reads by default:

```bash
uv run python -m src.processors.text_processor.gliner_train --out model/gliner/adapters/v25
```

Use `--domains date person` for some domains only, or `--single` for one adapter on all labels.

## Important Current Notes

1. The YOLO weights default to `model/best.pt`, which is gitignored and must be placed there manually. Override it with `--weights` on the CLI or the YOLO weights field in the dashboard.
2. Table processing needs a running vLLM server (see Prerequisites). Without one, the TABLE PROCESSING stage fails at its check. Uncheck that stage to run without it.
3. The YOLO layout model's header-candidate labels (`Section-header`, `Title`) and OCR-excluded labels (`Table`, `Picture`, `Formula`, `Caption`) are tuned for a DocLayNet-style taxonomy. See `RecordProcessor.__init__` if the model is retrained with different class names.

## Project Layout (Main Parts)

- `main.py`: app entry point
- `src/belhisapp/`: Textual app and widgets
- `src/processors/record_processor/`: OCR and record pipeline
  - `pipeline/tables/`: table detection, cropping, GLM-OCR transcription, and vLLM structuring
  - `utils/glm_ocr_engine.py`: shared GLM-OCR model, used for header text, line OCR, and table transcription
- `src/utils/`: standalone utility scripts
- `model/`: YOLO model weights (for layout/header/table detection) — not git-tracked, place `best.pt` here manually
- `experiments/`: notebooks and research work
