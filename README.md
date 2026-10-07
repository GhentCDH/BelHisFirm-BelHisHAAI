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
- **ADVANCED**: YOLO weights, vLLM model and base URL, TEST CONNECTION, a fine-tuned GLM-OCR checkpoint for the table transcription (blank = base model), the structuring concurrency, and `Focus shareholders only` (keeps an image copy next to the JSON only for shareholder registers).
- **RUN PIPELINE** runs the enabled stages in order in one process, so the models are loaded once. **ABORT** stops the run. Press `q` to quit.

A stage can be skipped to resume a partial run, as long as its earlier outputs are already in the output folder. The STRUCTURING sub-stage needs a running vLLM server; without one its tables end up under `tables/FAILED/` and are retried on the next run.

### Run the pipeline from the command line

Each stage can run on its own. All stages share the same output folder:

```bash
uv run python -m src.processors.record_processor.cli --stage records --pages /path/to/images --output /path/to/output
uv run python -m src.processors.record_processor.cli --stage tables  --output /path/to/output
uv run python -m src.processors.record_processor.cli --stage ocr     --output /path/to/output
```

`--stage` also takes several stages (e.g. `--stage tables ocr`), which then share one process and load the models once. `--stage all` runs the three in order. Add `--weights`, `--vllm-url` and `--vllm-model` to override the defaults.

The tables stage runs its sub-stages `crop`, `transcribe`, `structure`, `parse` and `excel` in that order. Pick some with `--table-steps`, and tune them with `--glm-checkpoint`, `--concurrency` and `--focus-shareholders`:

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
