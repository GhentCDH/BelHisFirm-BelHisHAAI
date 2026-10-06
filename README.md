# BelHisFirm-BelHisHAAI

Human and agentic AI based data extraction workflow for BelHisFirm.

<div align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</div>

## What This Project Does

This repository contains:

- A Textual TUI app to navigate configuration and utilities.
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
- A running vLLM server (OpenAI-compatible API) hosting the table-structuring model (e.g. `qwen3.6-35b-nvfp4`), reachable at the URL passed to `RecordProcessor(vllm_base_url=..., vllm_model=...)` (default `http://localhost:8000/v1`). This is an external, separately-managed process — the pipeline does not start or manage it. Only needed if running with `skip_tables=False` (the default).

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

### Run the TUI app

```bash
python main.py
```

Inside the app footer, use:

- `utils` to open utilities
- `config` to open configuration form
- `quit` to exit

### Run the record pipeline directly (recommended for now)

The most reliable way to run end-to-end processing is currently from Python:

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
      page_001_table_00.png
      page_001_table_00.json
      FAILED/
        ...
  001-<record_title>/
    ...
  records_index.csv
```

Artifacts generated:

- `page_XXX.jpg`: cropped/split record pages
- `ocr_data.json`: OCR results and bounding boxes, plus a `tables` key with pointers to each table's crop/JSON (or error) under `tables/`
- `tables/page_NNN_table_NN.png` + `.json`: each detected table's crop image and structured transcription (plain text + schema-structured JSON); failures are mirrored under `tables/FAILED/` instead, with a `.txt` error message
- `<record_folder>.pdf`: searchable PDF (image + text layer for body text; table regions intentionally have no text layer, since GLM-OCR's table transcription has no per-line bounding boxes to anchor it to)
- `records_index.csv`: global index of all extracted records, including `num_tables`/`num_tables_failed` per record

Note: appending to a `records_index.csv` left over from before this pipeline upgrade will misalign columns — delete or move aside any pre-existing one before a fresh run.

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

1. YOLO model path is currently hardcoded in `src/processors/record_processor/record_processor.py`. The weights file itself (`model/best.pt`) is gitignored and must be placed there manually.
2. The config screen writes `src/config/config.json`, but the pipeline currently uses hardcoded values in `RecordProcessor`.
3. The Utilities form UI exists, but execution handling in `UtilRunWindow` is not wired yet, so direct Python invocation is recommended for pipeline runs.
4. Table structuring (`skip_tables=False`, the default) requires a separately-running vLLM server — see Prerequisites above. Pass `skip_tables=True` to run without one (tables are still detected and excluded from body OCR, just not transcribed/structured).
5. The YOLO layout model's header-candidate labels (`Section-header`, `Title`) and OCR-excluded labels (`Table`, `Picture`, `Formula`, `Caption`) are tuned for a DocLayNet-style taxonomy — see `ConfigParameter` in `record_processor.py` if the model is retrained with different class names.

## Project Layout (Main Parts)

- `main.py`: app entry point
- `src/belhisapp/`: Textual app and widgets
- `src/processors/record_processor/`: OCR and record pipeline
  - `pipeline/tables/`: table detection, cropping, GLM-OCR transcription, and vLLM structuring
  - `utils/glm_ocr_engine.py`: shared GLM-OCR model, used for header text, line OCR, and table transcription
- `src/utils/`: standalone utility scripts
- `model/`: YOLO model weights (for layout/header/table detection) — not git-tracked, place `best.pt` here manually
- `experiments/`: notebooks and research work
