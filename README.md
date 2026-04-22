# BelHisFirm-BelHisHAAI

Human and agentic AI based data extraction workflow for BelHisFirm.

<div align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</div>

## What This Project Does

This repository contains:

- A Textual TUI app to navigate configuration and utilities.
- A record-processing pipeline that:
  - splits scanned pages into records based on detected headers,
  - runs OCR on each record page,
  - exports per-record images, JSON, searchable PDF, and an index CSV.
- Utility scripts for image conversion and validation.

## Quick Start

### 1. Prerequisites

- Linux (recommended for current setup)
- Python 3.13+
- Optional but recommended: NVIDIA GPU with CUDA support
- Tesseract OCR installed on your system (used for header text validation)

Install Tesseract on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

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
pip install ultralytics pytesseract
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
from src.recordprocessing.record_processor import RecordProcessor

input_dir = Path("/path/to/images")
output_dir = Path("/path/to/output")

processor = RecordProcessor(skip_ocr=False)
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
  001-<record_title>/
    ...
  records_index.csv
```

Artifacts generated:

- `page_XXX.jpg`: cropped/split record pages
- `ocr_data.json`: OCR results and bounding boxes
- `<record_folder>.pdf`: searchable PDF (image + text layer)
- `records_index.csv`: global index of all extracted records

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

1. YOLO model path is currently hardcoded in `src/recordprocessing/record_processor.py`.
2. The config screen writes `src/config/config.json`, but the pipeline currently uses hardcoded values in `RecordProcessor`.
3. The Utilities form UI exists, but execution handling in `UtilRunWindow` is not wired yet, so direct Python invocation is recommended for pipeline runs.

## Project Layout (Main Parts)

- `main.py`: app entry point
- `src/belhisapp/`: Textual app and widgets
- `src/recordprocessing/`: OCR and record pipeline
- `src/utils/`: standalone utility scripts
- `model/`: YOLO model weights (for layout/header detection)
- `experiments/`: notebooks and research work
