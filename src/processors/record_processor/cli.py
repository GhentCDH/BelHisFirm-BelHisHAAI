""" Command-line entry point for the record pipeline, one stage per invocation.

    uv run python -m src.processors.record_processor.cli --stage records --pages input/ --output output/
    uv run python -m src.processors.record_processor.cli --stage tables  --output output/
    uv run python -m src.processors.record_processor.cli --stage ocr     --output output/
"""

import argparse
import logging
import sys
from pathlib import Path

from src.processors.record_processor.record_processor import RecordProcessor

STAGES = ["records", "tables", "ocr", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the record pipeline (or one stage of it).")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--pages", type=Path, help="Folder of scanned pages (required for the records stage)")
    parser.add_argument("--output", required=True, type=Path, help="Output folder shared by all stages")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--vllm-model", default="qwen3.6-35b-nvfp4", help="Model name served by vLLM")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stage in ("records", "all"):
        if args.pages is None:
            parser.error("--pages is required for the records stage")
        if not args.pages.exists():
            parser.error(f"--pages folder not found: {args.pages}")
    if not args.weights.exists():
        parser.error(f"YOLO weights not found: {args.weights}")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s", force=True)

    needs_ocr = args.stage in ("ocr", "all")
    needs_tables = args.stage in ("tables", "all")

    processor = RecordProcessor(
        skip_ocr=not needs_ocr,
        skip_tables=not needs_tables,
        yolo_model_path=str(args.weights),
        vllm_base_url=args.vllm_url,
        vllm_model=args.vllm_model,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    if args.stage in ("records", "all"):
        processor.run_records(args.pages, args.output)
    if args.stage in ("tables", "all"):
        processor.run_tables(args.output)
    if args.stage in ("ocr", "all"):
        processor.run_ocr(args.output)

    print(f"Stage '{args.stage}' complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
