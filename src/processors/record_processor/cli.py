""" Command-line entry point for the record pipeline, one stage per invocation.

    uv run python -m src.processors.record_processor.cli --stage records --pages input/ --output output/
    uv run python -m src.processors.record_processor.cli --stage tables  --output output/
    uv run python -m src.processors.record_processor.cli --stage ocr     --output output/

    Several stages can share one process, so the models are only loaded once:

    uv run python -m src.processors.record_processor.cli --stage tables ocr --output output/

    The tables stage has sub-stages (crop, transcribe, structure, parse, excel), pick some with --table-steps:

    uv run python -m src.processors.record_processor.cli --stage tables --table-steps parse excel --output output/
"""

import argparse
import logging
import sys
from pathlib import Path

from src.processors.record_processor.data.table_config import TABLE_STEPS
from src.processors.record_processor.record_processor import RecordProcessor

PIPELINE_STAGES = ["records", "tables", "ocr"]
STAGES = PIPELINE_STAGES + ["all"]


def announce_stage(stage: str, event: str) -> None:
    """ Marker line the dashboard follows a multi-stage run with (parsed by belhisapp.pipeline_runner.parse_stage_marker). """
    print(f"[stage:{stage}] {event}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the record pipeline (or one stage of it).")
    parser.add_argument("--stage", required=True, nargs="+", choices=STAGES,
                        help="One or more stages, always run in pipeline order")
    parser.add_argument("--pages", type=Path, help="Folder of scanned pages (required for the records stage)")
    parser.add_argument("--output", required=True, type=Path, help="Output folder shared by all stages")
    parser.add_argument("--weights", type=Path, default=Path("model/best.pt"), help="YOLO layout model weights")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--vllm-model", default="qwen3.6-35b-nvfp4", help="Model name served by vLLM")
    parser.add_argument("--table-steps", nargs="+", choices=TABLE_STEPS, default=TABLE_STEPS,
                        help="Sub-stages of the tables stage to run, always in pipeline order (default: all)")
    parser.add_argument("--glm-checkpoint", type=Path, default=None,
                        help="Fine-tuned GLM-OCR checkpoint for the table transcription (default: the base GLM-OCR model)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Structuring calls in flight against the vLLM server at once")
    parser.add_argument("--focus-shareholders", action="store_true",
                        help="Only keep an image copy next to the JSON of tables that are shareholder registers")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Always in pipeline order, whatever order they were given in
    stages = [stage for stage in PIPELINE_STAGES if stage in args.stage or "all" in args.stage]

    if "records" in stages:
        if args.pages is None:
            parser.error("--pages is required for the records stage")
        if not args.pages.exists():
            parser.error(f"--pages folder not found: {args.pages}")
    if not args.weights.exists():
        parser.error(f"YOLO weights not found: {args.weights}")
    if args.glm_checkpoint is not None and not args.glm_checkpoint.exists():
        parser.error(f"GLM-OCR checkpoint not found: {args.glm_checkpoint}")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s", force=True)

    needs_ocr = "ocr" in stages
    needs_tables = "tables" in stages

    processor = RecordProcessor(
        skip_ocr=not needs_ocr,
        skip_tables=not needs_tables,
        yolo_model_path=str(args.weights),
        vllm_base_url=args.vllm_url,
        vllm_model=args.vllm_model,
        table_steps=args.table_steps,
        glm_checkpoint=str(args.glm_checkpoint) if args.glm_checkpoint else None,
        structuring_concurrency=args.concurrency,
        focus_shareholders=args.focus_shareholders,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        announce_stage(stage, "start")
        if stage == "records":
            processor.run_records(args.pages, args.output)
        elif stage == "tables":
            processor.run_tables(args.output, on_step=lambda step, event: announce_stage(f"tables.{step}", event))
        else:
            processor.run_ocr(args.output)
        announce_stage(stage, "done")

    print(f"Stage '{', '.join(stages)}' complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
