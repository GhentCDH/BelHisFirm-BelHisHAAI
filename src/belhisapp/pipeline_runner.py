""" Builds the per-stage subprocess commands and checks what the dashboard needs before a run. """

import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "model" / "best.pt"
CLI_MODULE = "src.processors.record_processor.cli"


def build_stage_command(stage: str, output: Path, pages: Path | None, weights: Path, vllm_url: str, vllm_model: str) -> list[str]:
    command = [sys.executable, "-m", CLI_MODULE, "--stage", stage, "--output", str(output),
               "--weights", str(weights), "--vllm-url", vllm_url, "--vllm-model", vllm_model]
    if stage == "records" and pages is not None:
        command += ["--pages", str(pages)]
    return command


def check_vllm_reachable(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    try:
        response = requests.get(url, timeout=timeout)
        return response.ok, f"HTTP {response.status_code}"
    except requests.RequestException as e:
        return False, f"{type(e).__name__}: {e}"


def classify_line(line: str) -> str | None:
    """ Picks a log color for one line of stage output. """
    lowered = line.lower()
    if "traceback" in lowered or "error" in lowered or "failed" in lowered:
        return "fail"
    if "warning" in lowered or "warn" in lowered:
        return "warn"
    if "complete" in lowered or "finished" in lowered:
        return "ok"
    return None


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
