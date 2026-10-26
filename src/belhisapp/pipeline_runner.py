""" Builds the pipeline subprocess command and checks what the dashboard needs before a run. """

import re
import signal
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "model" / "best.pt"
CLI_MODULE = "src.processors.record_processor.cli"


# Printed by the CLI around each stage (announce_stage in cli.py). Matched here by format
# rather than imported, since importing the CLI would load torch into the dashboard.
STAGE_MARKER = re.compile(r"^\[stage:([\w.]+)\] (start|done)$")


ERROR_PREFIX = "[ ERROR ]"
ERROR_CONTINUATION = "[ ERROR ] |"  # a further line of the same error (its traceback)
WARNING_PREFIX = "[ WARN ]"


def build_pipeline_command(stages: list[str], output: Path, pages: Path | None, weights: Path, vllm_url: str, vllm_model: str,
                           table_steps: list[str], glm_checkpoint: Path | None, concurrency: int,
                           focus_shareholders: bool) -> list[str]:
    """ One command for all given stages, so they share a process and the models are loaded once. """
    command = [sys.executable, "-m", CLI_MODULE, "--stage", *stages, "--output", str(output),
               "--weights", str(weights), "--vllm-url", vllm_url, "--vllm-model", vllm_model]
    if "records" in stages and pages is not None:
        command += ["--pages", str(pages)]
    if glm_checkpoint is not None:
        command += ["--glm-checkpoint", str(glm_checkpoint)]
    if "tables" in stages:
        command += ["--table-steps", *table_steps, "--concurrency", str(concurrency)]
        if focus_shareholders:
            command.append("--focus-shareholders")
    return command


def parse_stage_marker(line: str) -> tuple[str, str] | None:
    """ (stage key, "start" | "done") for a stage marker line, None for any other line.
        A sub-stage has a dotted key, e.g. "tables.structure". """
    match = STAGE_MARKER.match(line.strip())
    return (match.group(1), match.group(2)) if match else None


def check_vllm_reachable(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    try:
        response = requests.get(url, timeout=timeout)
        return response.ok, f"HTTP {response.status_code}"
    except requests.RequestException as e:
        return False, f"{type(e).__name__}: {e}"


def classify_line(line: str) -> str | None:
    """ Picks a log color for one line of stage output. "error" is a line of an error the pipeline
        logged itself (LevelPrefixFormatter in cli.py), "fail" only looks like trouble. """
    if line.startswith(ERROR_PREFIX):
        return "error"
    if line.startswith(WARNING_PREFIX):
        return "warn"
    lowered = line.lower()
    if "traceback" in lowered or "error" in lowered or "failed" in lowered:
        return "fail"
    if "warning" in lowered or "warn" in lowered:
        return "warn"
    if "complete" in lowered or "finished" in lowered:
        return "ok"
    return None


def describe_exit(returncode: int) -> str:
    """ Why the pipeline process ended, for a non-zero exit. A killed process cannot log a reason itself. """
    if returncode >= 0:
        return f"The pipeline process stopped with exit code {returncode} before finishing."

    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"

    reason = f"The pipeline process was killed by {name} before finishing, so it could not log why."
    if -returncode == signal.SIGKILL:
        reason += (" This is almost always the system running out of memory:"
                   " check with `journalctl -k | grep -i 'out of memory'`.")
    return reason


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
