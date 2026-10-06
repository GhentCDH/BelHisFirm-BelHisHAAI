""" BelHisHAAI dashboard: runs the record pipeline one stage at a time as a subprocess, streaming its log. """

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Checkbox, Footer, Input, Label, RichLog, Static

from src.belhisapp.clipboard import ClipboardInput
from src.belhisapp.pipeline_runner import (
    DEFAULT_YOLO_WEIGHTS,
    REPO_ROOT,
    build_stage_command,
    check_vllm_reachable,
    classify_line,
    format_duration,
)
from src.belhisapp.splash import BANNER, BootSplashScreen, HIGHLIGHT_CONFIG, LoopingEffectLabel
from src.belhisapp.theme import (
    ABORTED, ACCENT, BASE, BORDER, DARK, DONE, FAILED, LEFT_COLUMN_WIDTH, LOG_STYLES, PANEL_BG,
    PHOSPHOR_THEME, RUNNING, SKIPPED,
)


@dataclass
class Stage:
    number: int
    key: str
    label: str
    checkbox: Checkbox = field(init=False, default=None)
    status: Label = field(init=False, default=None)


class BelhisApp(App):
    CSS = f"""
    Screen {{
        background: {DARK};
        color: {BASE};
    }}

    #body {{
        height: 1fr;
    }}

    #left-column {{
        width: {LEFT_COLUMN_WIDTH};
        height: 100%;
    }}

    #banner {{
        background: {DARK};
        margin-bottom: 1;
    }}

    #log-column {{
        width: 1fr;
        height: 100%;
    }}

    #controls-scroll {{
        height: 1fr;
        scrollbar-color: {BORDER};
        scrollbar-color-hover: {ACCENT};
        scrollbar-color-active: {ACCENT};
        scrollbar-background: {DARK};
    }}

    .panel {{
        height: auto;
        border: double {BASE};
        background: {DARK};
        color: {BASE};
        padding: 1 2;
        margin-bottom: 1;
    }}

    .panel-title {{
        text-style: bold underline;
        color: {ACCENT};
        margin-bottom: 1;
    }}

    .stage-row {{
        height: 1;
        align: left middle;
    }}

    .stage-label {{
        width: 42;
    }}

    .stage-status {{
        width: 1fr;
        text-align: right;
    }}

    .status-pending {{ color: {BASE}; }}
    .status-running {{ color: {RUNNING}; text-style: bold; }}
    .status-done {{ color: {DONE}; text-style: bold; }}
    .status-skipped {{ color: {SKIPPED}; }}
    .status-failed {{ color: {FAILED}; text-style: bold; }}
    .status-aborted {{ color: {ABORTED}; text-style: bold; }}

    .button-row {{
        height: auto;
        margin-top: 1;
        align: center middle;
    }}

    Input {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Input:focus {{
        border: solid {ACCENT};
    }}

    Checkbox {{
        height: 1;
        border: none;
        padding: 0 1;
        background: {DARK};
        color: {BASE};
    }}

    Checkbox:focus {{
        background-tint: {PANEL_BG} 40%;
    }}

    Button {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Button:hover {{
        background: {BASE};
        color: {DARK};
    }}

    Button:focus {{
        text-style: bold;
        border: solid {ACCENT};
    }}

    Button:disabled {{
        color: {SKIPPED};
        border: solid {SKIPPED};
    }}

    #run-btn {{
        border: solid {DONE};
        color: {DONE};
    }}

    #abort-btn {{
        border: solid {FAILED};
        color: {FAILED};
    }}

    #test-vllm-btn {{
        border: solid {RUNNING};
        color: {RUNNING};
        margin-top: 1;
    }}

    #log {{
        height: 1fr;
        border: double {BASE};
        background: {DARK};
        color: {BASE};
        scrollbar-color: {BORDER};
        scrollbar-color-hover: {ACCENT};
        scrollbar-color-active: {ACCENT};
        scrollbar-background: {DARK};
    }}

    Label {{
        color: {BASE};
    }}
    """

    BINDINGS = [("q", "quit", "Quit")]
    TITLE = "BelHisHAAI"

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(PHOSPHOR_THEME)
        self.theme = "phosphor"
        self.stages: list[Stage] = [
            Stage(1, "records", "1/3  RECORD SPLITTING"),
            Stage(2, "tables", "2/3  TABLE PROCESSING"),
            Stage(3, "ocr", "3/3  OCR + EXPORT"),
        ]
        self._process: subprocess.Popen | None = None
        self._aborted = False
        self._elapsed_timer: Timer | None = None
        self._stage_start = 0.0
        self._running_stage: Stage | None = None

    # -- layout --

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="left-column"):
                banner = LoopingEffectLabel(BANNER, effect="Highlight", config=HIGHLIGHT_CONFIG)
                banner.id = "banner"
                yield banner
                with VerticalScroll(id="controls-scroll"):
                    with Vertical(classes="panel", id="paths-panel"):
                        yield Label("PATHS", classes="panel-title")
                        yield Label("PAGES (input dir)")
                        yield ClipboardInput(placeholder="input/staatsblad_1923", id="pages-input")
                        yield Label("OUTPUT (output dir)")
                        yield ClipboardInput(placeholder="output/run_042", id="output-input")

                    with Vertical(classes="panel"):
                        yield Label("STAGES", classes="panel-title")
                        for stage in self.stages:
                            with Horizontal(classes="stage-row"):
                                stage.checkbox = Checkbox(value=True, id=f"stage-{stage.key}-check")
                                yield stage.checkbox
                                yield Label(stage.label, classes="stage-label")
                                stage.status = Label("PENDING", classes="stage-status status-pending")
                                yield stage.status

                    with Vertical(classes="panel"):
                        yield Label("ADVANCED", classes="panel-title")
                        yield Label("YOLO weights")
                        yield Input(value=str(DEFAULT_YOLO_WEIGHTS), id="weights-input")
                        yield Label("vLLM model")
                        yield Input(value="qwen3.6-35b-nvfp4", id="model-input")
                        yield Label("vLLM base URL")
                        yield Input(value="http://localhost:8000/v1", id="base-url-input")
                        yield Button("TEST CONNECTION", id="test-vllm-btn")

            with Vertical(id="log-column"):
                yield RichLog(id="log", wrap=True, highlight=False, markup=False)
                with Horizontal(classes="button-row"):
                    yield Button("RUN PIPELINE", id="run-btn", variant="success")
                    yield Button("ABORT", id="abort-btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(BootSplashScreen(), self._after_splash)

    def _after_splash(self, _result: object = None) -> None:
        self._log("=== SYSTEM CHECK ===", "header")
        self._log_check("YOLO weights", DEFAULT_YOLO_WEIGHTS.exists())
        self.run_worker(self._startup_vllm_check, thread=True, group="vllm-test")
        self._log("")
        self._log("Ready. Set PAGES/OUTPUT, toggle stages, press RUN PIPELINE.")

    def _startup_vllm_check(self) -> None:
        base_url = self.query_one("#base-url-input", Input).value.strip()
        ok, detail = check_vllm_reachable(base_url)
        if ok:
            self.call_from_thread(self._log, f"[ OK ]  vLLM base URL ({base_url})", "ok")
        else:
            self.call_from_thread(self._log, f"[ FAIL ]  vLLM base URL unreachable ({base_url}): {detail}", "fail")

    def _log(self, text: str, style: str | None = None) -> None:
        log = self.query_one("#log", RichLog)
        log.write(Text(text, style=LOG_STYLES.get(style, style)) if style else text)

    def _log_check(self, name: str, ok: bool) -> None:
        self._log(f"[ {'OK' if ok else 'FAIL'} ]  {name}", "ok" if ok else "fail")

    # -- run / abort --

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self._start_pipeline()
        elif event.button.id == "abort-btn":
            self._abort_pipeline()
        elif event.button.id == "test-vllm-btn":
            self.run_worker(self._test_vllm_connection, thread=True, group="vllm-test")

    def _test_vllm_connection(self) -> None:
        base_url = self.query_one("#base-url-input", Input).value.strip()
        self.call_from_thread(self._log, f"\nTesting vLLM connection at {base_url} ...")
        ok, detail = check_vllm_reachable(base_url)
        if ok:
            self.call_from_thread(self._log, f"[ OK ]  vLLM reachable ({detail})", "ok")
        else:
            self.call_from_thread(self._log, f"[ FAIL ]  vLLM unreachable: {detail}", "fail")

    def _set_status(self, stage: Stage, text: str, css_class: str) -> None:
        stage.status.set_classes(f"stage-status {css_class}")
        stage.status.update(text)

    def _start_pipeline(self) -> None:
        pages_value = self.query_one("#pages-input", Input).value.strip()
        output_value = self.query_one("#output-input", Input).value.strip()

        if not output_value:
            self._log("[ FAIL ] OUTPUT directory is required.", "fail")
            return
        if self.stages[0].checkbox.value and not pages_value:
            self._log("[ FAIL ] PAGES directory is required when RECORD SPLITTING is enabled.", "fail")
            return
        if self.stages[0].checkbox.value and not Path(pages_value).exists():
            self._log(f"[ FAIL ] PAGES directory not found: {pages_value}", "fail")
            return

        for stage in self.stages:
            if stage.checkbox.value:
                self._set_status(stage, "PENDING", "status-pending")
            else:
                self._set_status(stage, "SKIPPED", "status-skipped")

        self._aborted = False
        self.query_one("#run-btn", Button).disabled = True
        self.query_one("#abort-btn", Button).disabled = False
        self._log("\n=== PIPELINE START ===", "header")
        self.run_worker(self._run_pipeline_worker, thread=True, exclusive=True, group="pipeline")

    def _abort_pipeline(self) -> None:
        self._aborted = True
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._running_stage is not None:
            self._set_status(self._running_stage, "ABORTED", "status-aborted")

    def _run_pipeline_worker(self) -> None:
        pipeline_start = time.monotonic()
        elapsed_by_stage: dict[str, float] = {}
        failed = False

        output = Path(self.query_one("#output-input", Input).value.strip())
        pages_text = self.query_one("#pages-input", Input).value.strip()
        pages = Path(pages_text) if pages_text else None
        weights = Path(self.query_one("#weights-input", Input).value.strip())
        vllm_url = self.query_one("#base-url-input", Input).value.strip()
        vllm_model = self.query_one("#model-input", Input).value.strip()

        try:
            for stage in self.stages:
                if self._aborted:
                    break
                if not stage.checkbox.value:
                    continue

                self.call_from_thread(self._set_status, stage, "RUNNING 0.0s", "status-running")
                self.call_from_thread(self._log, f"\n=== {stage.label} ===", "header")
                self._running_stage = stage
                self._stage_start = time.monotonic()
                self._elapsed_timer = self.call_from_thread(self.set_interval, 1.0, lambda s=stage: self._tick(s))

                if stage.key == "tables":
                    ok, detail = check_vllm_reachable(vllm_url)
                    if ok:
                        self.call_from_thread(self._log, f"[ OK ]  vLLM reachable at {vllm_url} ({detail})", "ok")
                    else:
                        self.call_from_thread(
                            self._log, f"[ WARN ]  vLLM may be unreachable at {vllm_url}: {detail} - attempting anyway", "warn"
                        )

                try:
                    command = build_stage_command(stage.key, output, pages, weights, vllm_url, vllm_model)
                    self.call_from_thread(self._log, f"$ {' '.join(command)}", "dim")
                    # Child stdout is a pipe, so Python would block-buffer it without this flag.
                    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                    self._process = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                    )
                    for line in self._process.stdout:
                        self.call_from_thread(self._log, line.rstrip(), classify_line(line))
                    returncode = self._process.wait()
                except Exception as e:
                    self._stop_timer()
                    self.call_from_thread(self._log, f"[ FAIL ] {type(e).__name__}: {e}", "fail")
                    self.call_from_thread(self._set_status, stage, "FAILED", "status-failed")
                    failed = True
                    break

                self._stop_timer()
                elapsed = time.monotonic() - self._stage_start
                elapsed_by_stage[stage.key] = elapsed

                if self._aborted:
                    break
                if returncode != 0:
                    self.call_from_thread(self._set_status, stage, "FAILED", "status-failed")
                    self.call_from_thread(self._log, f"[ FAIL ] stage exited with code {returncode}", "fail")
                    failed = True
                    break

                self.call_from_thread(self._set_status, stage, f"DONE {format_duration(elapsed)}", "status-done")

            if not failed and not self._aborted:
                total = time.monotonic() - pipeline_start
                self.call_from_thread(self._log, "\n=== TIMELINE ===", "header")
                for stage in self.stages:
                    if stage.key in elapsed_by_stage:
                        self.call_from_thread(self._log, f"{stage.label:<28} {format_duration(elapsed_by_stage[stage.key]):>10}")
                self.call_from_thread(self._log, f"{'TOTAL':<28} {format_duration(total):>10}", "header")
                self.call_from_thread(self._log, "\nPipeline complete.", "ok")
            elif self._aborted:
                self.call_from_thread(self._log, "\n=== PIPELINE ABORTED ===", "warn")
        finally:
            self._running_stage = None
            self._process = None
            self.call_from_thread(self._reset_buttons)

    def _tick(self, stage: Stage) -> None:
        if self._aborted:
            return
        elapsed = time.monotonic() - self._stage_start
        self._set_status(stage, f"RUNNING {elapsed:.1f}s", "status-running")

    def _stop_timer(self) -> None:
        if self._elapsed_timer is not None:
            self.call_from_thread(self._elapsed_timer.stop)
            self._elapsed_timer = None

    def _reset_buttons(self) -> None:
        self.query_one("#run-btn", Button).disabled = False
        self.query_one("#abort-btn", Button).disabled = True

    def action_quit(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self.exit()
