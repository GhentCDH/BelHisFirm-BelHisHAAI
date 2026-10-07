""" BelHisHAAI dashboard: runs the enabled record pipeline stages in one subprocess, streaming its log. """

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
    ERROR_CONTINUATION,
    REPO_ROOT,
    build_pipeline_command,
    check_vllm_reachable,
    classify_line,
    describe_exit,
    format_duration,
    parse_stage_marker,
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
    parent: "Stage | None" = None  # set on a sub-stage, its key is then "<parent key>.<step>"
    checkbox: Checkbox = field(init=False, default=None)
    status: Label = field(init=False, default=None)
    started: float = field(init=False, default=0.0)
    errors: int = field(init=False, default=0)

    @property
    def step(self) -> str:
        return self.key.split(".")[-1]


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

    .substage-row {{
        height: 1;
        align: left middle;
        padding-left: 4;
    }}

    .substage-label {{
        width: 38;
        color: {BORDER};
    }}

    #focus-shareholders-check {{
        margin-top: 1;
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

    #alarm {{
        display: none;
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: {FAILED};
        color: {DARK};
    }}

    /* An error in the run turns the frames red until the next run */
    .has-errors #alarm {{
        display: block;
    }}

    .has-errors #log, .has-errors .panel {{
        border: double {FAILED};
    }}

    .has-errors .panel-title {{
        color: {FAILED};
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
        tables = self.stages[1]
        self.table_steps: list[Stage] = [
            Stage(1, "tables.crop", "2a  TABLE DETECTION", tables),
            Stage(2, "tables.transcribe", "2b  OCR TRANSCRIPTION", tables),
            Stage(3, "tables.structure", "2c  STRUCTURING", tables),
            Stage(4, "tables.parse", "2d  RULE PARSING", tables),
            Stage(5, "tables.excel", "2e  EXCEL EXPORT", tables),
        ]
        self._process: subprocess.Popen | None = None
        self._aborted = False
        self._elapsed_timer: Timer | None = None
        self._active_stages: list[Stage] = []  # the running stage, followed by its running sub-stage if any
        self._errors: list[tuple[str, list[str]]] = []  # (where it happened, lines of the error) per error of this run
        self._in_error = False  # whether the previous output line belonged to an error

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
                            for step in self.table_steps if stage.key == "tables" else []:
                                with Horizontal(classes="substage-row"):
                                    step.checkbox = Checkbox(value=True, id=f"step-{step.step}-check")
                                    yield step.checkbox
                                    yield Label(step.label, classes="substage-label")
                                    step.status = Label("PENDING", classes="stage-status status-pending")
                                    yield step.status

                    with Vertical(classes="panel"):
                        yield Label("ADVANCED", classes="panel-title")
                        yield Label("YOLO weights")
                        yield ClipboardInput(value=str(DEFAULT_YOLO_WEIGHTS), id="weights-input")
                        yield Label("vLLM model")
                        yield ClipboardInput(value="qwen3.6-35b-nvfp4", id="model-input")
                        yield Label("vLLM base URL")
                        yield ClipboardInput(value="http://localhost:8000/v1", id="base-url-input")
                        yield Button("TEST CONNECTION", id="test-vllm-btn")
                        yield Label("GLM-OCR table checkpoint (blank = base model)")
                        yield ClipboardInput(value="", id="checkpoint-input")
                        yield Label("Structuring concurrency")
                        yield ClipboardInput(value="8", id="concurrency-input")
                        yield Checkbox("Focus shareholders only (image copies)", id="focus-shareholders-check")

            with Vertical(id="log-column"):
                yield Static("", id="alarm")
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

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        # The sub-stages only apply while TABLE PROCESSING itself is enabled
        if event.checkbox is self.stages[1].checkbox:
            for step in self.table_steps:
                step.checkbox.disabled = not event.value

    def _selected_table_steps(self) -> list[Stage]:
        if not self.stages[1].checkbox.value:
            return []
        return [step for step in self.table_steps if step.checkbox.value]

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

        if self.stages[1].checkbox.value and not self._selected_table_steps():
            self._log("[ FAIL ] TABLE PROCESSING is enabled without any of its sub-stages.", "fail")
            return
        concurrency_value = self.query_one("#concurrency-input", Input).value.strip()
        if concurrency_value and not (concurrency_value.isdigit() and int(concurrency_value) > 0):
            self._log(f"[ FAIL ] Structuring concurrency must be a positive number: {concurrency_value}", "fail")
            return

        selected_steps = self._selected_table_steps()
        for stage in self.stages + self.table_steps:
            if stage.checkbox.value and (stage.parent is None or stage in selected_steps):
                self._set_status(stage, "PENDING", "status-pending")
            else:
                self._set_status(stage, "SKIPPED", "status-skipped")

        self._aborted = False
        self._errors = []
        self._in_error = False
        for stage in self.stages + self.table_steps:
            stage.errors = 0
        self.screen.remove_class("has-errors")
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
        for stage in self._active_stages:
            self._set_status(stage, "ABORTED", "status-aborted")

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

        checkpoint_text = self.query_one("#checkpoint-input", Input).value.strip()
        checkpoint = Path(checkpoint_text) if checkpoint_text else None
        concurrency = int(self.query_one("#concurrency-input", Input).value.strip() or "8")
        focus_shareholders = self.query_one("#focus-shareholders-check", Checkbox).value

        selected = [stage for stage in self.stages if stage.checkbox.value]
        selected_steps = self._selected_table_steps()
        stage_by_key = {stage.key: stage for stage in selected + selected_steps}

        try:
            if selected:
                if "tables.structure" in stage_by_key:
                    ok, detail = check_vllm_reachable(vllm_url)
                    if ok:
                        self.call_from_thread(self._log, f"[ OK ]  vLLM reachable at {vllm_url} ({detail})", "ok")
                    else:
                        self.call_from_thread(
                            self._log, f"[ WARN ]  vLLM may be unreachable at {vllm_url}: {detail} - attempting anyway", "warn"
                        )

                # All stages share one process, so the models are loaded once. The first stage
                # counts as running from here on, which puts the model loading on its clock.
                self._begin(selected[0])
                self._elapsed_timer = self.call_from_thread(self.set_interval, 1.0, self._tick)

                try:
                    command = build_pipeline_command(
                        [stage.key for stage in selected], output, pages, weights, vllm_url, vllm_model,
                        [step.step for step in selected_steps], checkpoint, concurrency, focus_shareholders,
                    )
                    self.call_from_thread(self._log, f"$ {' '.join(command)}", "dim")
                    # Child stdout is a pipe, so Python would block-buffer it without this flag.
                    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                    self._process = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                    )
                    for line in self._process.stdout:
                        marker = parse_stage_marker(line)
                        if marker is None or marker[0] not in stage_by_key:
                            style = classify_line(line)
                            self._track_error(line.rstrip(), style == "error")
                            self.call_from_thread(self._log, line.rstrip(), style)
                            continue

                        stage, event = stage_by_key[marker[0]], marker[1]
                        if event == "start":
                            self._begin(stage)
                        elif not self._aborted and stage in self._active_stages:
                            elapsed = time.monotonic() - stage.started
                            elapsed_by_stage[stage.key] = elapsed
                            self._active_stages.remove(stage)
                            if stage.errors:
                                self.call_from_thread(
                                    self._set_status, stage, f"⚠ {stage.errors} ERR  DONE {format_duration(elapsed)}", "status-failed"
                                )
                            else:
                                self.call_from_thread(self._set_status, stage, f"DONE {format_duration(elapsed)}", "status-done")
                    returncode = self._process.wait()
                except Exception as e:
                    self.call_from_thread(self._log, f"[ FAIL ] {type(e).__name__}: {e}", "fail")
                    returncode = None
                    failed = True

                self._stop_timer()

                if not self._aborted and (failed or returncode != 0):
                    failed = True
                    if returncode is not None:
                        reason = describe_exit(returncode)
                        self._track_error(reason, True)
                        self.call_from_thread(self._log, f"[ FAIL ] {reason}", "fail")
                    for stage in self._active_stages:
                        self.call_from_thread(self._set_status, stage, "FAILED", "status-failed")

            if not failed and not self._aborted:
                total = time.monotonic() - pipeline_start
                self.call_from_thread(self._log, "\n=== TIMELINE ===", "header")
                for stage in self.stages:
                    if stage.key in elapsed_by_stage:
                        self.call_from_thread(self._log, f"{stage.label:<28} {format_duration(elapsed_by_stage[stage.key]):>10}")
                    for step in self.table_steps:
                        if step.parent is stage and step.key in elapsed_by_stage:
                            self.call_from_thread(
                                self._log, f"    {step.label:<24} {format_duration(elapsed_by_stage[step.key]):>10}", "dim"
                            )
                self.call_from_thread(self._log, f"{'TOTAL':<28} {format_duration(total):>10}", "header")
                if not self._errors:
                    self.call_from_thread(self._log, "\nPipeline complete.", "ok")

            if self._errors and not self._aborted:
                self.call_from_thread(self._log_error_summary, failed)
            elif self._aborted:
                self.call_from_thread(self._log, "\n=== PIPELINE ABORTED ===", "warn")
        finally:
            self._active_stages = []
            self._process = None
            self.call_from_thread(self._reset_buttons)

    def _begin(self, stage: Stage) -> None:
        """ Called from the pipeline worker thread when a stage or sub-stage starts running. """
        if stage in self._active_stages:
            return
        stage.started = time.monotonic()
        self._active_stages.append(stage)
        self.call_from_thread(self._set_status, stage, "RUNNING 0.0s", "status-running")
        if stage.parent is None:
            self.call_from_thread(self._log, f"\n=== {stage.label} ===", "header")
        else:
            self.call_from_thread(self._log, f"\n--- {stage.label} ---", "header")

    def _track_error(self, line: str, is_error: bool) -> None:
        """ Called from the pipeline worker thread for every output line. An error is its first line plus
            the continuation lines after it (its traceback), every new error updates the alarm. """
        if not is_error:
            self._in_error = False
            return

        if self._in_error and line.startswith(ERROR_CONTINUATION):
            self._errors[-1][1].append(line)
            return

        self._in_error = True
        where = " / ".join(stage.label.strip() for stage in self._active_stages) or "PIPELINE"
        self._errors.append((where, [line]))
        for stage in self._active_stages:
            stage.errors += 1
        self.call_from_thread(self._raise_alarm, len(self._errors))

    def _raise_alarm(self, count: int) -> None:
        self.query_one("#alarm", Static).update(
            f"⚠  {count} ERROR{'S' if count != 1 else ''} IN THIS RUN  ⚠\nfull details in the log and in the summary at the end"
        )
        if not self.screen.has_class("has-errors"):
            self.screen.add_class("has-errors")
            self.notify("The pipeline reported an error, see the log.", title="ERROR", severity="error", timeout=10)

    def _log_error_summary(self, failed: bool) -> None:
        """ Repeats every error of the run in full at the end of the log, so none is lost in the output above. """
        count = len(self._errors)
        self._log(f"\n=== ⚠ {count} ERROR{'S' if count != 1 else ''} ===", "error")
        for number, (where, lines) in enumerate(self._errors, start=1):
            self._log(f"\n--- error {number}/{count} during {where} ---", "error")
            for line in lines:
                self._log(line, "error")
        outcome = "Pipeline FAILED" if failed else "Pipeline finished, but not everything went well"
        self._log(f"\n{outcome}: {count} error{'s' if count != 1 else ''}, listed above.", "error")

    def _tick(self) -> None:
        if self._aborted:
            return
        for stage in list(self._active_stages):
            self._set_status(stage, f"RUNNING {time.monotonic() - stage.started:.1f}s", "status-running")

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
