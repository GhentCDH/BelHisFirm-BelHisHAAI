""" Pre-annotation dashboard: sends pre-annotated pages to Label Studio. Every pre-annotation flow (see flows.py)
is a page of its own, run as a subprocess whose log streams into the dashboard, like the BelHisHAAI dashboard. """

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, ContentSwitcher, Footer, Input, Label, RichLog, Static

from src.belhisapp.clipboard import ClipboardInput
from src.belhisapp.pipeline_runner import REPO_ROOT, classify_line, describe_exit, format_duration, parse_stage_marker
from src.utils.labelstudio_preannotate.client import LabelStudioClient
from src.utils.labelstudio_preannotate.flows import FLOWS, FlowStage, PreannotationFlow
from src.utils.labelstudio_preannotate.theme import (
    ABORTED, ACCENT, AMBER_THEME, BASE, BORDER, DARK, DONE, FAILED, LEFT_COLUMN_WIDTH, LOG_STYLES, PANEL_BG, RUNNING,
    SKIPPED,
)

LS_URL_DEFAULT = "http://localhost:8080"


class PreannotateApp(App):
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

    #title {{
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: {ACCENT};
        border: double {ACCENT};
        margin-bottom: 1;
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

    ContentSwitcher, .project-field {{
        height: auto;
    }}

    .flow-row {{
        height: auto;
    }}

    .flow-btn {{
        margin-right: 1;
    }}

    .flow-btn.-selected {{
        background: {BASE};
        color: {DARK};
        text-style: bold;
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

    .label-check:last-of-type {{
        margin-bottom: 1;
    }}

    Select > SelectCurrent {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Select:focus > SelectCurrent {{
        border: solid {ACCENT};
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

    #dry-run-btn {{
        border: solid {RUNNING};
        color: {RUNNING};
    }}

    #send-btn {{
        border: solid {DONE};
        color: {DONE};
    }}

    #abort-btn {{
        border: solid {FAILED};
        color: {FAILED};
    }}

    #test-ls-btn {{
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

    /* A failed run turns the frames red until the next run */
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

    BINDINGS = [Binding("q", "quit", "Quit")] + [
        Binding(str(number), f"switch_flow('{flow.key}')", flow.title) for number, flow in enumerate(FLOWS, start=1)
    ]
    TITLE = "BelHisHAAI pre-annotation"

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(AMBER_THEME)
        self.theme = "amber"
        self.flows: dict[str, PreannotationFlow] = {flow.key: flow for flow in FLOWS}
        self.flow = FLOWS[0]
        self._statuses: dict[tuple[str, str], Label] = {}  # (flow key, stage key) -> its status label
        self._process: subprocess.Popen | None = None
        self._aborted = False
        self._elapsed_timer: Timer | None = None
        self._active_stage: FlowStage | None = None
        self._stage_started = 0.0

    # -- layout --

    def compose(self) -> ComposeResult:
        load_dotenv(REPO_ROOT / ".env")

        with Horizontal(id="body"):
            with Vertical(id="left-column"):
                yield Static("LABELSTUDIO PRE-ANNOTATION", id="title")
                with VerticalScroll(id="controls-scroll"):
                    with Vertical(classes="panel"):
                        yield Label("FLOW", classes="panel-title")
                        with Horizontal(classes="flow-row"):
                            for number, flow in enumerate(FLOWS, start=1):
                                yield Button(f"{number} {flow.title}", id=f"flow-{flow.key}-btn", classes="flow-btn")

                    with Vertical(classes="panel"):
                        yield Label("PATHS", classes="panel-title")
                        yield Label("PAGES (input dir)")
                        yield ClipboardInput(placeholder="input/staatsblad_1923", id="pages-input")
                        yield Label("TASKS JSON (written on every run)")
                        yield ClipboardInput(value="labelstudio_tasks.json", id="tasks-json-input")

                    with Vertical(classes="panel"):
                        yield Label("LABEL STUDIO (defaults from .env)", classes="panel-title")
                        yield Label("URL")
                        yield ClipboardInput(value=os.environ.get("LS_URL") or LS_URL_DEFAULT, id="ls-url-input")
                        yield Label("API token")
                        yield ClipboardInput(value=os.environ.get("LS_TOKEN", ""), password=True, id="ls-token-input")
                        # Every flow imports into a project of its own, with a labeling config to match
                        with ContentSwitcher(initial=f"flow-{self.flow.key}-project", id="project-switcher"):
                            for flow in FLOWS:
                                with Vertical(id=f"flow-{flow.key}-project", classes="project-field"):
                                    yield Label(f"Project ID for {flow.title} (env: {flow.project_env})")
                                    yield ClipboardInput(value=os.environ.get(flow.project_env, ""), id=f"{flow.key}-project-input")
                        yield Button("TEST CONNECTION", id="test-ls-btn")

                    with ContentSwitcher(initial=f"flow-{self.flow.key}-settings", id="settings-switcher"):
                        for flow in FLOWS:
                            with Vertical(classes="panel", id=f"flow-{flow.key}-settings"):
                                yield Label(f"{flow.title} SETTINGS", classes="panel-title")
                                yield from flow.compose_settings()

                    with ContentSwitcher(initial=f"flow-{self.flow.key}-stages", id="stages-switcher"):
                        for flow in FLOWS:
                            with Vertical(classes="panel", id=f"flow-{flow.key}-stages"):
                                yield Label("STAGES", classes="panel-title")
                                for stage in flow.stages:
                                    with Horizontal(classes="stage-row"):
                                        yield Label(stage.label, classes="stage-label")
                                        status = Label("PENDING", classes="stage-status status-pending")
                                        self._statuses[flow.key, stage.key] = status
                                        yield status

            with Vertical(id="log-column"):
                yield Static("", id="alarm")
                yield RichLog(id="log", wrap=True, highlight=False, markup=False)
                with Horizontal(classes="button-row"):
                    yield Button("DRY RUN", id="dry-run-btn")
                    yield Button("SEND TO LABEL STUDIO", id="send-btn", variant="success")
                    yield Button("ABORT", id="abort-btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._select_flow(self.flow)
        self._log("=== SYSTEM CHECK ===", "header")
        self._log_check(".env found", (REPO_ROOT / ".env").exists())
        for name in ("LS_URL", "LS_TOKEN"):
            self._log_check(f"{name} set", bool(os.environ.get(name)))
        # A flow without a project can still DRY RUN, so that is only a warning
        for flow in FLOWS:
            if os.environ.get(flow.project_env):
                self._log_check(f"{flow.project_env} set ({flow.title} project)", True)
            else:
                self._log(f"[ WARN ]  {flow.project_env} not set, fill in the {flow.title} project ID to SEND", "warn")
        # Tested for the first flow with a project, the connection and token are the same for every flow
        tested = next((flow for flow in FLOWS if os.environ.get(flow.project_env)), None)
        if os.environ.get("LS_TOKEN") and tested is not None:
            self.run_worker(lambda: self._test_ls_connection(tested), thread=True, group="ls-test")
        self._log("")
        self._log("Ready. Pick a FLOW, set PAGES, then DRY RUN to inspect or SEND to import.")

    def _log(self, text: str, style: str | None = None) -> None:
        log = self.query_one("#log", RichLog)
        log.write(Text(text, style=LOG_STYLES.get(style, style)) if style else text)

    def _log_check(self, name: str, ok: bool) -> None:
        self._log(f"[ {'OK' if ok else 'FAIL'} ]  {name}", "ok" if ok else "fail")

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _project_id(self, flow: PreannotationFlow) -> str:
        return self._value(f"{flow.key}-project-input")

    # -- flows --

    def action_switch_flow(self, key: str) -> None:
        if self._process is None:
            self._select_flow(self.flows[key])

    def _select_flow(self, flow: PreannotationFlow) -> None:
        self.flow = flow
        self.query_one("#project-switcher", ContentSwitcher).current = f"flow-{flow.key}-project"
        self.query_one("#settings-switcher", ContentSwitcher).current = f"flow-{flow.key}-settings"
        self.query_one("#stages-switcher", ContentSwitcher).current = f"flow-{flow.key}-stages"
        for other in FLOWS:
            self.query_one(f"#flow-{other.key}-btn", Button).set_class(other is flow, "-selected")

    # -- buttons --

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "dry-run-btn":
            self._start_run(commit=False)
        elif button_id == "send-btn":
            self._start_run(commit=True)
        elif button_id == "abort-btn":
            self._abort_run()
        elif button_id == "test-ls-btn":
            flow = self.flow
            self.run_worker(lambda: self._test_ls_connection(flow), thread=True, group="ls-test")
        elif button_id.startswith("flow-"):
            self.action_switch_flow(button_id.removeprefix("flow-").removesuffix("-btn"))

    def _test_ls_connection(self, flow: PreannotationFlow) -> None:
        url, token, project = self._value("ls-url-input"), self._value("ls-token-input"), self._project_id(flow)
        self.call_from_thread(self._log, f"\nTesting Label Studio connection at {url} for {flow.title} ...")
        if not token or not project.isdigit():
            self.call_from_thread(self._log, "[ FAIL ]  An API token and a numeric project ID are needed to test.", "fail")
            return
        for result in LabelStudioClient(url, token, int(project)).health_check():
            self.call_from_thread(self._log_check, f"{result.label}: {result.detail}", result.ok)

    # -- run / abort --

    def _set_status(self, stage: FlowStage, text: str, css_class: str) -> None:
        status = self._statuses[self.flow.key, stage.key]
        status.set_classes(f"stage-status {css_class}")
        status.update(text)

    def _start_run(self, commit: bool) -> None:
        pages = self._value("pages-input")
        tasks_json = self._value("tasks-json-input")
        project = self._project_id(self.flow)

        if not pages:
            return self._log("[ FAIL ] PAGES directory is required.", "fail")
        if not Path(pages).is_dir():
            return self._log(f"[ FAIL ] PAGES directory not found: {pages}", "fail")
        if not tasks_json:
            return self._log("[ FAIL ] TASKS JSON path is required.", "fail")
        if project and not project.isdigit():
            return self._log(f"[ FAIL ] Project ID must be a number: {project}", "fail")
        if commit and not (project and self._value("ls-token-input")):
            return self._log("[ FAIL ] SEND needs a Label Studio API token and project ID.", "fail")
        if error := self.flow.validate(self):
            return self._log(f"[ FAIL ] {error}", "fail")

        # The subprocess runs from the repo root, so paths typed relative to here are made absolute
        command = [sys.executable, "-m", self.flow.module, "--pages", str(Path(pages).resolve()),
                   "--tasks-json", str(Path(tasks_json).resolve()), "--ls-url", self._value("ls-url-input")]
        if project:
            command += ["--project-id", project]
        command += self.flow.flow_args(self)
        if commit:
            command.append("--commit")

        for stage in self.flow.stages:
            if stage.commit_only and not commit:
                self._set_status(stage, "SKIPPED", "status-skipped")
            else:
                self._set_status(stage, "PENDING", "status-pending")

        self._aborted = False
        self.screen.remove_class("has-errors")
        self._set_running(True)
        self._log(f"\n=== {self.flow.title} {'SEND' if commit else 'DRY RUN'} ===", "header")
        self._log(f"$ {' '.join(command)}", "dim")
        self.run_worker(lambda: self._run_worker(command), thread=True, exclusive=True, group="run")

    def _set_running(self, running: bool) -> None:
        self.query_one("#dry-run-btn", Button).disabled = running
        self.query_one("#send-btn", Button).disabled = running
        self.query_one("#abort-btn", Button).disabled = not running
        for flow in FLOWS:
            self.query_one(f"#flow-{flow.key}-btn", Button).disabled = running

    def _abort_run(self) -> None:
        self._aborted = True
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._active_stage is not None:
            self._set_status(self._active_stage, "ABORTED", "status-aborted")

    def _run_worker(self, command: list[str]) -> None:
        run_start = time.monotonic()
        elapsed_by_stage: dict[str, float] = {}
        stage_by_key = {stage.key: stage for stage in self.flow.stages}
        returncode = None

        try:
            self._elapsed_timer = self.call_from_thread(self.set_interval, 1.0, self._tick)
            # Child stdout is a pipe, so Python would block-buffer it without this flag. The token goes
            # through the environment so it never shows up in the logged command.
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "LS_TOKEN": self._value("ls-token-input")}
            self._process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in self._process.stdout:
                marker = parse_stage_marker(line)
                if marker is None or marker[0] not in stage_by_key:
                    self.call_from_thread(self._log, line.rstrip(), classify_line(line))
                    continue

                stage, event = stage_by_key[marker[0]], marker[1]
                if event == "start":
                    self._active_stage, self._stage_started = stage, time.monotonic()
                    self.call_from_thread(self._set_status, stage, "RUNNING 0.0s", "status-running")
                    self.call_from_thread(self._log, f"\n--- {stage.label} ---", "header")
                elif not self._aborted and stage is self._active_stage:
                    elapsed_by_stage[stage.key] = time.monotonic() - self._stage_started
                    self._active_stage = None
                    self.call_from_thread(self._set_status, stage, f"DONE {format_duration(elapsed_by_stage[stage.key])}", "status-done")
            returncode = self._process.wait()
        except Exception as e:
            self.call_from_thread(self._log, f"[ FAIL ] {type(e).__name__}: {e}", "fail")
        finally:
            if self._elapsed_timer is not None:
                self.call_from_thread(self._elapsed_timer.stop)
                self._elapsed_timer = None

        if self._aborted:
            self.call_from_thread(self._log, "\n=== RUN ABORTED ===", "warn")
        elif returncode != 0:
            if returncode is not None:
                self.call_from_thread(self._log, f"[ FAIL ] {describe_exit(returncode)}", "fail")
            if self._active_stage is not None:
                self.call_from_thread(self._set_status, self._active_stage, "FAILED", "status-failed")
            self.call_from_thread(self._raise_alarm)
        else:
            self.call_from_thread(self._log, "\n=== TIMELINE ===", "header")
            for stage in self.flow.stages:
                if stage.key in elapsed_by_stage:
                    self.call_from_thread(self._log, f"{stage.label:<28} {format_duration(elapsed_by_stage[stage.key]):>10}")
            self.call_from_thread(self._log, f"{'TOTAL':<28} {format_duration(time.monotonic() - run_start):>10}", "header")

        self._active_stage = None
        self._process = None
        self.call_from_thread(self._set_running, False)

    def _raise_alarm(self) -> None:
        self.query_one("#alarm", Static).update("⚠  THE RUN FAILED  ⚠\nfull details in the log")
        self.screen.add_class("has-errors")
        self.notify("The pre-annotation run failed, see the log.", title="ERROR", severity="error", timeout=10)

    def _tick(self) -> None:
        if self._active_stage is not None and not self._aborted:
            self._set_status(self._active_stage, f"RUNNING {time.monotonic() - self._stage_started:.1f}s", "status-running")

    def action_quit(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self.exit()


def main() -> None:
    PreannotateApp().run()
