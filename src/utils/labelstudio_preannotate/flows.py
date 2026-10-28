""" The pre-annotation flows the dashboard can run. Each flow is a standalone CLI module that the dashboard
starts as a subprocess, this file only describes its settings panel and how those become CLI arguments.

To add a flow: subclass PreannotationFlow and add an instance to FLOWS. The dashboard adds the shared
arguments itself (--pages, --tasks-json, --ls-url, --project-id, --commit) and hands the token over as
LS_TOKEN, so the flow's CLI has to accept those. Every flow has its own Label Studio project, whose ID
field the dashboard puts at the top of the flow's panel. Everything tied to the flow's labeling config
(data key, control names) is a setting of the flow itself. """

from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Label, Select

from src.belhisapp.clipboard import ClipboardInput
from src.belhisapp.pipeline_runner import DEFAULT_YOLO_WEIGHTS, REPO_ROOT
from src.utils.labelstudio_preannotate.preannotate_regions import BASE_GLM_CHECKPOINT, DEFAULT_GLM_CHECKPOINT, DEFAULT_LABELS

MODEL_DIRS = ["model", "src/model", "experiments"]


@dataclass(frozen=True)
class FlowStage:
    key: str  # as in the CLI's "[stage:<key>] start|done" markers
    label: str
    commit_only: bool = False  # only runs on SEND, skipped on a dry run


class PreannotationFlow:
    key: str
    title: str
    module: str  # run as `python -m <module>`
    project_env: str  # .env variable with this flow's default Label Studio project ID
    stages: list[FlowStage]

    def compose_settings(self) -> ComposeResult:
        """ The widgets of this flow's settings panel. Their ids start with the flow's key. """
        raise NotImplementedError

    def validate(self, app: App) -> str | None:
        """ What is wrong with the settings, or None when the flow can run. """
        raise NotImplementedError

    def flow_args(self, app: App) -> list[str]:
        """ This flow's own CLI arguments. """
        raise NotImplementedError


def find_yolo_models() -> list[Path]:
    """ The YOLO weights in the repo, relative to it, with the pipeline's default first. """
    default = DEFAULT_YOLO_WEIGHTS.relative_to(REPO_ROOT)
    found = sorted({path.relative_to(REPO_ROOT) for folder in MODEL_DIRS for path in (REPO_ROOT / folder).rglob("*.pt")} - {default})
    return [default, *found]


def find_glm_checkpoints() -> list[tuple[str, str]]:
    """ (prompt, value) options for the GLM-OCR checkpoint: the GLM checkpoints under model/, the pipeline's
        default first, then the base model. """
    found = sorted(path.parent.relative_to(REPO_ROOT) for path in (REPO_ROOT / "model").glob("*/config.json")
                   if "glm" in path.parent.name.lower())
    found.sort(key=lambda path: path != DEFAULT_GLM_CHECKPOINT)
    return [(str(path), str(path)) for path in found] + [("base GLM-OCR (zai-org/GLM-OCR)", BASE_GLM_CHECKPOINT)]


def weights_select(widget_id: str) -> Select:
    models = find_yolo_models()
    return Select([(str(path), str(path)) for path in models], value=str(models[0]), allow_blank=False, id=widget_id)


def _value(app: App, widget_id: str) -> str:
    return app.query_one(f"#{widget_id}", Input).value.strip()


def _int_setting(app: App, widget_id: str, name: str, minimum: int = 0) -> str | None:
    value = _value(app, widget_id)
    if value and not (value.isdigit() and int(value) >= minimum):
        return f"{name} must be a whole number >= {minimum}: {value}"
    return None


def _weights_error(app: App, widget_id: str) -> str | None:
    weights = REPO_ROOT / app.query_one(f"#{widget_id}", Select).value
    return None if weights.exists() else f"YOLO weights not found: {weights}"


def _names_error(app: App, widget_ids: list[str]) -> str | None:
    if not all(_value(app, widget_id) for widget_id in widget_ids):
        return "The data key, from_name and to_name are required."
    return None


def _absolute(path: str) -> str:
    """ The subprocess runs from the repo root, so a path typed relative to here is made absolute. """
    return str(Path(path).resolve())


class YoloLayoutFlow(PreannotationFlow):
    key = "yolo"
    title = "YOLO LAYOUT"
    module = "src.utils.labelstudio_preannotate.preannotate_yolo"
    project_env = "LS_YOLO_PROJECT_ID"
    stages = [
        FlowStage("detect", "1/3  YOLO DETECTION"),
        FlowStage("upload", "2/3  UPLOAD PAGES", commit_only=True),
        FlowStage("import", "3/3  IMPORT TASKS", commit_only=True),
    ]

    def compose_settings(self) -> ComposeResult:
        yield Label("YOLO model")
        yield weights_select("yolo-weights-select")
        yield Label("Max pages (0 = all)")
        yield ClipboardInput(value="20", id="yolo-max-pages-input")
        yield Label("Max total detections (0 = no limit)")
        yield ClipboardInput(value="0", id="yolo-max-detections-input")
        yield Label("Min confidence (0-1)")
        yield ClipboardInput(value="0.25", id="yolo-confidence-input")
        yield Label("Shuffle seed (blank = pages in order)")
        yield ClipboardInput(value="", id="yolo-seed-input")
        yield Label("Task data key (Image value=\"$...\")")
        yield ClipboardInput(value="image", id="yolo-data-key-input")
        yield Label("RectangleLabels name (from_name)")
        yield ClipboardInput(value="label", id="yolo-from-name-input")
        yield Label("Image name (to_name)")
        yield ClipboardInput(value="image", id="yolo-to-name-input")

    def validate(self, app: App) -> str | None:
        if error := _weights_error(app, "yolo-weights-select"):
            return error

        for widget_id, name in [("yolo-max-pages-input", "Max pages"), ("yolo-max-detections-input", "Max total detections"),
                                ("yolo-seed-input", "Shuffle seed")]:
            if error := _int_setting(app, widget_id, name):
                return error

        confidence = _value(app, "yolo-confidence-input")
        try:
            if not 0 <= float(confidence) <= 1:
                raise ValueError
        except ValueError:
            return f"Min confidence must be a number between 0 and 1: {confidence}"

        return _names_error(app, ["yolo-data-key-input", "yolo-from-name-input", "yolo-to-name-input"])

    def flow_args(self, app: App) -> list[str]:
        args = ["--weights", str(app.query_one("#yolo-weights-select", Select).value),
                "--max-pages", _value(app, "yolo-max-pages-input") or "0",
                "--max-detections", _value(app, "yolo-max-detections-input") or "0",
                "--confidence", _value(app, "yolo-confidence-input"),
                "--data-key", _value(app, "yolo-data-key-input"),
                "--from-name", _value(app, "yolo-from-name-input"),
                "--to-name", _value(app, "yolo-to-name-input")]
        if _value(app, "yolo-seed-input"):
            args += ["--seed", _value(app, "yolo-seed-input")]
        return args


class OcrRegionsFlow(PreannotationFlow):
    key = "ocr"
    title = "OCR REGIONS"
    module = "src.utils.labelstudio_preannotate.preannotate_regions"
    project_env = "LS_PROJECT_ID"
    stages = [
        FlowStage("sample", "1/3  SAMPLE + GLM GUESS"),
        FlowStage("upload", "2/3  UPLOAD CROPS", commit_only=True),
        FlowStage("import", "3/3  IMPORT TASKS", commit_only=True),
    ]

    @staticmethod
    def _label_check_id(label: str) -> str:
        return f"ocr-label-{label.lower()}-check"

    def compose_settings(self) -> ComposeResult:
        yield Label("YOLO model")
        yield weights_select("ocr-weights-select")
        yield Label("GLM-OCR checkpoint (gives the text guess)")
        checkpoints = find_glm_checkpoints()
        yield Select(checkpoints, value=checkpoints[0][1], allow_blank=False, id="ocr-glm-select")
        yield Label("Labels to sample")
        for label in DEFAULT_LABELS:
            yield Checkbox(label, value=True, id=self._label_check_id(label), classes="label-check")
        yield Label("Crops per label")
        yield ClipboardInput(value="20", id="ocr-per-label-input")
        yield Label("Max crops of a label from one page")
        yield ClipboardInput(value="2", id="ocr-per-page-input")
        yield Label("Max pages (also when a label is still short)")
        yield ClipboardInput(value="100", id="ocr-max-pages-input")
        yield Label("Shuffle seed")
        yield ClipboardInput(value="0", id="ocr-seed-input")
        yield Label("Crops dir (the sampled crops are saved here)")
        yield ClipboardInput(value="ocr_preanno_crops", id="ocr-crops-dir-input")
        yield Label("Exclude dir (crops of an earlier round, blank = none)")
        yield ClipboardInput(value="", id="ocr-exclude-dir-input")
        yield Label("Task data key (Image value=\"$...\")")
        yield ClipboardInput(value="captioning", id="ocr-data-key-input")
        yield Label("TextArea name (from_name)")
        yield ClipboardInput(value="caption", id="ocr-from-name-input")
        yield Label("Image name (to_name)")
        yield ClipboardInput(value="image", id="ocr-to-name-input")

    def _labels(self, app: App) -> list[str]:
        return [label for label in DEFAULT_LABELS if app.query_one(f"#{self._label_check_id(label)}", Checkbox).value]

    def validate(self, app: App) -> str | None:
        if error := _weights_error(app, "ocr-weights-select"):
            return error
        if not self._labels(app):
            return "Pick at least one label to sample."

        for widget_id, name, minimum in [("ocr-per-label-input", "Crops per label", 1), ("ocr-per-page-input", "Max crops per page", 1),
                                          ("ocr-max-pages-input", "Max pages", 1), ("ocr-seed-input", "Shuffle seed", 0)]:
            if not _value(app, widget_id):
                return f"{name} is required."
            if error := _int_setting(app, widget_id, name, minimum):
                return error

        if not _value(app, "ocr-crops-dir-input"):
            return "A crops dir is required."
        exclude_dir = _value(app, "ocr-exclude-dir-input")
        if exclude_dir and not Path(exclude_dir).is_dir():
            return f"Exclude dir not found: {exclude_dir}"

        return _names_error(app, ["ocr-data-key-input", "ocr-from-name-input", "ocr-to-name-input"])

    def flow_args(self, app: App) -> list[str]:
        args = ["--weights", str(app.query_one("#ocr-weights-select", Select).value),
                "--glm-checkpoint", str(app.query_one("#ocr-glm-select", Select).value),
                "--labels", *self._labels(app),
                "--per-label", _value(app, "ocr-per-label-input"),
                "--per-page", _value(app, "ocr-per-page-input"),
                "--max-pages", _value(app, "ocr-max-pages-input"),
                "--seed", _value(app, "ocr-seed-input"),
                "--crops-dir", _absolute(_value(app, "ocr-crops-dir-input")),
                "--data-key", _value(app, "ocr-data-key-input"),
                "--from-name", _value(app, "ocr-from-name-input"),
                "--to-name", _value(app, "ocr-to-name-input")]
        if _value(app, "ocr-exclude-dir-input"):
            args += ["--exclude-dir", _absolute(_value(app, "ocr-exclude-dir-input"))]
        return args


FLOWS: list[PreannotationFlow] = [YoloLayoutFlow(), OcrRegionsFlow()]
