""" Reads the system clipboard and pastes it into dashboard inputs.

Textual's own Ctrl+V only pastes text copied inside the app, so the system clipboard is read here
through whichever command-line tool the platform provides. """

import os
import shutil
import subprocess
import sys

from textual import events
from textual.binding import Binding
from textual.widgets import Input

RIGHT_MOUSE_BUTTON = 3


def _clipboard_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbpaste"]]
    if sys.platform == "win32":
        return [["powershell", "-NoProfile", "-Command", "Get-Clipboard"]]
    commands = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(["wl-paste", "--no-newline"])
    commands += [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]
    return commands


def read_system_clipboard() -> str | None:
    """ Returns the clipboard text, or None if no clipboard tool is available or readable. """
    for command in _clipboard_commands():
        if shutil.which(command[0]) is None:
            continue
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
        except (subprocess.SubprocessError, OSError):
            continue
        return result.stdout
    return None


def clean_path_text(text: str) -> str:
    """ First line only, without surrounding whitespace or quotes (Windows Explorer copies paths quoted). """
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.strip().strip('"').strip("'")


def paste_from_clipboard(input_widget: Input) -> bool:
    text = read_system_clipboard()
    if text is None:
        input_widget.app.notify(
            "No clipboard tool found. Install xclip or xsel (X11) or wl-clipboard (Wayland).",
            severity="warning",
        )
        return False

    value = clean_path_text(text)
    if not value:
        input_widget.app.notify("The clipboard is empty.", severity="warning")
        return False

    start, end = input_widget.selection
    input_widget.replace(value, start, end)
    return True


class ClipboardInput(Input):
    """ Input whose Ctrl+V pastes from the system clipboard. """

    BINDINGS = [Binding("ctrl+v", "paste_system_clipboard", "Paste", show=False)]

    def action_paste_system_clipboard(self) -> None:
        paste_from_clipboard(self)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == RIGHT_MOUSE_BUTTON:
            event.stop()
            paste_from_clipboard(self)
