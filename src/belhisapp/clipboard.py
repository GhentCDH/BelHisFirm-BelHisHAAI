""" Pastes into dashboard inputs.

There are two ways text gets pasted. The terminal's own paste (Ctrl+Shift+V, or its right-click menu)
arrives as a paste event and works everywhere, including over SSH. Ctrl+V and right-click inside the
app arrive as a plain key or click, so the clipboard is read here through whichever command-line tool
the platform provides. That only works when the app runs on the machine the text was copied on. """

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


def insert_pasted_text(input_widget: Input, text: str) -> bool:
    """ Puts the cleaned text at the cursor, replacing the selection if there is one. """
    value = clean_path_text(text)
    if not value:
        return False

    start, end = input_widget.selection
    input_widget.replace(value, start, end)
    return True


def paste_from_clipboard(input_widget: Input) -> bool:
    text = read_system_clipboard()
    if text is None:
        input_widget.app.notify(
            "Could not read the clipboard from here. Paste with your terminal's shortcut instead (usually Ctrl+Shift+V).",
            severity="warning",
        )
        return False

    if not insert_pasted_text(input_widget, text):
        input_widget.app.notify("The clipboard is empty.", severity="warning")
        return False
    return True


class ClipboardInput(Input):
    """ Input that pastes from the terminal's paste as well as from Ctrl+V and right-click. """

    BINDINGS = [Binding("ctrl+v", "paste_system_clipboard", "Paste", show=False)]

    def action_paste_system_clipboard(self) -> None:
        paste_from_clipboard(self)

    def _on_paste(self, event: events.Paste) -> None:
        # Replaces Input's own paste handling, which would keep the quotes around a copied path
        event.prevent_default()
        event.stop()
        insert_pasted_text(self, event.text)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == RIGHT_MOUSE_BUTTON:
            event.stop()
            paste_from_clipboard(self)
