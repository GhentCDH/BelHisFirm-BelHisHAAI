from textual.reactive import reactive
from textual.widgets import Static

from src.belhisapp.constants import AppConstants
from src.belhisapp.widgets.window.window import Window
from src.belhisapp.constants import EasterEggConstants

class EasterEggWindow(Window):
    """ Easter egg window that displays after the user inputs the correct code. """

    def __init__(self) -> None:
        super().__init__([Haai(id="easter-shark"), Static(AppConstants.INFO, id="easter-text")])

class Haai(Static):
    frame = reactive(0)

    def on_mount(self) -> None:
        self.set_interval(0.5, self.next_frame)

    def next_frame(self) -> None:
        self.frame = (self.frame + 1) % len(EasterEggConstants.SHARK_FRAMES)

    def watch_frame(self, frame: int) -> None:
        self.update(EasterEggConstants.SHARK_FRAMES[str(frame + 1)])