from textual.widgets import Static
from src.belhisapp.constants import DefaultWindowConstants
from src.belhisapp.widgets.window import Window

class DefaultWindow(Window):
    """ Window object containing widgets displayed at app creation"""

    def __init__(self) -> None:
        shark: Static = Static(DefaultWindowConstants.SHARK)
        info: Static = Static(DefaultWindowConstants.INFO, id="default-window-text")

        super().__init__([shark, info])