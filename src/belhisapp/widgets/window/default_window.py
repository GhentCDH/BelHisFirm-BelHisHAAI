from textual.widgets import Static
from src.belhisapp.constants import AppConstants
from src.belhisapp.widgets.window import Window

class DefaultWindow(Window):
    """ Window object containing widgets displayed at app creation """

    def __init__(self) -> None:
        shark = Static(AppConstants.SHARK)
        info = Static(AppConstants.INFO, id="default-window-text")

        super().__init__([shark, info])