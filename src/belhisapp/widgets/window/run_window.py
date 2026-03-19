from textual.widgets import Static

from src.belhisapp.widgets.window import Window

class RunWindow(Window):

    def __init__(self):
        super().__init__([Static("Test")])