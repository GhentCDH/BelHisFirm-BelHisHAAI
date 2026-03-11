from textual.containers import Container
from textual.app import ComposeResult
from textual.widget import Widget

class WindowContainer(Container):
    """ Container object for window widgets. """

    widgets: list[Widget]

    def __init__(self, widgets: list[Widget]) -> None:
        super().__init__()
        self.widgets = widgets

    def compose(self) -> ComposeResult:
        for i, widget in enumerate(self.widgets):
            yield widget