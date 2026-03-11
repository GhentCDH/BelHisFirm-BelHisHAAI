from textual.containers import Container, Center
from textual.app import ComposeResult

from .window import Window

class WindowContainer(Container):
    """ Container which recomposes dynamically to display Window objects"""

    _window: Window

    def __init__(self, window: Window) -> None:
        super().__init__()
        self._window = window

    def compose(self) -> ComposeResult:

        # Compose all widgets within the window
        for i, widget in enumerate(self._window.widgets):
            yield widget

    # Changes the currently displayed window
    def set_window(self, window: Window) -> None:
        """ Displays a window within the container

            Args: window (Window): Window to display
        """

        self._window = window

        # Remove old children
        for child in self.children:
            self.remove(child)

        # Mount all new children in the window
        for widget in self._window.widgets:
            self.mount(widget)