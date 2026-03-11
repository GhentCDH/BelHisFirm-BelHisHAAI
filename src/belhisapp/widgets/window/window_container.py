from textual.containers import Container, Center
from textual.app import ComposeResult
from textual.widget import Widget

from .window import Window

class WindowContainer(Container):
    """ Container which recomposes dynamically to display Window objects"""

    _window: Window | None

    def __init__(self) -> None:
        super().__init__()
        self._window = None

    # Changes the currently displayed window
    def set_window(self, window: Window) -> None:
        """ Displays a window within the container

            Args: window (Window): Window to display
        """
        if self._window == window:
            return

        self._window = window

        # Remove old children
        for child in list(self.children):
            child.remove()

        # Mount all new children in the window
        for widget in self._window.widgets:
            self.mount(widget)