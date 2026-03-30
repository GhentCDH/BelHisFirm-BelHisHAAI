from textual.containers import Container
from textual.widgets import Button

from .window import Window

class WindowContainer(Container):
    """ Container which recomposes dynamically to display Window objects"""

    def __init__(self) -> None:
        super().__init__()
        self._window = None

    # Changes the currently displayed window
    def set_window(self, window: Window) -> None:
        """ Displays a window within the container

            Args: window (Window): Window to display
        """

        self._window = window

        # Remove old children
        for child in list(self.children):
            child.remove()

        # Mount all new children in the window
        for widget in self._window.widgets:
            self.mount(widget)

    async def on_button_pressed(self, event: Button.Pressed) -> None:

        # If we have an active window object, manually call its handler for any events
        if self._window and hasattr(self._window, "on_button_pressed"):
            await self._window.on_button_pressed(event)