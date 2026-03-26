from textual.containers import Center, Vertical
from textual.widgets import Button, Static

from src.belhisapp.constants import AppConstants
from src.belhisapp.widgets.window.window_container import WindowContainer
from src.belhisapp.widgets.window.util_run_window import UtilRunWindow
from src.belhisapp.widgets.window import Window

from src.belhisapp.constants import UtilConstants

class UtilitiesWindow(Window):

    def __init__(self, window_container: WindowContainer) -> None:

        # Store a reference to the window container so we can switch windows on button press
        self._window_container = window_container

        self.header = Static(AppConstants.UTIL_LOGO, classes="FormHeader")

        gap = Static("")

        self.buttons: list[Button] = []

        # Build widget list of buttons for the window to display
        for button in UtilConstants.UtilButtons.keys():
            self.buttons.append(button)

        super().__init__([self.header, gap, Center(Vertical(*self.buttons))])

    async def on_button_pressed(self, event: Button.Pressed) -> None:

        # Get the callable method associated with this button
        method = UtilConstants.UtilButtons[event.button]

        label = str(event.button.label)

        self._window_container.set_window(UtilRunWindow(label, method))