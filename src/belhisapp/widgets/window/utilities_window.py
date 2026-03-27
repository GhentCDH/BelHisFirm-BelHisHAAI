from textual.containers import Center, Vertical
from textual.widgets import Button, Static

from src.belhisapp.constants import AppConstants
from src.belhisapp.widgets.window.window_container import WindowContainer
from src.belhisapp.widgets.window.util_run_window import UtilRunWindow
from src.belhisapp.widgets.window import Window

from src.belhisapp.constants import UtilConstants

class UtilitiesWindow(Window):
    """ Utilities window, renders buttons specified in the UtilConstants and switches to their window based on which one is selected. """

    def __init__(self, window_container: WindowContainer) -> None:

        # Store a reference to the window container so we can switch windows on button press
        self._window_container = window_container

        self.header = Static(AppConstants.UTIL_LOGO, classes="FormHeader")

        # Build widget list of buttons for the window to display
        self.widgets = []

        # Rebuild buttons because this window is recreated after being unmounted
        for util_button in UtilConstants.UtilButtons:

            button = Button(util_button.label, classes="FormButton")
            button.method = util_button.method

            self.widgets.append(Center(button))

            self.widgets.append(Center(Static(""))) # Hacky way to add gap between buttons, padding isn't willing to work right now

        super().__init__([self.header, Static(""), Vertical(*self.widgets)])

    async def on_button_pressed(self, event: Button.Pressed) -> None:

        method = event.button.method  # Pycharm does not like this but it works
        label = str(event.button.label)

        self._window_container.set_window(UtilRunWindow(label, method))
