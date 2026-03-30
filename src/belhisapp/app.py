from textual.app import App, ComposeResult
from textual.containers import CenterMiddle

from src.belhisapp.widgets.window import DefaultWindow, WindowContainer, ConfigWindow, UtilitiesWindow, EasterEggWindow
from src.belhisapp.widgets.common import FooterOption, FooterWidget, HeaderItem, HeaderWidget
from src.belhisapp.constants import AppConstants

class BelhisApp(App):

    CSS = AppConstants.CSS

    def __init__(self) -> None:

        self._window_container = None

        # Easter egg activation
        self._easter_egg_code = "88881111"
        self._input_buffer = ""
        self._input_time = 3
        self._input_timer_active = False

        super().__init__()


    def compose(self) -> ComposeResult:

        self._window_container = WindowContainer()

        logo = HeaderItem(AppConstants.LOGO)

        yield HeaderWidget([logo])
        yield CenterMiddle(self._window_container)
        yield FooterWidget(["utils", "config", "quit"])

    def on_mount(self) -> None:

        # Start by displaying default window
        default_window = DefaultWindow()
        self._window_container.set_window(default_window)

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:

        if message.option == "quit":
            self.exit()

        elif message.option == "config":

            config_window = ConfigWindow()
            self._window_container.set_window(config_window)
            config_window.load_json()

        elif message.option == "utils":

            utilities_window = UtilitiesWindow(self._window_container)
            self._window_container.set_window(utilities_window)


    def on_key(self, event) -> None:
        
        # Start a timer to reset the buffer, if there isn't one already
        if not self._input_timer_active:
            self._input_timer_active = True
            self.set_timer(self._input_time, self._reset_input_buffer)

        self._input_buffer += event.key

        if self._input_buffer == self._easter_egg_code:
            easter_egg_window = EasterEggWindow()
            self._window_container.set_window(easter_egg_window)

            self._input_buffer = ""

    def _reset_input_buffer(self) -> None:
        self._input_buffer = ""
        self._input_timer_active = False