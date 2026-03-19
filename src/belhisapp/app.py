from textual.app import App, ComposeResult
from textual.containers import CenterMiddle

from src.belhisapp.widgets.window import UtilitiesWindow
from src.belhisapp.widgets.window import DefaultWindow, WindowContainer, ConfigWindow
from src.belhisapp.widgets.common import FooterOption, FooterWidget, HeaderItem, HeaderWidget
from src.belhisapp.constants import AppConstants

class BelhisApp(App):

    CSS = AppConstants.CSS

    _window_container: WindowContainer

    def compose(self) -> ComposeResult:

        self._window_container: WindowContainer = WindowContainer()

        logo: HeaderItem = HeaderItem(AppConstants.LOGO)

        yield HeaderWidget([logo])
        yield CenterMiddle(self._window_container)
        yield FooterWidget(["utils", "config", "quit"])

    def on_mount(self) -> None:

        # Start by displaying default window
        default_window: DefaultWindow = DefaultWindow()
        self._window_container.set_window(default_window)

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        elif message.option == "config":
            config_window: ConfigWindow = ConfigWindow()
            self._window_container.set_window(config_window)
            config_window.load_json()
        elif message.option == "utils":
            utilities_window: UtilitiesWindow = UtilitiesWindow()
            self._window_container.set_window(utilities_window)