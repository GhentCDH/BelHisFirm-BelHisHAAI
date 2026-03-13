from textual.app import App, ComposeResult
from textual.containers import CenterMiddle

from src.belhisapp.widgets.window import DefaultWindow, WindowContainer, ConfigWindow
from src.belhisapp.widgets.common import FooterOption, FooterWidget, HeaderItem, HeaderWidget
from src.belhisapp.constants import AppConstants

class BelhisApp(App):

    CSS = AppConstants.CSS

    _window_container: WindowContainer
    _default_window: DefaultWindow
    _config_window: ConfigWindow

    def compose(self) -> ComposeResult:

        self._default_window: DefaultWindow = DefaultWindow()
        self._config_window: ConfigWindow = ConfigWindow()
        self._window_container: WindowContainer = WindowContainer()

        logo: HeaderItem = HeaderItem(AppConstants.LOGO)

        yield HeaderWidget([logo])
        yield CenterMiddle(self._window_container)
        yield FooterWidget(["utils", "config", "quit"])

    def on_mount(self) -> None:

        # Start with default window
        self._window_container.set_window(self._default_window)

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        elif message.option == "config":
            self._window_container.set_window(self._config_window)
            self._config_window.load()
        elif message.option == "utils":
            self.notify(f"Selected: {message.option}")