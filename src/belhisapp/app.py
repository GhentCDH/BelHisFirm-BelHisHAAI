from textual.app import App, ComposeResult
from textual.containers import CenterMiddle

from src.belhisapp.widgets.window import DefaultWindow, WindowContainer
from src.belhisapp.widgets import FooterOption, FooterWidget, HeaderItem, HeaderWidget
from src.belhisapp.constants import AppConstants

class BelhisApp(App):
    CSS = AppConstants.CSS

    def compose(self) -> ComposeResult:
        logo = HeaderItem(AppConstants.LOGO)
        yield HeaderWidget([logo])

        window = DefaultWindow()

        yield CenterMiddle(WindowContainer(window))

        yield FooterWidget(["utils", "config", "quit"])

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        else:
            self.notify(f"Selected: {message.option}")