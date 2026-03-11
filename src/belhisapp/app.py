from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.containers import CenterMiddle

from src.belhisapp.widgets import FooterOption
from src.belhisapp.widgets import FooterWidget
from src.belhisapp.widgets import HeaderWidget
from src.belhisapp.widgets import HeaderItem
from src.belhisapp.widgets.window import WindowContainer

from src.belhisapp.constants import AppConstants

class BelhisApp(App):
    CSS = AppConstants.CSS

    def compose(self) -> ComposeResult:
        logo = HeaderItem(AppConstants.LOGO)
        yield HeaderWidget([logo])

        yield CenterMiddle(WindowContainer([Static(AppConstants.SHARK), Static(AppConstants.INFO)]))

        yield FooterWidget(["utils", "config", "quit"])

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        else:
            self.notify(f"Selected: {message.option}")