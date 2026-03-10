from textual.app import App, ComposeResult
from textual.widgets import Static

from src.belhisapp.widgets import FooterOption
from src.belhisapp.widgets import FooterWidget
from src.belhisapp.widgets import HeaderWidget
from src.belhisapp.widgets import HeaderItem

from src.belhisapp.constants import AppConstants

class BelhisApp(App):
    CSS = AppConstants.CSS

    def compose(self) -> ComposeResult:
        logo = HeaderItem(AppConstants.LOGO)
        info = HeaderItem(AppConstants.INFO)

        yield HeaderWidget([logo])

        yield Static(AppConstants.SHARK)
        yield Static(AppConstants.INFO)

        yield FooterWidget(["utils", "config", "quit"])

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        else:
            self.notify(f"Selected: {message.option}")