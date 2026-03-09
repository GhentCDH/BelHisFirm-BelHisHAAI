from textual.app import App, ComposeResult

from src.belhisapp.widgets import FooterOption
from src.belhisapp.widgets import FooterWidget
from src.belhisapp.widgets import Logo

from src.belhisapp.constants import AppConstants

class BelhisApp(App):

    CSS = AppConstants.CSS

    def compose(self) -> ComposeResult:
        yield Logo(AppConstants.LOGO)
        yield Logo(AppConstants.INFO)
        yield FooterWidget(["utils", "config", "quit"])

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        else:
            self.notify(f"Selected: {message.option}")