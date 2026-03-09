import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.reactive import reactive
from textual.containers import Vertical
from textual.message import Message

# Load shark animation frames from JSON
SHARK_FRAMES = json.loads(Path(__file__).parent.joinpath("assets/animated_shark.json").read_text())

LOGO = """
    ██████╗░███████╗██╗░░░░░██╗░░██╗██╗░██████╗██╗░░██╗░█████╗░░█████╗░██╗
    ██╔══██╗██╔════╝██║░░░░░██║░░██║██║██╔════╝██║░░██║██╔══██╗██╔══██╗██║
    ██████╦╝█████╗░░██║░░░░░███████║██║╚█████╗░███████║███████║███████║██║
    ██╔══██╗██╔══╝░░██║░░░░░██╔══██║██║░╚═══██╗██╔══██║██╔══██║██╔══██║██║
    ██████╦╝███████╗███████╗██║░░██║██║██████╔╝██║░░██║██║░░██║██║░░██║██║
    ╚═════╝░╚══════╝╚══════╝╚═╝░░╚═╝╚═╝╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝
"""

INFO = """
                  Welcome to BelHisHAAI V.0.1 Alpha
                  @basvercruysse @vincentducatteeuw 

"""

class Logo(Static):
    pass



class FooterOption(Static):

    class Selected(Message):
        """Message sent when an option is selected."""
        def __init__(self, option: str) -> None:
            self.option = option
            super().__init__()

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self.label = label

    def on_click(self) -> None:
        self.post_message(self.Selected(self.label))


class FooterWidget(Vertical):
    """Footer widget with navigable options."""

    selected_index = reactive(0)

    def __init__(self, options: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.options = options

    def compose(self) -> ComposeResult:
        for i, option in enumerate(self.options):
            yield FooterOption(option, id=f"footer-option-{i}")

    def on_mount(self) -> None:
        self.can_focus = True
        self._update_selection()

    def _update_selection(self) -> None:
        for i, child in enumerate(self.query(FooterOption)):
            if i == self.selected_index:
                child.add_class("selected")
            else:
                child.remove_class("selected")

    def watch_selected_index(self, value: int) -> None:
        self._update_selection()

    def on_key(self, event) -> None:
        if event.key == "up":
            self.selected_index = (self.selected_index - 1) % len(self.options)
            event.stop()
        elif event.key == "down":
            self.selected_index = (self.selected_index + 1) % len(self.options)
            event.stop()
        elif event.key == "enter":
            option = self.options[self.selected_index]
            self.post_message(FooterOption.Selected(option))
            event.stop()


class BelHisApp(App):
    CSS = """
    Logo {
        color: skyblue;
    }

    FooterWidget {
        dock: bottom;
        height: 20%;
        padding: 1;
        border-top: solid $primary;
    }

    FooterOption {
        padding: 0 2;
        width: 100%;
    }

    FooterOption.selected {
        background: lightskyblue;
        color: $text;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield Logo(LOGO)
        yield Logo(INFO)
        yield FooterWidget(["utils", "config", "quit"])

    def on_footer_option_selected(self, message: FooterOption.Selected) -> None:
        if message.option == "quit":
            self.exit()
        else:
            self.notify(f"Selected: {message.option}")


if __name__ == "__main__":
    BelHisApp().run()
