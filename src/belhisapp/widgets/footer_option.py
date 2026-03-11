from textual.widgets import Button, Static
from textual.message import Message

class FooterOption(Static):
    """ Clickable static widget with customizable message. """

    class Selected(Message):
        """ Message sent when an option is selected."""
        option: str

        def __init__(self, option: str) -> None:
            self.option = option
            super().__init__()

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self.label = label

    def on_click(self) -> None:
        self.post_message(self.Selected(self.label))