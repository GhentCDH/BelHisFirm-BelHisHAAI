from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical

from .header_item import HeaderItem

class HeaderWidget(Vertical):
    """ Header widget with customizable header items. """

    def __init__(self, items: list[HeaderItem], **kwargs) -> None:

        super().__init__(**kwargs)
        self.items = items

    def compose(self) -> ComposeResult:

        for i, item in enumerate(self.items):
            yield item