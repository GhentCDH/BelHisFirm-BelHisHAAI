from textual.containers import Horizontal
from textual.reactive import reactive
from textual.app import ComposeResult

from src.belhisapp.widgets.footer_option import FooterOption

class FooterWidget(Horizontal):
    """Footer widget with navigable options."""

    selected_index = reactive(0)

    def __init__(self, options: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.options = options

    def compose(self) -> ComposeResult:
        num_options = len(self.options)
        for i, option in enumerate(self.options):
            btn = FooterOption(option)

            # Decide button width automatically
            btn.styles.width = int(100 / num_options)
            yield btn

    def on_mount(self) -> None:
        self.can_focus = True
        self._update_selection()

    def _update_selection(self) -> None:
        for i, child in enumerate(self.query(FooterOption)):
            if i == self.selected_index:
                child.add_class("hover")
            else:
                child.remove_class("hover")

    def watch_selected_index(self, value: int) -> None:
        self._update_selection()

    def on_key(self, event) -> None:

        # Remove selected class from all buttons
        for i, child in enumerate(self.query(FooterOption)):
            child.remove_class("selected")

        if event.key == "left":
            self.selected_index = (self.selected_index - 1) % len(self.options)
            event.stop()
        elif event.key == "right":
            self.selected_index = (self.selected_index + 1) % len(self.options)
            event.stop()
        elif event.key == "enter":

            # Add selected class to selected button
            for i, child in enumerate(self.query(FooterOption)):
                if i == self.selected_index:
                    child.add_class("selected")

            option = self.options[self.selected_index]
            self.post_message(FooterOption.Selected(option))
            event.stop()