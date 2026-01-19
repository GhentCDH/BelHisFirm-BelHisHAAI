import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.reactive import reactive

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

class Haai(Static):
    frame = reactive(0)

    def on_mount(self) -> None:
        self.set_interval(0.5, self.next_frame)

    def next_frame(self) -> None:
        self.frame = (self.frame + 1) % len(SHARK_FRAMES)

    def watch_frame(self, frame: int) -> None:
        self.update(SHARK_FRAMES[str(frame + 1)])

class App(App):
    CSS = """
    Logo {
        color: skyblue;
    }

    Haai {
        color: cornflowerblue;
    }
    """

    def compose(self) -> ComposeResult:
        yield Logo(LOGO)
        yield Haai()
        yield Logo(INFO)


if __name__ == "__main__":
    App().run()
