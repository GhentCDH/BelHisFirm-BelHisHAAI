""" Animated banner and boot splash, adapted from BelHisFirm-TaPro's tui.py. """

import asyncio
import textwrap

from rich.text import Text
from terminaltexteffects.utils.graphics import Color as EffectColor
from textual import on
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textualeffects.widgets import EffectLabel, effects

from src.belhisapp.constants import AppConstants
from src.belhisapp.theme import BASE, DARK, LEFT_COLUMN_WIDTH

SUBTITLE = "RECORD PROCESSING PIPELINE"

LOGO_TEXT = textwrap.dedent(AppConstants.LOGO).strip("\n")


def build_banner(total_width: int) -> str:
    """ Plain text banner: the logo inside a double-line frame. EffectLabel pads its width by 2,
        so this is built 2 columns narrower than the panel column. """

    logo_rows = LOGO_TEXT.split("\n")
    logo_width = max(len(row) for row in logo_rows)
    logo_rows = [row.ljust(logo_width) for row in logo_rows]

    width = total_width - 2

    def framed(text: str) -> str:
        return f"║{text.center(width)}║"

    lines = [f"╔{'═' * width}╗"]
    lines += [framed(row) for row in logo_rows]
    lines.append(framed(SUBTITLE))
    lines.append(f"╚{'═' * width}╝")
    return "\n".join(lines)


BANNER = build_banner(LEFT_COLUMN_WIDTH - 2)

HIGHLIGHT_CONFIG = {"final_gradient_stops": (EffectColor(BASE.lstrip("#")),)}
PRINT_CONFIG = {
    "print_speed": 15,
    "print_head_return_speed": 15.0,
    "final_gradient_stops": (EffectColor(BASE.lstrip("#")),),
}


class FixedEffectLabel(EffectLabel):
    """ EffectLabel with two fixes: no per-frame delay (its default is visibly slow), and a canvas
        height that counts every line of a multi-line string, not one fewer. """

    FRAME_DELAY = 0.02

    def __init__(self, text: str, effect: str = "Beams", config: dict | None = None, frame_delay: float | None = None) -> None:
        super().__init__(text, effect, config or {})
        self._source_text = text
        if frame_delay is not None:
            self.FRAME_DELAY = frame_delay
        line_count = len(text.split("\n"))
        self.height = line_count
        self.styles.height = line_count

    async def run_effect(self) -> None:
        effect = effects[self.effect](self._source_text)
        for key in self.config:
            if hasattr(effect.effect_config, key):
                setattr(effect.effect_config, key, self.config[key])

        effect.terminal_config.canvas_width = self.width
        effect.terminal_config.canvas_height = self.height
        effect.terminal_config.ignore_terminal_dimensions = True
        for frame in effect:
            self.text = frame
            self.update(Text.from_ansi(self.text))
            await asyncio.sleep(self.FRAME_DELAY)
        self.post_message(self.EffectFinished(self.effect))


class LoopingEffectLabel(FixedEffectLabel):
    """ Repeats its effect with a pause between sweeps. Looping is driven from EffectFinished,
        since overriding on_mount would race EffectLabel's own on_mount. """

    REPEAT_EVERY = 7.0

    @on(EffectLabel.EffectFinished)
    def _schedule_next_sweep(self, message: EffectLabel.EffectFinished) -> None:
        message.stop()
        self.run_worker(self._restart_after_pause, exclusive=True)

    async def _restart_after_pause(self) -> None:
        await asyncio.sleep(self.REPEAT_EVERY)
        await self.run_effect()


class BootSplashScreen(ModalScreen):
    """ Prints the logo once at startup, then holds it briefly. Skippable with any key or click. """

    CSS = f"""
    BootSplashScreen {{
        background: {DARK};
        align: center middle;
    }}
    """

    BINDINGS = [("escape,enter,space", "skip", "Skip")]

    HOLD_AFTER_FINISHED = 1.0
    SAFETY_TIMEOUT = 10.0
    FRAME_DELAY = 0.006

    def compose(self) -> ComposeResult:
        yield FixedEffectLabel(LOGO_TEXT, effect="Print", config=PRINT_CONFIG, frame_delay=self.FRAME_DELAY)

    def on_mount(self) -> None:
        self.set_timer(self.SAFETY_TIMEOUT, self.action_skip)

    @on(EffectLabel.EffectFinished)
    def _on_finished(self, message: EffectLabel.EffectFinished) -> None:
        self.set_timer(self.HOLD_AFTER_FINISHED, self.action_skip)

    def action_skip(self) -> None:
        self.dismiss()

    def on_click(self) -> None:
        self.dismiss()
