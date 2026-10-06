""" Phosphor palette shared with BelHisFirm-TaPro: a muted sage base with a lighter accent,
the darkest derived tint as the dominant surface. """

from textual.theme import Theme


def scale(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (min(255, max(0, round(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


BASE = "#A8D3C6"
ACCENT = "#D0E7E0"
DARK = scale(BASE, 0.16)
PANEL_BG = scale(BASE, 0.24)
BORDER = scale(BASE, 0.60)
RUNNING = "#E8C97A"
DONE = "#8FE3B0"
SKIPPED = scale(BASE, 0.40)
FAILED = "#E88A8A"
ABORTED = "#E0A96D"
DIM = scale(BASE, 0.45)

LEFT_COLUMN_WIDTH = 76  # wide enough for the 70-column BelHisHAAI logo inside its frame

LOG_STYLES = {
    "header": f"bold {ACCENT}",
    "ok": f"bold {DONE}",
    "fail": f"bold {FAILED}",
    "warn": f"bold {ABORTED}",
    "dim": DIM,
}

PHOSPHOR_THEME = Theme(
    name="phosphor",
    primary=ACCENT,
    secondary=BASE,
    accent=ACCENT,
    foreground=BASE,
    background=DARK,
    surface=DARK,
    panel=PANEL_BG,
    success=DONE,
    warning=ABORTED,
    error=FAILED,
    boost=BORDER,
    dark=True,
)
