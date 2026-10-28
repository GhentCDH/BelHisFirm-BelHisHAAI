""" Amber palette for the pre-annotation dashboard: a highlighter-marker base, so it reads as draft work
next to the sage production dashboard. Built the same way as src/belhisapp/theme.py. """

from textual.theme import Theme

from src.belhisapp.theme import LEFT_COLUMN_WIDTH, scale

BASE = "#E8B86A"
ACCENT = "#F6DDA8"
DARK = scale(BASE, 0.16)
PANEL_BG = scale(BASE, 0.24)
BORDER = scale(BASE, 0.60)
RUNNING = "#8FC8E8"  # the base is amber already, so a running stage stands out in blue
DONE = "#A8E38F"
SKIPPED = scale(BASE, 0.40)
FAILED = "#F08080"
ABORTED = "#E89AD0"
DIM = scale(BASE, 0.45)

LOG_STYLES = {
    "header": f"bold {ACCENT}",
    "ok": f"bold {DONE}",
    "fail": f"bold {FAILED}",
    "error": f"bold {FAILED}",
    "warn": f"bold {ABORTED}",
    "dim": DIM,
}

AMBER_THEME = Theme(
    name="amber",
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

__all__ = ["ABORTED", "ACCENT", "AMBER_THEME", "BASE", "BORDER", "DARK", "DONE", "FAILED", "LEFT_COLUMN_WIDTH",
           "LOG_STYLES", "PANEL_BG", "RUNNING", "SKIPPED"]
