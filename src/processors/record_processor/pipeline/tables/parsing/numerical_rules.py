"""Cross-checks printed share counts against their written form.

Ported from BelHisFirm---HisTableFinder's parsing/numerical_rules.py."""

from __future__ import annotations

import re
from typing import Any

from text_to_num import alpha2digit

from .rule_engine import RuleSet

rules = RuleSet()
apply_rules = rules.apply

# aantal: the written form ("aantal") is the reliable transcription, but the
# printed numeral form ("aantal_numeriek") is frequently wrong for fractions -
# e.g. "quatre cinquiemes" (4/5 = 0.8) gets transcribed as "4" instead of
# "0.8" (the numerator is used verbatim instead of numerator/denominator).
# Re-derive the true value from the written text with text2num and compare;
# only when the two disagree do we add "aantal_numerical_parsed" holding the
# value re-derived from "aantal" - "aantal_numeriek" itself is left untouched
# since the schema defines it as a literal, uncorrected transcription.
_FRACTION_WORDS = {"quart": 4, "quarts": 4, "tiers": 3, "demi": 2, "demie": 2}
_ORDINAL_FRACTION = re.compile(r"(\d+)\s+(\d+)i?èmes?")
_WORD_FRACTION = re.compile(r"(\d+)\s+(" + "|".join(_FRACTION_WORDS) + r")\b")
_NUMBER = re.compile(r"\d+")


def _parse_aantal(aantal: str) -> float | None:
    normalized = re.sub(r"ieme", "ième", aantal, flags=re.IGNORECASE)
    try:
        converted = alpha2digit(normalized, "fr", threshold=0)
    except Exception:
        return None

    fraction: tuple[re.Match[str], int, int] | None = None
    if match := _ORDINAL_FRACTION.search(converted):
        fraction = (match, int(match.group(1)), int(match.group(2)))
    elif match := _WORD_FRACTION.search(converted):
        fraction = (match, int(match.group(1)), _FRACTION_WORDS[match.group(2)])

    if fraction is not None:
        match, numerator, denominator = fraction
        whole = 0
        before_numbers = _NUMBER.findall(converted[: match.start()])
        if before_numbers:
            whole = int(before_numbers[-1])
        return whole + numerator / denominator

    numbers = _NUMBER.findall(converted)
    return float(numbers[0]) if numbers else None


def _as_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.strip().replace(",", "."))
    except ValueError:
        return None


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


@rules.rule("type", path="shareholder_register.entries.*.type")
@rules.rule("stemmen", path="shareholder_register.entries.*.stemmen")
def check_aantal_numeriek(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in value:
        parsed = _parse_aantal(item.get("aantal", ""))
        printed = _as_float(item.get("aantal_numeriek"))
        if parsed is not None and (printed is None or abs(parsed - printed) > 1e-6):
            item["aantal_numerical_parsed"] = _format_number(parsed)
    return value
