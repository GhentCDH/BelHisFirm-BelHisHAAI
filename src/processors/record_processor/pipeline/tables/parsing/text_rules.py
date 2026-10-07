"""Gender and address rules for shareholder-register entries.

Ported from BelHisFirm---HisTableFinder's parsing/text_rules.py. The geoparser
is only built on the first address (it loads spaCy and a sentence-transformer
model), and an address that was already parsed is left as it is, so the rules
can be applied to the same table more than once.

Address parsing needs the geonames gazetteer installed once, machine-wide:
    uv run python -m geoparser install geonames
Without it, addresses are left as plain text and the other rules still apply.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any

from .rule_engine import RuleSet

logger = getLogger(__name__)

rules = RuleSet()
apply_rules = rules.apply

GAZETTEER_NAME = "geonames"

_geoparser = None
_geoparser_failed = False


def _get_geoparser():
    global _geoparser, _geoparser_failed
    if _geoparser is None and not _geoparser_failed:
        try:
            from geoparser import Geoparser
            from geoparser.modules import SentenceTransformerResolver, SpacyRecognizer

            _geoparser = Geoparser(
                recognizer=SpacyRecognizer(),
                resolver=SentenceTransformerResolver(gazetteer_name=GAZETTEER_NAME),
            )
        except Exception as e:
            _geoparser_failed = True
            logger.warning(f"Address parsing unavailable, addresses are left as plain text: {type(e).__name__}: {e}")
    return _geoparser


# gender
M_PREFIX = ("M.", "Mr", "De heer")
V_PREFIX = ("Juffr", "Mme", "Mlle", "Me", "Mw.")


def _gender_for_name(name: str) -> str | None:
    if name.startswith(M_PREFIX):
        return "M"
    if name.startswith(V_PREFIX):
        return "V"
    return None


def _parse_address(value: Any) -> Any:
    if not value or not isinstance(value, str):
        return value
    geoparser = _get_geoparser()
    if geoparser is None:
        return value
    document = geoparser.parse(value.replace("à", ""))
    for toponym in document.toponyms:
        location = toponym.location
        if location is None:
            continue
        return {
            "text": value,
            "city": location.data.get("name"),
            "country": location.data.get("country_name"),
            "latitude": location.data.get("latitude"),
            "longitude": location.data.get("longitude"),
        }
    return value


@rules.rule("aandeelhouder", path="shareholder_register.entries.*.aandeelhouder")
def determine_gender_aandeelhouder(value: dict[str, Any]) -> dict[str, Any]:
    if value["type"] == "persoon":
        value["gender"] = _gender_for_name(value["aandeelhouder"])
    else:
        value["gender"] = None
    return value


@rules.rule("vertegenwoordiger", path="shareholder_register.entries.*.vertegenwoordiger")
def determine_gender_vertegenwoordiger(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for representative in value:
        representative["gender"] = _gender_for_name(representative["vertegenwoordiger"])
    return value


@rules.rule("adres", path="shareholder_register.entries.*.adres")
def determine_address_aandeelhouder(value: str | None) -> dict[str, Any] | str | None:
    return _parse_address(value)


@rules.rule("adres", path="shareholder_register.entries.*.vertegenwoordiger.*.adres")
def determine_address_vertegenwoordiger(value: str | None) -> dict[str, Any] | str | None:
    return _parse_address(value)
