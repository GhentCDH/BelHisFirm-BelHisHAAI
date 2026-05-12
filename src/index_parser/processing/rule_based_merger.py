import re
import logging

_log = logging.getLogger("index_parser")

try:
    from .merge_rules import END_RULES, START_RULES
except ImportError:
    from merge_rules import END_RULES, START_RULES


class RuleBasedMerger:
    """
    Second-pass line merger that runs after the DBSCAN-based merge.

    Applies ordered end/start rules (defined in merge_rules.py) to decide
    whether a new OCR line continues the current entry or opens a new one.
    Rules can be changed by editing merge_rules.py without touching this file.
    """

    def __init__(self, rules_path=None):
        if rules_path is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_merge_rules_custom", rules_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            end_defs = getattr(mod, "END_RULES", [])
            start_defs = getattr(mod, "START_RULES", [])
        else:
            end_defs = END_RULES
            start_defs = START_RULES

        self._end_rules = [(r["name"], re.compile(r["pattern"])) for r in end_defs]
        self._start_rules = [(r["name"], re.compile(r["pattern"])) for r in start_defs]

    def process(self, lines: list) -> list:
        """Merge lines into complete entries and return the resulting list."""
        if not lines:
            return []

        entries = []
        current = [lines[0]]

        for line in lines[1:]:
            rule_name = self._new_entry_reason(current[-1], line)
            if rule_name:
                merged = self._combine(current)
                if len(current) > 1:
                    parts_preview = "\n    ".join(f'"{l[:70]}"' for l in current)
                    _log.info(
                        f"[RuleMerger] regel '{rule_name}' — {len(current)} regels samengevoegd:\n"
                        f"    {parts_preview}\n"
                        f"    → \"{merged[:80]}\""
                    )
                entries.append(merged)
                current = [line]
            else:
                current.append(line)

        entries.append(self._combine(current))
        return entries

    def _new_entry_reason(self, last_line: str, current_line: str):
        """Return the matching rule name, or None if lines should be merged."""
        for name, pattern in self._end_rules:
            if pattern.search(last_line):
                return name
        for name, pattern in self._start_rules:
            if pattern.match(current_line):
                return name
        return None

    def _combine(self, lines: list) -> str:
        parts = [lines[0]]
        for line in lines[1:]:
            if parts[-1].endswith("-"):
                parts[-1] = parts[-1][:-1] + line
            else:
                parts.append(line)
        return " ".join(parts)
