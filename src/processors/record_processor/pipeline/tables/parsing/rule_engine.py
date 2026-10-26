"""Path-scoped rule engine for rewriting transcribed table JSON, shared by
text_rules.py and numerical_rules.py.

Ported from BelHisFirm---HisTableFinder's parsing/rule_engine.py, without its
--source/--output CLI (TablePipeline walks the record folders here)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

RuleFunc = Callable[..., Any]


def _path_str(path: tuple[Any, ...]) -> str:
    return ".".join("*" if isinstance(p, int) else str(p) for p in path)


def _path_matches(path: tuple[Any, ...], pattern: str) -> bool:
    segments = pattern.split(".")
    if len(segments) != len(path):
        return False
    return all(segment == "*" or segment == str(p) for segment, p in zip(segments, path))


def _call_rule(fn: RuleFunc, value: Any, obj: dict[str, Any], path: str) -> Any:
    arity = len(inspect.signature(fn).parameters)
    return fn(*(value, obj, path)[:arity])


class RuleSet:
    """A registry of `@ruleset.rule(key, path=...)`-decorated functions, and
    the machinery to apply them all in one pass over a table's JSON."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, str | None, RuleFunc]] = []

    def rule(self, key: str, path: str | None = None) -> Callable[[RuleFunc], RuleFunc]:
        """Register `fn` to replace the value of every `key` occurrence found
        while walking a table's JSON, optionally scoped to a dotted path glob
        (list indices stand in as "*", e.g. "entries.*.adres")."""

        def decorator(fn: RuleFunc) -> RuleFunc:
            self._rules.append((key, path, fn))
            return fn

        return decorator

    def apply(self, table: dict[str, Any]) -> dict[str, Any]:
        """Apply every registered rule to a table's parsed JSON, in place,
        and return it."""
        return self._walk(table)

    def _walk(self, node: Any, path: tuple[Any, ...] = ()) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = path + (key,)
                value = self._walk(value, child_path)
                for rule_key, rule_path, fn in self._rules:
                    if rule_key != key:
                        continue
                    if rule_path is not None and not _path_matches(child_path, rule_path):
                        continue
                    value = _call_rule(fn, value, node, _path_str(child_path))
                node[key] = value
            return node
        if isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = self._walk(item, path + (i,))
            return node
        return node
