# ---------------------------------------------------------------------------
# Merge rules for the second-pass (rule-based) line merger.
#
# END_RULES   check the PREVIOUS line: when a pattern matches, the current
#             entry is considered complete and a new one starts.
#
# START_RULES check the CURRENT  line: when a pattern matches, a new entry
#             is started regardless of the previous line.
#
# Rules are evaluated in list order; the first match wins.
# Each rule is a dict with:
#   name        – short identifier (shown in debug output)
#   description – human-readable explanation
#   pattern     – Python regex string applied via re.search (end_rules) or
#                 re.match (start_rules)
# ---------------------------------------------------------------------------

END_RULES = [
    {
        "name": "ends_with_period",
        "description": "A line ending with '.' closes the current entry.",
        "pattern": r"\.$",
    },
]

START_RULES = [
    {
        "name": "starts_with_digit",
        "description": "A line starting with a digit begins a new record (new RecordID).",
        "pattern": r"^\d",
    },
]
