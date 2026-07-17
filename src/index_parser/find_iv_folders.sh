#!/bin/bash
# Finds TIF/<year>/IV* folders for years counting up from START_YEAR to 1950
# (inclusive). Years with no match are silently skipped. When a year has more
# than one IV* folder (e.g. IV, IV.2, IV.3 — some volumes get re-split into a
# higher-numbered subpart), only the highest-numbered one is kept.
#
# Usage:
#   ./find_iv_folders.sh [BASE_DIR] [START_YEAR] > paths.txt
#
# Defaults:
#   BASE_DIR    /mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O
#   START_YEAR  1930

set -euo pipefail

BASE_DIR="${1:-/mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O}"
START_YEAR="${2:-1930}"
END_YEAR=1950

for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    matches=$(find "$BASE_DIR" -type d -path "*/TIF/${year}/IV*" 2>/dev/null | sort -V)
    [[ -z "$matches" ]] && continue
    echo "$matches" | tail -n 1
done
