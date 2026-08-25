#!/bin/bash
# Finds TIF/<year>/IV* folders for years counting up from START_YEAR to 1950
# (inclusive). Years with no match are silently skipped. When a year has more
# than one IV* folder (e.g. IV, IV.2, IV.3 — some volumes get re-split into a
# higher-numbered subpart), only the highest-numbered one is kept, unless
# --all is passed, in which case every matching folder for that year is printed.
#
# Usage:
#   ./find_iv_folders.sh [--all] [BASE_DIR] [START_YEAR] > paths.txt
#
# Defaults:
#   BASE_DIR    /mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O
#   START_YEAR  1930

set -euo pipefail

ALL=0
args=()
for arg in "$@"; do
    case "$arg" in
        -a|--all) ALL=1 ;;
        *) args+=("$arg") ;;
    esac
done

BASE_DIR="${args[0]:-/mnt/UGent_Share/ghentcdh_belhisfirm/EHC_B665_O}"
START_YEAR="${args[1]:-1923}"
END_YEAR=1923

for (( year=START_YEAR; year<=END_YEAR; year++ )); do
    matches=$(find "$BASE_DIR" -type d -path "*/TIF/${year}/IV*" 2>/dev/null | sort -V)
    [[ -z "$matches" ]] && continue
    if [[ "$ALL" -eq 1 ]]; then
        echo "$matches"
    else
        echo "$matches" | tail -n 1
    fi
done
