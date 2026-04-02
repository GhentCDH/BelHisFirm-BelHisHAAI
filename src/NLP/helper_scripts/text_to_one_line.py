#!/usr/bin/env python3

"""Convert text input into a single line with normalized spacing."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def to_one_line(text: str) -> str:
    """Collapse all whitespace runs into single spaces and trim edges."""
    return " ".join(text.split())


def get_next_paragraph_output_path(base_dir: Path) -> Path:
    """Find the next available paragraph_n.txt file name in base_dir."""
    pattern = re.compile(r"^paragraph_(\d+)\.txt$")
    max_n = -1

    for candidate in base_dir.iterdir():
        if not candidate.is_file():
            continue

        match = pattern.match(candidate.name)
        if match:
            max_n = max(max_n, int(match.group(1)))

    return base_dir / f"paragraph_{max_n + 1}.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert input text to a single line."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to convert. If omitted, use --input-file or stdin.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to a text file to read as input.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help=(
            "Optional output .txt file path. If omitted, saves as the next "
            "paragraph_n.txt in the input file folder (or current folder)."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding for reading and writing files (default: utf-8).",
    )
    return parser


def read_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text

    if args.input_file is not None:
        return args.input_file.read_text(encoding=args.encoding)

    return input("Enter text: ")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.text is not None and args.input_file is not None:
        parser.error("Use either text or --input-file, not both.")

    raw_text = read_input_text(args)
    one_line = to_one_line(raw_text)

    output_path: Path
    if args.output_file is not None:
        output_path = args.output_file
    else:
        base_dir = args.input_file.parent if args.input_file is not None else Path.cwd()
        output_path = get_next_paragraph_output_path(base_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(one_line, encoding=args.encoding)
    print(f"Saved one-line text to: {output_path}")


if __name__ == "__main__":
    main()
