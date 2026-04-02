#!/usr/bin/env python3

"""Convert text content from JSON into a single-line TXT file.

Rules:
- Collapse all extracted text into one line.
- Insert one space between chunks by default.
- Do not insert a space before punctuation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


NO_SPACE_BEFORE_RE = re.compile(r"^[\]\)\}\.,;:!?%]")


def parse_input_path(path_value: str) -> Path:
	"""Accept both normal paths and file:// URIs."""
	if path_value.startswith("file://"):
		parsed = urlparse(path_value)
		return Path(unquote(parsed.path))
	return Path(path_value)


def extract_text_values(data: Any) -> Iterable[str]:
	"""Yield textual leaf values from nested JSON data."""
	if isinstance(data, dict):
		for value in data.values():
			yield from extract_text_values(value)
	elif isinstance(data, list):
		for item in data:
			yield from extract_text_values(item)
	elif isinstance(data, str):
		yield data


def normalize_chunk(text: str) -> str:
	"""Trim and collapse whitespace inside one chunk."""
	return " ".join(text.split())


def merge_chunks(chunks: Iterable[str]) -> str:
	"""Merge chunks into one line with punctuation-aware spacing."""
	merged = ""

	for raw_chunk in chunks:
		chunk = normalize_chunk(raw_chunk)
		if not chunk:
			continue

		if not merged:
			merged = chunk
			continue

		if NO_SPACE_BEFORE_RE.match(chunk):
			merged += chunk
		else:
			merged += f" {chunk}"

	return merged


def convert_json_to_oneline_txt(input_path: Path, output_path: Path, encoding: str) -> None:
	with input_path.open("r", encoding=encoding) as f:
		data = json.load(f)

	line = merge_chunks(extract_text_values(data))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding=encoding) as f:
		f.write(line)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Convert all JSON text values into a one-line TXT file."
	)
	parser.add_argument(
		"input_json",
		help="Input JSON file path or file:// URI.",
	)
	parser.add_argument(
		"output_txt",
		nargs="?",
		help="Output TXT path (default: input name with .txt).",
	)
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="File encoding for input and output (default: utf-8).",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	input_path = parse_input_path(args.input_json)
	output_path = Path(args.output_txt) if args.output_txt else input_path.with_suffix(".txt")

	convert_json_to_oneline_txt(input_path=input_path, output_path=output_path, encoding=args.encoding)
	print(f"Saved single-line text to: {output_path}")


if __name__ == "__main__":
	main()
