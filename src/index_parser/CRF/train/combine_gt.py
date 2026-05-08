import argparse
import re
import pandas as pd
from pathlib import Path

_GT_DIR = Path(__file__).parent.parent / "BelHisFirm-GT"
_STEM_RE = re.compile(r"EHC_B665_O_\d{4}_(.+)_(\d{3,})_GT$")


def _parse_stem(stem: str) -> str:
    """Return 'year_page' extracted from the GT stem, or the stem itself as fallback."""
    m = _STEM_RE.match(stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return stem


def combine_gt(paths: list[Path], output_dir: Path) -> Path:
    combined = []
    id_offset = 0

    for path in paths:
        df = pd.read_csv(path)
        df["id"] = df["id"] + id_offset
        id_offset = int(df["id"].max())
        combined.append(df)

    result = pd.concat(combined, ignore_index=True)

    name_parts = [_parse_stem(p.stem) for p in paths]
    output_name = "_".join(name_parts) + "_combined_GT.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    result.to_csv(output_path, index=False)
    print(f"Combined GT saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple GT CSV files into one, re-sequencing entry IDs."
    )
    parser.add_argument("gt_files", nargs="+", type=str, help="Paths to GT CSV files to combine")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {_GT_DIR / 'combined'})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _GT_DIR / "combined"
    combine_gt([Path(p) for p in args.gt_files], out_dir)
