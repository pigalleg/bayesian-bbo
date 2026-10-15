"""Convert NumPy-printed weekly .txt files to JSON, one week folder per list.

Each list starts on its own line, written with NumPy's repr: rows as
`array([...])` (inputs) or plain numbers (outputs). A file holds one list per
week, in order; the first list is --first-week (default 2), the next is the
following week, and so on. Each list is written to
`<weeks-dir>/week_<N>/<file stem>.json`. Existing JSON with identical values is
left alone; other existing files need --overwrite. Nothing is written if any
target conflicts.
"""

import argparse
import ast
import json
import re
from pathlib import Path

from bayesian_bbo import FUNCTION_SPECS

ARRAY_CALL = re.compile(r"array\(")
DEFAULT_WEEKS_DIR = Path("data/weekly_updates")


def parse_lists(text: str) -> list[list]:
    """Parse every list in order, where each list starts a line with `[`; no code is executed."""
    chunks = [chunk.strip() for chunk in re.split(r"^(?=\[)", text, flags=re.MULTILINE) if chunk.strip()]
    values = []
    for chunk in chunks:
        value = ast.literal_eval(ARRAY_CALL.sub("(", chunk))
        if not isinstance(value, list):
            raise ValueError("Expected a list of rows or numbers")
        values.append(value)
    return values


def to_floats(value):
    if isinstance(value, list):
        return [to_floats(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number, found {value!r}")
    return float(value)


def check_rows(values: list) -> None:
    if len(values) != len(FUNCTION_SPECS):
        raise ValueError(f"Expected {len(FUNCTION_SPECS)} entries, found {len(values)}")
    for function_id, row in zip(FUNCTION_SPECS, values):
        if isinstance(row, list) and len(row) != FUNCTION_SPECS[function_id]:
            raise ValueError(
                f"Function {function_id} expects {FUNCTION_SPECS[function_id]} coordinates, "
                f"found {len(row)}"
            )


def write_json(path: Path, values: list) -> None:
    content = json.dumps(values, indent=2, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != values:
        raise ValueError(f"Saved values do not match: {path}")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="NumPy-printed .txt files to convert")
    parser.add_argument("--weeks-dir", type=Path, default=DEFAULT_WEEKS_DIR, help="folder holding week_N subfolders")
    parser.add_argument("--first-week", type=int, default=2, help="week number of the first list in each file")
    parser.add_argument("--overwrite", action="store_true", help="replace existing JSON files with different values")
    args = parser.parse_args()
    if args.first_week < 1:
        parser.error("--first-week must be a positive integer")

    conversions = []
    for source in args.paths:
        if source.suffix != ".txt":
            parser.error(f"Expected a .txt file: {source}")
        for offset, raw in enumerate(parse_lists(source.read_text(encoding="utf-8"))):
            week = args.first_week + offset
            values = to_floats(raw)
            check_rows(values)
            target = args.weeks_dir / f"week_{week}" / f"{source.stem}.json"
            if target.exists():
                if json.loads(target.read_text(encoding="utf-8")) == values:
                    print(f"Unchanged {target}")
                    continue
                if not args.overwrite:
                    raise FileExistsError(f"{target} exists with different values; pass --overwrite to replace it")
            conversions.append((target, values))

    for target, values in conversions:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, values)


if __name__ == "__main__":
    main()
