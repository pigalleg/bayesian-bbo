"""Watch week folders for cumulative evaluator text exports and import results."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from bayesian_bbo import append_weekly_batch
from convert_weekly_txt import check_rows, parse_lists, to_floats, write_json


def weekly_values(path: Path, week: int, *, inputs: bool) -> list:
    batches = parse_lists(path.read_text(encoding="utf-8"))
    if len(batches) != week:
        raise ValueError(f"{path}: expected {week} cumulative weekly lists, found {len(batches)}")
    values = to_floats(batches[-1])
    check_rows(values)
    if inputs:
        if not all(isinstance(row, list) for row in values):
            raise ValueError(f"{path}: inputs must be coordinate lists")
    elif any(isinstance(row, list) for row in values):
        raise ValueError(f"{path}: outputs must be scalars")
    return values


def import_week(root: Path, folder: Path, week: int) -> None:
    inputs = weekly_values(folder / "inputs.txt", week, inputs=True)
    outputs = weekly_values(folder / "outputs.txt", week, inputs=False)
    inputs_path = folder / "inputs.json"
    outputs_path = folder / "outputs.json"
    predictions_path = folder / "predictions.json"

    if json.loads(inputs_path.read_text(encoding="utf-8")) != inputs:
        raise ValueError(f"{inputs_path} differs from the submitted text inputs")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing pre-evaluation predictions: {predictions_path}")
    if outputs_path.exists():
        if json.loads(outputs_path.read_text(encoding="utf-8")) != outputs:
            raise FileExistsError(f"{outputs_path} exists with different values")
    else:
        write_json(outputs_path, outputs)

    append_weekly_batch(root, inputs_path, outputs_path, predictions_path)
    print(f"Week {week}: your weekly observations have been incorporated in the main database.", flush=True)


def snapshot(folder: Path) -> tuple:
    files = ("inputs.txt", "outputs.txt", "inputs.json", "outputs.json", "predictions.json")
    states = []
    for name in files:
        try:
            stat = (folder / name).stat()
        except FileNotFoundError:
            states.append(None)
        else:
            states.append((stat.st_mtime_ns, stat.st_size))
    return tuple(states)


def watch(root: Path, interval: float) -> None:
    updates = root / "weekly_updates"
    if not updates.is_dir():
        raise FileNotFoundError(f"Weekly updates directory not found: {updates}")
    previous = {}
    processed = {}
    print(f"Watching {updates}; press Ctrl+C to stop.", flush=True)
    while True:
        current = {}
        for folder in sorted(updates.iterdir()):
            match = re.fullmatch(r"week_([1-9]\d*)", folder.name)
            if not match or not folder.is_dir():
                continue
            state = snapshot(folder)
            current[folder] = state
            if state[0] is None or state[1] is None:
                continue
            if previous.get(folder) != state or processed.get(folder) == state:
                continue
            try:
                import_week(root, folder, int(match.group(1)))
            except (OSError, ValueError, SyntaxError, TypeError) as error:
                print(f"Week {match.group(1)} import failed: {error}", file=sys.stderr, flush=True)
            after = snapshot(folder)
            if all(state[index] == after[index] for index in (0, 1, 2, 4)):
                processed[folder] = after
                current[folder] = after
        previous = current
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="data directory")
    parser.add_argument("--interval", type=float, default=2.0, help="polling interval in seconds")
    args = parser.parse_args()
    if not 0 < args.interval < float("inf"):
        parser.error("--interval must be a positive finite number")
    try:
        watch(args.data_dir, args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.", flush=True)


if __name__ == "__main__":
    main()
