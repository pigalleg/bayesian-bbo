"""Add a per-row `weeks` label to each function's all_observations.npz.

Rows matching initial_inputs.npy are week 1. Other rows are matched to the
weekly_updates/week_N batch that holds the same input and output. Dry run by
default; pass --apply to rewrite files (each original is copied first).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from bayesian_bbo import FUNCTION_SPECS
from bayesian_bbo.data import _load_json_values

BACKUP_SUFFIX = ".pre-week-labels.npz"


def batch_rows(root: Path, function_id: int) -> list[tuple[int, np.ndarray, float]]:
    """Return (week, input, output) for this function from each batch, using row order."""
    rows = []
    for batch in sorted((root / "weekly_updates").glob("week_*")):
        if not (batch / "outputs.json").exists():
            continue
        week = int(batch.name.removeprefix("week_"))
        inputs = _load_json_values(batch / "inputs.json")
        outputs = _load_json_values(batch / "outputs.json")
        if len(inputs) >= function_id and len(outputs) >= function_id:
            rows.append(
                (week, np.asarray(inputs[function_id - 1], dtype=float), float(outputs[function_id - 1]))
            )
    return rows


def label_function(root: Path, function_id: int) -> tuple[Path, dict[str, np.ndarray]] | None:
    directory = root / f"function_{function_id}"
    aggregate = directory / "all_observations.npz"
    if not aggregate.exists():
        return None
    with np.load(aggregate, allow_pickle=False) as recorded:
        if "weeks" in recorded.files:
            print(f"function_{function_id}: already labelled, skipped")
            return None
        arrays = {key: recorded[key] for key in recorded.files}
    initial = np.load(directory / "initial_inputs.npy", allow_pickle=False)
    n_initial = len(initial)
    if not np.array_equal(arrays["inputs"][:n_initial], initial):
        raise ValueError(f"function_{function_id}: leading rows do not match initial_inputs.npy")

    batches = batch_rows(root, function_id)
    weeks = np.ones(len(arrays["inputs"]), dtype=int)
    for index in range(n_initial, len(arrays["inputs"])):
        point = arrays["inputs"][index]
        value = arrays["outputs"][index]
        matched = {week for week, x, y in batches if np.array_equal(x, point) and y == value}
        if len(matched) != 1:
            raise ValueError(
                f"function_{function_id} row {index}: matches weeks {sorted(matched)}; expected exactly one"
            )
        weeks[index] = matched.pop()
    arrays["weeks"] = weeks
    return aggregate, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--apply", action="store_true", help="rewrite files instead of printing a dry run")
    args = parser.parse_args()

    results = []
    for function_id in FUNCTION_SPECS:
        result = label_function(args.root, function_id)
        if result is not None:
            results.append((function_id, *result))

    for function_id, aggregate, arrays in results:
        counts = {int(week): int(count) for week, count in zip(*np.unique(arrays["weeks"], return_counts=True))}
        print(f"function_{function_id}: {len(arrays['weeks'])} rows, weeks {counts}")

    if not args.apply:
        print("Dry run only; pass --apply to write.")
        return

    for function_id, aggregate, arrays in results:
        backup = aggregate.with_name(aggregate.stem + BACKUP_SUFFIX)
        if backup.exists():
            raise FileExistsError(f"Backup already exists, refusing to overwrite: {backup}")
        shutil.copy2(aggregate, backup)
        np.savez(aggregate, **arrays)
        print(f"Wrote {aggregate} (original copied to {backup.name})")


if __name__ == "__main__":
    main()
