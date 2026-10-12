"""Read original and cumulative observations without changing the original files."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np

from .submission import format_submission


FUNCTION_SPECS = {
    1: 2,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 6,
    8: 8,
}


def _checked_week(week: object) -> int:
    if isinstance(week, bool) or not isinstance(week, (int, np.integer)) or week < 1:
        raise ValueError("Week must be a positive integer")
    return int(week)


@dataclass(frozen=True)
class Observations:
    inputs: np.ndarray
    outputs: np.ndarray
    dimension: int
    expected_outputs: np.ndarray | None = None
    stds: np.ndarray | None = None
    weeks: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.inputs, dtype=float)
        y = np.asarray(self.outputs, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not len(x):
            raise ValueError("Inputs must be a nonempty (n, dimension) array")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("Outputs must contain one scalar per input")
        if not np.all(np.isfinite(x)) or np.any((x < 0) | (x >= 1)):
            raise ValueError("Inputs must be finite and within [0, 1)")
        if not np.all(np.isfinite(y)):
            raise ValueError("Outputs must be finite")
        expected = (
            np.full(len(x), np.nan)
            if self.expected_outputs is None
            else np.asarray(self.expected_outputs, dtype=float)
        )
        if expected.ndim == 2 and expected.shape[1] == 1:
            expected = expected[:, 0]
        if expected.ndim != 1 or len(expected) != len(x):
            raise ValueError("Expected outputs must contain one value per input")
        if not np.all(np.isfinite(expected) | np.isnan(expected)):
            raise ValueError("Expected outputs must be finite or missing")
        stds = (
            np.full(len(x), np.nan)
            if self.stds is None
            else np.asarray(self.stds, dtype=float)
        )
        if stds.ndim == 2 and stds.shape[1] == 1:
            stds = stds[:, 0]
        if stds.ndim != 1 or len(stds) != len(x):
            raise ValueError("Standard deviations must contain one value per input")
        if not np.all((np.isfinite(stds) & (stds >= 0)) | np.isnan(stds)):
            raise ValueError("Standard deviations must be finite and nonnegative or missing")
        object.__setattr__(self, "inputs", x.copy())
        object.__setattr__(self, "outputs", y.copy())
        object.__setattr__(self, "expected_outputs", expected.copy())
        object.__setattr__(self, "stds", stds.copy())
        weeks = np.ones(len(x), dtype=int) if self.weeks is None else np.asarray(self.weeks)
        if weeks.ndim != 1 or len(weeks) != len(x) or not np.issubdtype(weeks.dtype, np.integer) or np.any(weeks < 1):
            raise ValueError("Weeks must contain one positive integer per input")
        object.__setattr__(self, "weeks", weeks.astype(int))

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.outputs))

    @property
    def best_input(self) -> np.ndarray:
        return self.inputs[self.best_index].copy()

    @property
    def best_output(self) -> float:
        return float(self.outputs[self.best_index])

    def with_observation(
        self, point: np.ndarray, value: float, *, week: int | None = None, std: float = np.nan,
    ) -> "Observations":
        """Return a copy with one new row; without a week, use the newest week already recorded."""
        candidate = np.asarray(point, dtype=float)
        if candidate.shape != (self.dimension,):
            raise ValueError("Point must have the configured dimension")
        if np.any(np.all(self.inputs == candidate, axis=1)):
            raise ValueError("Duplicate observation")
        row_week = int(self.weeks.max()) if week is None else _checked_week(week)
        return Observations(
            np.vstack((self.inputs, candidate)),
            np.append(self.outputs, value),
            self.dimension,
            np.append(self.expected_outputs, np.nan),
            np.append(self.stds, std),
            np.append(self.weeks, row_week),
        )

    def through_week(self, week: int) -> "Observations":
        keep = self.weeks <= _checked_week(week)
        return Observations(
            self.inputs[keep],
            self.outputs[keep],
            self.dimension,
            self.expected_outputs[keep],
            self.stds[keep],
            self.weeks[keep],
        )


def _directory(root: Path, function_id: int) -> Path:
    if function_id not in FUNCTION_SPECS:
        raise ValueError("Function ID must be between 1 and 8")
    return Path(root) / f"function_{function_id}"


def load_observations(
    root: Path, function_id: int | None = None, *, up_to_week: int | None = None,
) -> Observations:
    """Load observations; without an ID, read a one-dimensional dataset at root."""
    if function_id is None:
        directory = Path(root)
        dimension = 1
    else:
        directory = _directory(root, function_id)
        dimension = FUNCTION_SPECS[function_id]
    aggregate = directory / "all_observations.npz"
    if aggregate.exists():
        with np.load(aggregate, allow_pickle=False) as recorded:
            if "weeks" not in recorded.files:
                raise ValueError(f"Missing week labels; run scripts/add_week_labels.py: {aggregate}")
            expected = recorded["expected_outputs"] if "expected_outputs" in recorded.files else None
            stds = recorded["stds"] if "stds" in recorded.files else None
            observations = Observations(
                recorded["inputs"],
                recorded["outputs"],
                dimension,
                expected,
                stds,
                recorded["weeks"],
            )
    else:
        initial_inputs = np.load(directory / "initial_inputs.npy", allow_pickle=False)
        observations = Observations(
            initial_inputs,
            np.load(directory / "initial_outputs.npy", allow_pickle=False),
            dimension,
            weeks=np.ones(len(initial_inputs), dtype=int),
        )
    if up_to_week is None:
        return observations
    return observations.through_week(up_to_week)


def save_predictions(expected_outputs, suggested_points, standard_deviations, directory):
    expected_rows = [float(value) for value in expected_outputs]
    std_rows = [float(value) for value in standard_deviations]
    submissions = [format_submission(point) for point in suggested_points]
    input_rows = [np.round(np.asarray(point, dtype=float), 6).tolist() for point in suggested_points]
    if not expected_rows or not (len(expected_rows) == len(input_rows) == len(std_rows)):
        raise ValueError("Provide equally sized, nonempty arrays in matching function order.")
    if not np.all(np.isfinite(expected_rows)) or not np.all(np.isfinite(std_rows)):
        raise ValueError("Expected outputs and standard deviations must be finite.")
    if np.any(np.asarray(std_rows) < 0):
        raise ValueError("Standard deviations must be nonnegative.")

    payloads = {
        "expected_outputs.json": expected_rows,
        "inputs.json": input_rows,
        "stds.json": std_rows,
    }
    serialized = {
        name: json.dumps(values, indent=2, allow_nan=False) + "\n"
        for name, values in payloads.items()
    }
    serialized["inputs.json"] = "[\n" + ",\n".join(
        "  [" + ", ".join(f"{coordinate:.6f}" for coordinate in point) + "]"
        for point in input_rows
    ) + "\n]\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for name, content in serialized.items():
        path = directory / name
        path.write_text(content, encoding="utf-8")
        restored = json.loads(path.read_text(encoding="utf-8"))
        if restored != payloads[name]:
            raise ValueError(f"Saved values do not match: {path}")
        if name == "inputs.json" and [format_submission(point) for point in restored] != submissions:
            raise ValueError(f"Saved inputs do not match formatted submissions: {path}")
        saved_paths[name] = path
        print(f"Saved and verified {path}")
    return saved_paths


def _load_json_values(path: Path) -> list[object]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to parse weekly data file: {path}") from error
    if not isinstance(values, list):
        raise ValueError(f"Weekly data file must contain a list: {path}")
    return values


def _save_observations(
    path: Path,
    inputs: np.ndarray,
    outputs: np.ndarray,
    expected_outputs: np.ndarray,
    stds: np.ndarray,
    weeks: np.ndarray,
) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            np.savez(
                temp,
                inputs=inputs,
                outputs=outputs,
                expected_outputs=expected_outputs,
                stds=stds,
                weeks=weeks,
            )
            temp.flush()
            os.fsync(temp.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def append_observation(
    root: Path, function_id: int, point: np.ndarray, value: float, *, week: int, std: float = np.nan,
) -> Observations:
    """Validate and record a real evaluation without changing the original arrays."""
    directory = _directory(root, function_id)
    current = load_observations(root, function_id)
    updated = current.with_observation(point, value, week=week, std=std)
    _save_observations(
        directory / "all_observations.npz",
        updated.inputs,
        updated.outputs,
        updated.expected_outputs,
        updated.stds,
        updated.weeks,
    )
    return updated


def append_weekly_batch(
    root: Path,
    inputs_path: Path,
    outputs_path: Path,
    expected_outputs_path: Path,
    function_ids: Sequence[int] | None = None,
    *,
    stds_path: Path | None = None,
    week: int | None = None,
) -> dict[int, Observations]:
    """Import a JSON batch, including neighboring stds.json when available."""
    if week is None:
        match = re.fullmatch(r"week_(\d+)", Path(inputs_path).parent.name)
        if match is None:
            raise ValueError("Pass week=... or keep inputs.json inside a week_N folder")
        week = int(match.group(1))
    week = _checked_week(week)
    input_rows = _load_json_values(Path(inputs_path))
    output_rows = _load_json_values(Path(outputs_path))
    expected_rows = _load_json_values(Path(expected_outputs_path))
    if stds_path is None:
        neighboring_stds = Path(expected_outputs_path).with_name("stds.json")
        if neighboring_stds.exists():
            stds_path = neighboring_stds
    std_rows = (
        _load_json_values(Path(stds_path))
        if stds_path is not None
        else [np.nan] * len(input_rows)
    )
    if not input_rows or len(input_rows) != len(output_rows) or len(input_rows) != len(expected_rows):
        raise ValueError("Weekly inputs, outputs, and expected outputs must have the same nonzero length")
    if len(std_rows) != len(input_rows):
        raise ValueError("Weekly standard deviations must have one value per input")
    if stds_path is not None:
        std_values = np.asarray(std_rows, dtype=float)
        if std_values.ndim != 1 or not np.all(np.isfinite(std_values) & (std_values >= 0)):
            raise ValueError("Weekly standard deviations must be finite and nonnegative")

    ids = list(function_ids) if function_ids is not None else list(range(1, len(input_rows) + 1))
    if len(ids) != len(input_rows) or len(set(ids)) != len(ids):
        raise ValueError("Provide one unique function ID for each weekly observation")

    pending: dict[int, Observations] = {}
    results: dict[int, Observations] = {}
    for function_id, input_row, output_row, expected_row, std_row in zip(
        ids, input_rows, output_rows, expected_rows, std_rows,
    ):
        current = load_observations(root, function_id)
        point = np.asarray(input_row, dtype=float).reshape(1, -1)
        value = float(output_row)
        expected_value = float(expected_row)
        std_value = float(std_row)
        candidate = Observations(
            point,
            np.asarray([value]),
            current.dimension,
            np.asarray([expected_value]),
            np.asarray([std_value]),
        )
        matches = np.all(current.inputs == candidate.inputs[0], axis=1)
        if np.any(matches):
            if np.any(current.weeks[matches] != week):
                raise ValueError(f"Function {function_id} already has this input in another week")
            if not np.all(current.outputs[matches] == candidate.outputs[0]):
                raise ValueError(
                    f"Function {function_id} already has this input with a different output"
                )
            existing_expected = current.expected_outputs[matches]
            if np.any(np.isfinite(existing_expected) & (existing_expected != expected_value)):
                raise ValueError(
                    f"Function {function_id} input already has a different expected output"
                )
            existing_stds = current.stds[matches]
            if np.isfinite(std_value) and np.any(
                np.isfinite(existing_stds) & (existing_stds != std_value)
            ):
                raise ValueError(
                    f"Function {function_id} input already has a different standard deviation"
                )
            if np.all(np.isfinite(existing_expected)) and (
                np.isnan(std_value) or np.all(np.isfinite(existing_stds))
            ):
                results[function_id] = current
                continue
            expected_outputs = current.expected_outputs.copy()
            expected_outputs[matches] = expected_value
            stds = current.stds.copy()
            if np.isfinite(std_value):
                stds[matches] = std_value
            updated = Observations(
                current.inputs,
                current.outputs,
                current.dimension,
                expected_outputs,
                stds,
                current.weeks,
            )
        else:
            updated = current.with_observation(
                candidate.inputs[0], candidate.outputs[0], week=week, std=std_value,
            )
            expected_outputs = updated.expected_outputs.copy()
            expected_outputs[-1] = candidate.expected_outputs[0]
            updated = Observations(
                updated.inputs,
                updated.outputs,
                updated.dimension,
                expected_outputs,
                updated.stds,
                updated.weeks,
            )
        pending[function_id] = updated
        results[function_id] = updated

    for function_id, observations in pending.items():
        directory = _directory(root, function_id)
        _save_observations(
            directory / "all_observations.npz",
            observations.inputs,
            observations.outputs,
            observations.expected_outputs,
            observations.stds,
            observations.weeks,
        )
    return results
