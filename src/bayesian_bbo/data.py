"""Read original and cumulative observations without changing the original files."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np

from .acquisition import expected_best_output
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


PREDICTION_FIELDS = ("expected_outputs", "stds", "expected_best_outputs")
ACQUISITION_FIELDS = ("acquisition_scores", "acquisition_settings")


def _number_list(values: object, name: str) -> list[float]:
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
    ):
        raise ValueError(f"Predictions field must be a list of numbers: {name}")
    return [float(value) for value in values]


def _checked_predictions(
    expected_outputs: Sequence[float],
    stds: Sequence[float],
    expected_best_outputs: Sequence[float],
    acquisition_scores: Sequence[float] | None = None,
    acquisition_settings: Sequence[dict[str, str | float]] | None = None,
) -> dict:
    payload = {
        "expected_outputs": [float(value) for value in expected_outputs],
        "stds": [float(value) for value in stds],
        "expected_best_outputs": [float(value) for value in expected_best_outputs],
    }
    lengths = {len(values) for values in payload.values()}
    if len(lengths) != 1 or not lengths.pop():
        raise ValueError("Provide equally sized, nonempty predictions in matching function order.")
    if not np.all(np.isfinite(payload["expected_outputs"])):
        raise ValueError("Expected outputs must be finite.")
    if not np.all(np.isfinite(payload["expected_best_outputs"])):
        raise ValueError("Expected best outputs must be finite.")
    if not np.all(np.isfinite(payload["stds"])) or np.any(np.asarray(payload["stds"]) < 0):
        raise ValueError("Standard deviations must be finite and nonnegative.")
    if (acquisition_scores is None) != (acquisition_settings is None):
        raise ValueError("Acquisition scores and settings must be provided together.")
    if acquisition_scores is not None and acquisition_settings is not None:
        scores = _number_list(list(acquisition_scores), "acquisition_scores")
        settings = list(acquisition_settings)
        if len(scores) != len(expected_outputs) or len(settings) != len(scores):
            raise ValueError("Provide equally sized acquisition scores, settings, and predictions.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Acquisition scores must be finite.")
        for score, setting in zip(scores, settings):
            if not isinstance(setting, dict):
                raise ValueError("Acquisition settings must be objects.")
            name = setting.get("name")
            required = {"name", "xi", "best"} if name == "ei" else {"name", "kappa"}
            if name not in {"ei", "ucb"} or set(setting) != required:
                raise ValueError("Acquisition settings require name='ei', xi, best or name='ucb', kappa.")
            for key in required - {"name"}:
                value = setting[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError("Acquisition parameters must be finite numbers.")
                if key != "best" and value < 0:
                    raise ValueError("Acquisition parameters must be nonnegative.")
            if name == "ei" and score < 0:
                raise ValueError("EI scores must be nonnegative.")
        payload = {**payload, "acquisition_scores": scores, "acquisition_settings": settings}
    return payload


def _predictions_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _refuse_changes(path: Path, text: str, *, overwrite: bool) -> None:
    """Raise before writing if the file exists with different content, unless overwrite is set."""
    if overwrite or not path.exists() or path.read_text(encoding="utf-8") == text:
        return
    raise FileExistsError(f"{path} exists with different values; pass overwrite=True to replace it")


def write_predictions(
    path: Path,
    expected_outputs: Sequence[float],
    stds: Sequence[float],
    expected_best_outputs: Sequence[float],
    *,
    acquisition_scores: Sequence[float] | None = None,
    acquisition_settings: Sequence[dict[str, str | float]] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one week's predictions and verify the saved file reads back unchanged."""
    payload = _checked_predictions(
        expected_outputs, stds, expected_best_outputs, acquisition_scores, acquisition_settings,
    )
    path = Path(path)
    text = _predictions_text(payload)
    _refuse_changes(path, text, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"Saved values do not match: {path}")
    print(f"Saved and verified {path}")
    return path


def _load_predictions(path: Path) -> dict:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to parse predictions file: {path}") from error
    if not isinstance(values, dict) or set(values) not in (
        set(PREDICTION_FIELDS), set(PREDICTION_FIELDS + ACQUISITION_FIELDS),
    ):
        raise ValueError(
            f"Predictions file must contain exactly {', '.join(PREDICTION_FIELDS)}, "
            f"optionally with both {', '.join(ACQUISITION_FIELDS)}: {path}"
        )
    fields = {name: _number_list(values[name], name) for name in PREDICTION_FIELDS}
    return _checked_predictions(
        fields["expected_outputs"], fields["stds"], fields["expected_best_outputs"],
        values.get("acquisition_scores"), values.get("acquisition_settings"),
    )


def save_predictions(
    expected_outputs,
    suggested_points,
    standard_deviations,
    best_outputs,
    directory,
    *,
    acquisition_scores: Sequence[float] | None = None,
    acquisition_settings: Sequence[dict[str, str | float]] | None = None,
    overwrite: bool = False,
):
    """Write inputs.json and predictions.json; best_outputs are the best-so-far before this week. Different existing files are kept unless overwrite is True."""
    expected = np.asarray(expected_outputs, dtype=float)
    stds = np.asarray(standard_deviations, dtype=float)
    best = np.asarray(best_outputs, dtype=float)
    submissions = [format_submission(point) for point in suggested_points]
    input_rows = [np.round(np.asarray(point, dtype=float), 6).tolist() for point in suggested_points]
    if not len(input_rows) or not (len(expected) == len(stds) == len(best) == len(input_rows)):
        raise ValueError("Provide equally sized, nonempty arrays in matching function order.")
    expected_best = expected_best_output(expected, stds, best)
    predictions = _checked_predictions(
        expected.tolist(), stds.tolist(), expected_best.tolist(),
        acquisition_scores, acquisition_settings,
    )

    directory = Path(directory)
    inputs_text = "[\n" + ",\n".join(
        "  [" + ", ".join(f"{coordinate:.6f}" for coordinate in point) + "]"
        for point in input_rows
    ) + "\n]\n"
    inputs_path = directory / "inputs.json"
    predictions_path = directory / "predictions.json"
    _refuse_changes(inputs_path, inputs_text, overwrite=overwrite)
    _refuse_changes(predictions_path, _predictions_text(predictions), overwrite=overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(inputs_text, encoding="utf-8")
    restored_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if restored_inputs != input_rows:
        raise ValueError(f"Saved values do not match: {inputs_path}")
    if [format_submission(point) for point in restored_inputs] != submissions:
        raise ValueError(f"Saved inputs do not match formatted submissions: {inputs_path}")
    print(f"Saved and verified {inputs_path}")

    predictions_path = write_predictions(
        predictions_path,
        predictions["expected_outputs"],
        predictions["stds"],
        predictions["expected_best_outputs"],
        acquisition_scores=acquisition_scores,
        acquisition_settings=acquisition_settings,
        overwrite=overwrite,
    )
    return {"inputs.json": inputs_path, "predictions.json": predictions_path}


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
    predictions_path: Path,
    function_ids: Sequence[int] | None = None,
    *,
    week: int | None = None,
) -> dict[int, Observations]:
    """Import one week's inputs, outputs, and predictions.json; the week defaults to the week_N folder."""
    if week is None:
        match = re.fullmatch(r"week_(\d+)", Path(inputs_path).parent.name)
        if match is None:
            raise ValueError("Pass week=... or keep inputs.json inside a week_N folder")
        week = int(match.group(1))
    week = _checked_week(week)
    input_rows = _load_json_values(Path(inputs_path))
    output_rows = _load_json_values(Path(outputs_path))
    predictions = _load_predictions(Path(predictions_path))
    expected_rows = predictions["expected_outputs"]
    std_rows = predictions["stds"]
    if not input_rows or not (len(input_rows) == len(output_rows) == len(expected_rows)):
        raise ValueError("Weekly inputs, outputs, and predictions must have the same nonzero length")

    ids = list(function_ids) if function_ids is not None else list(range(1, len(input_rows) + 1))
    if len(ids) != len(input_rows) or len(set(ids)) != len(ids):
        raise ValueError("Provide one unique function ID for each weekly observation")

    pending: dict[int, Observations] = {}
    results: dict[int, Observations] = {}
    for function_id, input_row, output_row, expected_value, std_value in zip(
        ids, input_rows, output_rows, expected_rows, std_rows,
    ):
        current = load_observations(root, function_id)
        point = np.asarray(input_row, dtype=float).reshape(1, -1)
        value = float(output_row)
        matches = np.all(current.inputs == point[0], axis=1)
        if np.any(matches):
            if np.any(current.weeks[matches] != week):
                raise ValueError(f"Function {function_id} already has this input in another week")
            if not np.all(current.outputs[matches] == value):
                raise ValueError(
                    f"Function {function_id} already has this input with a different output"
                )
            existing_expected = current.expected_outputs[matches]
            if np.any(np.isfinite(existing_expected) & (existing_expected != expected_value)):
                raise ValueError(
                    f"Function {function_id} input already has a different expected output"
                )
            existing_stds = current.stds[matches]
            if np.any(np.isfinite(existing_stds) & (existing_stds != std_value)):
                raise ValueError(
                    f"Function {function_id} input already has a different standard deviation"
                )
            if np.all(np.isfinite(existing_expected)) and np.all(np.isfinite(existing_stds)):
                results[function_id] = current
                continue
            expected_outputs = current.expected_outputs.copy()
            expected_outputs[matches] = expected_value
            stds = current.stds.copy()
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
            updated = Observations(
                np.vstack((current.inputs, point)),
                np.append(current.outputs, value),
                current.dimension,
                np.append(current.expected_outputs, expected_value),
                np.append(current.stds, std_value),
                np.append(current.weeks, week),
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
