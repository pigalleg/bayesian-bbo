"""Read original and cumulative observations without changing the original files."""

from dataclasses import dataclass
from pathlib import Path
import fcntl
import os
import tempfile

import numpy as np


FUNCTION_SPECS = {
    1: (2, 10),
    2: (2, 10),
    3: (3, 15),
    4: (4, 30),
    5: (4, 20),
    6: (5, 20),
    7: (6, 30),
    8: (8, 40),
}


@dataclass(frozen=True)
class Observations:
    inputs: np.ndarray
    outputs: np.ndarray
    dimension: int

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
        object.__setattr__(self, "inputs", x.copy())
        object.__setattr__(self, "outputs", y.copy())

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.outputs))

    @property
    def best_input(self) -> np.ndarray:
        return self.inputs[self.best_index].copy()

    @property
    def best_output(self) -> float:
        return float(self.outputs[self.best_index])

    def with_observation(self, point: np.ndarray, value: float) -> "Observations":
        candidate = np.asarray(point, dtype=float)
        if candidate.shape != (self.dimension,):
            raise ValueError("Point must have the configured dimension")
        if np.any(np.all(self.inputs == candidate, axis=1)):
            raise ValueError("Duplicate observation")
        return Observations(
            np.vstack((self.inputs, candidate)),
            np.append(self.outputs, value),
            self.dimension,
        )


def _directory(root: Path, function_id: int) -> Path:
    if function_id not in FUNCTION_SPECS:
        raise ValueError("Function ID must be between 1 and 8")
    return Path(root) / f"function_{function_id}"


def load_observations(root: Path, function_id: int) -> Observations:
    """Load the provided initial pair plus any subsequently appended observations."""
    directory = _directory(root, function_id)
    dimension, initial_count = FUNCTION_SPECS[function_id]
    initial = Observations(
        np.load(directory / "initial_inputs.npy", allow_pickle=False),
        np.load(directory / "initial_outputs.npy", allow_pickle=False),
        dimension,
    )
    if len(initial.inputs) != initial_count:
        raise ValueError(f"Function {function_id} requires {initial_count} initial points")
    extra = directory / "added_observations.npz"
    if not extra.exists():
        return initial
    with np.load(extra, allow_pickle=False) as recorded:
        x = recorded["inputs"]
        y = recorded["outputs"]
    additional = Observations(x, y, dimension)
    return Observations(
        np.concatenate((initial.inputs, additional.inputs), axis=0),
        np.concatenate((initial.outputs, additional.outputs), axis=0),
        dimension,
    )


def _save_observations(path: Path, inputs: np.ndarray, outputs: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            np.savez(temp, inputs=inputs, outputs=outputs)
            temp.flush()
            os.fsync(temp.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def append_observation(root: Path, function_id: int, point: np.ndarray, value: float) -> Observations:
    """Validate and record a real evaluation without changing the original arrays."""
    directory = _directory(root, function_id)
    with (directory / "added_observations.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = load_observations(root, function_id)
        updated = current.with_observation(point, value)
        initial_count = FUNCTION_SPECS[function_id][1]
        _save_observations(
            directory / "added_observations.npz",
            updated.inputs[initial_count:],
            updated.outputs[initial_count:],
        )
    return updated
