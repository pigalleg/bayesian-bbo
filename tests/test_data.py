from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from bayesian_bbo.data import Observations, append_observation, load_observations


def initial_data(tmp_path):
    directory = tmp_path / "function_1"
    directory.mkdir()
    x = np.arange(20).reshape(10, 2) / 100
    y = np.arange(10, dtype=float).reshape(-1, 1)
    np.save(directory / "initial_inputs.npy", x)
    np.save(directory / "initial_outputs.npy", y)
    return x, y


def test_load_append_and_preserve_originals(tmp_path):
    x, y = initial_data(tmp_path)
    initial = load_observations(tmp_path, 1)
    assert initial.inputs.shape == (10, 2)
    assert initial.outputs.shape == (10,)
    updated = append_observation(tmp_path, 1, [0.2, 0.3], 12.0)
    assert updated.best_output == 12
    np.testing.assert_array_equal(updated.best_input, [0.2, 0.3])
    assert len(load_observations(tmp_path, 1).inputs) == 11
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_inputs.npy"), x)
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_outputs.npy"), y)
    with pytest.raises(ValueError, match="Duplicate"):
        append_observation(tmp_path, 1, [0.2, 0.3], 13)


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        (np.zeros((10, 3)), np.zeros(10)),
        (np.zeros((10, 2)), np.zeros(9)),
        (np.full((10, 2), 1.0), np.zeros(10)),
        (np.full((10, 2), -0.01), np.zeros(10)),
        (np.full((10, 2), np.nan), np.zeros(10)),
        (np.zeros((10, 2)), np.full(10, np.inf)),
    ],
)
def test_invalid_initial_data(tmp_path, inputs, outputs):
    initial_data(tmp_path)
    directory = tmp_path / "function_1"
    np.save(directory / "initial_inputs.npy", inputs)
    np.save(directory / "initial_outputs.npy", outputs)
    with pytest.raises(ValueError):
        load_observations(tmp_path, 1)


def test_wrong_count_and_invalid_append(tmp_path):
    initial_data(tmp_path)
    directory = tmp_path / "function_1"
    np.save(directory / "initial_inputs.npy", np.zeros((9, 2)))
    np.save(directory / "initial_outputs.npy", np.zeros(9))
    with pytest.raises(ValueError, match="10 initial"):
        load_observations(tmp_path, 1)
    initial_data_arrays = Observations(np.zeros((2, 2)), np.array([1, 2]), 2)
    with pytest.raises(ValueError, match="dimension"):
        initial_data_arrays.with_observation([0.1], 3)
    with pytest.raises(ValueError, match="within"):
        initial_data_arrays.with_observation([0.1, 1], 3)
    with pytest.raises(ValueError, match="finite"):
        initial_data_arrays.with_observation([0.1, 0.2], float("nan"))


def test_concurrent_appends_retain_every_observation(tmp_path):
    initial_data(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(
            lambda i: append_observation(tmp_path, 1, [0.2 + i / 100, 0.3], float(i)),
            range(8),
        ))
    loaded = load_observations(tmp_path, 1)
    assert len(loaded.inputs) == 18
    assert set(loaded.outputs[10:]) == set(range(8))
