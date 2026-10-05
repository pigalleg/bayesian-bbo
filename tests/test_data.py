import json
import re

import numpy as np
import pytest

from bayesian_bbo.data import (
    Observations,
    append_observation,
    append_weekly_batch,
    load_observations,
    save_predictions,
)


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
    assert (tmp_path / "function_1" / "all_observations.npz").exists()
    assert not (tmp_path / "function_1" / "added_observations.npz").exists()
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_inputs.npy"), x)
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_outputs.npy"), y)
    with pytest.raises(ValueError, match="Duplicate"):
        append_observation(tmp_path, 1, [0.2, 0.3], 13)


def test_save_predictions_preserves_values_and_submission_precision(tmp_path):
    points = [np.array([0.75533, 0.123456789]), np.array([0.0, 0.29667, 0.57364])]
    paths = save_predictions([np.float64(1.25), -2.0], points, [0.125, 0.5], tmp_path)

    assert json.loads(paths["expected_outputs.json"].read_text()) == [1.25, -2.0]
    assert json.loads(paths["stds.json"].read_text()) == [0.125, 0.5]
    inputs_text = paths["inputs.json"].read_text()
    assert json.loads(inputs_text) == [[0.755330, 0.123457], [0.000000, 0.296670, 0.573640]]
    tokens = re.findall(r"-?\d+(?:\.\d+)?", inputs_text)
    assert len(tokens) == 5
    assert all(re.fullmatch(r"-?\d+\.\d{6}", token) for token in tokens)
    assert not (tmp_path / "outputs.json").exists()


def test_append_weekly_batch_bootstraps_and_is_idempotent(tmp_path):
    initial_data(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    outputs_path = tmp_path / "outputs.json"
    expected_path = tmp_path / "expected_outputs.json"
    inputs_path.write_text("[[0.2, 0.3]]\n", encoding="utf-8")
    outputs_path.write_text("[12.0]\n", encoding="utf-8")
    expected_path.write_text("[11.5]\n", encoding="utf-8")
    stds_path = tmp_path / "stds.json"
    stds_path.write_text("[0.5]\n", encoding="utf-8")

    appended = append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)
    repeated = append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)

    assert len(appended[1].inputs) == 11
    assert len(repeated[1].inputs) == 11
    np.testing.assert_array_equal(repeated[1].inputs[-1], [0.2, 0.3])
    assert repeated[1].outputs[-1] == 12.0
    assert np.isnan(repeated[1].expected_outputs[:10]).all()
    assert repeated[1].expected_outputs[-1] == 11.5
    assert np.isnan(repeated[1].stds[:10]).all()
    assert repeated[1].stds[-1] == 0.5
    assert load_observations(tmp_path, 1).stds[-1] == 0.5
    with np.load(tmp_path / "function_1" / "all_observations.npz") as stored:
        assert len(stored["inputs"]) == 11
        assert stored["outputs"][-1] == 12.0
        assert stored["expected_outputs"][-1] == 11.5
        assert stored["stds"][-1] == 0.5
    assert not (tmp_path / "function_1" / "added_observations.npz").exists()

    outputs_path.write_text("[13.0]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different output"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)

    updated = append_observation(tmp_path, 1, [0.4, 0.5], 13.0, std=0.25)
    assert updated.expected_outputs[-2] == 11.5
    assert np.isnan(updated.expected_outputs[-1])
    np.testing.assert_array_equal(updated.stds[-2:], [0.5, 0.25])
    np.testing.assert_array_equal(load_observations(tmp_path, 1).stds[-2:], [0.5, 0.25])

    outputs_path.write_text("[12.0]\n", encoding="utf-8")
    expected_path.write_text("[12.5]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different expected output"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)


def test_weekly_stds_backfill_legacy_aggregate_and_reject_conflicts(tmp_path):
    initial_data(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    outputs_path = tmp_path / "outputs.json"
    expected_path = tmp_path / "expected_outputs.json"
    inputs_path.write_text("[[0.2, 0.3]]", encoding="utf-8")
    outputs_path.write_text("[12.0]", encoding="utf-8")
    expected_path.write_text("[11.5]", encoding="utf-8")
    imported = append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)[1]
    assert np.isnan(imported.stds).all()

    aggregate = tmp_path / "function_1" / "all_observations.npz"
    np.savez(
        aggregate, inputs=imported.inputs, outputs=imported.outputs,
        expected_outputs=imported.expected_outputs,
    )
    assert np.isnan(load_observations(tmp_path, 1).stds).all()
    stds_path = tmp_path / "uncertainties.json"
    stds_path.write_text("[0.5]", encoding="utf-8")
    updated = append_weekly_batch(
        tmp_path, inputs_path, outputs_path, expected_path, stds_path=stds_path,
    )[1]
    assert len(updated.outputs) == len(imported.outputs)
    assert updated.stds[-1] == 0.5
    assert np.isnan(updated.stds[:-1]).all()
    before = aggregate.read_bytes()
    append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path, stds_path=stds_path)
    assert aggregate.read_bytes() == before
    without_stds = append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)[1]
    assert without_stds.stds[-1] == 0.5
    stds_path.write_text("[0.6]", encoding="utf-8")
    with pytest.raises(ValueError, match="different standard deviation"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path, stds_path=stds_path)
    assert aggregate.read_bytes() == before


@pytest.mark.parametrize("contents", ["[]", "[0.5, 0.6]", "[-0.1]", "[NaN]", "[Infinity]", "[[0.5]]"])
def test_weekly_stds_reject_invalid_values_before_saving(tmp_path, contents):
    initial_data(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    outputs_path = tmp_path / "outputs.json"
    expected_path = tmp_path / "expected_outputs.json"
    inputs_path.write_text("[[0.2, 0.3]]", encoding="utf-8")
    outputs_path.write_text("[12.0]", encoding="utf-8")
    expected_path.write_text("[11.5]", encoding="utf-8")
    (tmp_path / "stds.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="standard deviations"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)
    assert not (tmp_path / "function_1" / "all_observations.npz").exists()


@pytest.mark.parametrize("stds", [np.zeros(9), np.full(10, -0.1), np.full(10, np.inf)])
def test_observations_validate_stds(tmp_path, stds):
    inputs, outputs = initial_data(tmp_path)
    with pytest.raises(ValueError, match="Standard deviations"):
        Observations(inputs, outputs, 2, stds=stds)


@pytest.mark.parametrize("contents", ["[array([0.2, 0.3])]", "[[0.2, 0.3],]", "{}"])
def test_weekly_batch_rejects_invalid_json(tmp_path, contents):
    initial_data(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    outputs_path = tmp_path / "outputs.json"
    expected_path = tmp_path / "expected_outputs.json"
    inputs_path.write_text(contents, encoding="utf-8")
    outputs_path.write_text("[12.0]", encoding="utf-8")
    expected_path.write_text("[11.5]", encoding="utf-8")

    with pytest.raises(ValueError, match="weekly data file|Weekly data file"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, expected_path)
    assert not (tmp_path / "function_1" / "all_observations.npz").exists()


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
    observations = load_observations(tmp_path, 1)
    assert len(observations.inputs) == 9
    initial_data_arrays = Observations(np.zeros((2, 2)), np.array([1, 2]), 2)
    with pytest.raises(ValueError, match="dimension"):
        initial_data_arrays.with_observation([0.1], 3)
    with pytest.raises(ValueError, match="within"):
        initial_data_arrays.with_observation([0.1, 1], 3)
    with pytest.raises(ValueError, match="finite"):
        initial_data_arrays.with_observation([0.1, 0.2], float("nan"))


