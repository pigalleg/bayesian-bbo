import json
import re

import numpy as np
import pytest

from bayesian_bbo.acquisition import expected_best_output
from bayesian_bbo.data import (
    FUNCTION_SPECS,
    Observations,
    append_observation,
    append_weekly_batch,
    load_observations,
    save_predictions,
    write_predictions,
)

PREDICTIONS = '{"expected_outputs": [11.5], "stds": [0.5], "expected_best_outputs": [12.1]}'


def write_week(directory, *, inputs="[[0.2, 0.3]]", outputs="[12.0]", predictions=PREDICTIONS):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.json").write_text(inputs, encoding="utf-8")
    (directory / "outputs.json").write_text(outputs, encoding="utf-8")
    (directory / "predictions.json").write_text(predictions, encoding="utf-8")
    return directory / "inputs.json", directory / "outputs.json", directory / "predictions.json"


def test_function_specs_contains_only_functions_one_through_eight():
    assert tuple(FUNCTION_SPECS) == tuple(range(1, 9))


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
    updated = append_observation(tmp_path, 1, [0.2, 0.3], 12.0, week=1)
    assert updated.best_output == 12
    np.testing.assert_array_equal(updated.best_input, [0.2, 0.3])
    assert len(load_observations(tmp_path, 1).inputs) == 11
    assert (tmp_path / "function_1" / "all_observations.npz").exists()
    assert not (tmp_path / "function_1" / "added_observations.npz").exists()
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_inputs.npy"), x)
    np.testing.assert_array_equal(np.load(tmp_path / "function_1" / "initial_outputs.npy"), y)
    with pytest.raises(ValueError, match="Duplicate"):
        append_observation(tmp_path, 1, [0.2, 0.3], 13, week=1)


def test_load_observations_without_function_id_defaults_to_one_dimension(tmp_path):
    x = np.array([[0.1], [0.3], [0.7]])
    y = np.array([1.0, 2.0, 3.0])
    np.save(tmp_path / "initial_inputs.npy", x)
    np.save(tmp_path / "initial_outputs.npy", y)

    observations = load_observations(tmp_path)

    assert observations.dimension == 1
    np.testing.assert_array_equal(observations.inputs, x)
    np.testing.assert_array_equal(observations.outputs, y)


def test_save_predictions_preserves_values_and_submission_precision(tmp_path):
    points = [np.array([0.75533, 0.123456789]), np.array([0.0, 0.29667, 0.57364])]
    expected = [np.float64(1.25), -2.0]
    stds = [0.125, 0.5]
    best = [1.0, -3.0]
    paths = save_predictions(expected, points, stds, best, tmp_path)

    assert set(paths) == {"inputs.json", "predictions.json"}
    predictions = json.loads(paths["predictions.json"].read_text())
    assert predictions["expected_outputs"] == [1.25, -2.0]
    assert predictions["stds"] == [0.125, 0.5]
    np.testing.assert_allclose(
        predictions["expected_best_outputs"],
        expected_best_output(np.array(expected), np.array(stds), np.array(best)),
    )
    inputs_text = paths["inputs.json"].read_text()
    assert json.loads(inputs_text) == [[0.755330, 0.123457], [0.000000, 0.296670, 0.573640]]
    tokens = re.findall(r"-?\d+(?:\.\d+)?", inputs_text)
    assert len(tokens) == 5
    assert all(re.fullmatch(r"-?\d+\.\d{6}", token) for token in tokens)
    assert not (tmp_path / "outputs.json").exists()
    assert not (tmp_path / "expected_outputs.json").exists()
    assert not (tmp_path / "stds.json").exists()


def test_save_predictions_rejects_unmatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="equally sized"):
        save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0, 1.0], tmp_path)
    assert not (tmp_path / "predictions.json").exists()


@pytest.mark.parametrize("name", ["ei", "ucb"])
def test_save_predictions_round_trips_acquisition_metadata(tmp_path, name):
    from bayesian_bbo.data import _load_predictions
    from bayesian_bbo.acquisition import expected_improvement, upper_confidence_bound

    setting = {"name": "ei", "xi": 0.2, "best": 1.0} if name == "ei" else {"name": "ucb", "kappa": 1.96}
    score = float(
        expected_improvement(np.array(1.25), np.array(0.5), 1.0, xi=0.2)
        if name == "ei" else upper_confidence_bound(1.25, 0.5, kappa=1.96)
    )
    paths = save_predictions(
        [1.25], [[0.2, 0.3]], [0.5], [1.0], tmp_path,
        acquisition_scores=[score], acquisition_settings=[setting],
    )
    predictions = _load_predictions(paths["predictions.json"])
    assert predictions["acquisition_scores"] == [score]
    assert predictions["acquisition_settings"] == [setting]
    assert predictions["expected_best_outputs"][0] == pytest.approx(expected_best_output(1.25, 0.5, 1.0))
    if name == "ei":
        assert predictions["expected_best_outputs"][0] != pytest.approx(1.0 + score)


@pytest.mark.parametrize("scores,settings", [
    ([0.1], None),
    (None, [{"name": "ucb", "kappa": 2.0}]),
    ([], []),
    ([float("nan")], [{"name": "ucb", "kappa": 2.0}]),
    ([0.1], [{"name": "ei", "xi": -0.1, "best": 1.0}]),
    ([-0.1], [{"name": "ei", "xi": 0.1, "best": 1.0}]),
    ([0.1], [{"name": "ei", "xi": 0.1}]),
])
def test_save_predictions_rejects_invalid_acquisition_metadata(tmp_path, scores, settings):
    with pytest.raises(ValueError):
        save_predictions(
            [1.25], [[0.2, 0.3]], [0.5], [1.0], tmp_path,
            acquisition_scores=scores, acquisition_settings=settings,
        )
    assert not (tmp_path / "inputs.json").exists()


@pytest.mark.parametrize("stds", [[-0.1], [np.nan], [np.inf]])
def test_write_predictions_rejects_invalid_standard_deviations(tmp_path, stds):
    with pytest.raises(ValueError, match="Standard deviations"):
        write_predictions(tmp_path / "predictions.json", [0.0], stds, [0.0])
    assert not (tmp_path / "predictions.json").exists()


def test_save_predictions_refuses_to_replace_different_files(tmp_path):
    save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)
    inputs_before = (tmp_path / "inputs.json").read_text()
    predictions_before = (tmp_path / "predictions.json").read_text()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_predictions([2.0], [[0.3, 0.4]], [0.2], [0.0], tmp_path)

    assert (tmp_path / "inputs.json").read_text() == inputs_before
    assert (tmp_path / "predictions.json").read_text() == predictions_before


def test_save_predictions_refuses_when_only_predictions_differ(tmp_path):
    save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)
    inputs_before = (tmp_path / "inputs.json").read_text()

    with pytest.raises(FileExistsError):
        save_predictions([2.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)

    assert (tmp_path / "inputs.json").read_text() == inputs_before


def test_save_predictions_allows_identical_rewrite(tmp_path):
    save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)
    paths = save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)
    assert set(paths) == {"inputs.json", "predictions.json"}


def test_save_predictions_overwrites_only_when_requested(tmp_path):
    save_predictions([1.0], [[0.1, 0.2]], [0.1], [0.0], tmp_path)
    save_predictions([2.0], [[0.3, 0.4]], [0.2], [0.0], tmp_path, overwrite=True)
    predictions = json.loads((tmp_path / "predictions.json").read_text())
    assert predictions["expected_outputs"] == [2.0]
    assert json.loads((tmp_path / "inputs.json").read_text()) == [[0.3, 0.4]]


def test_write_predictions_refuses_to_replace_different_file(tmp_path):
    path = tmp_path / "predictions.json"
    write_predictions(path, [1.0], [0.1], [0.0])
    with pytest.raises(FileExistsError):
        write_predictions(path, [2.0], [0.1], [0.0])
    assert json.loads(path.read_text())["expected_outputs"] == [1.0]


def test_expected_best_output_is_best_plus_ei_without_xi():
    mean, std, best = 0.3, 0.7, 0.5
    # E[max(f, best)] for f ~ N(mean, std), computed by Monte Carlo.
    samples = np.random.default_rng(0).normal(mean, std, size=2_000_000)
    assert expected_best_output(mean, std, best) == pytest.approx(
        np.mean(np.maximum(samples, best)), abs=3e-3,
    )
    # Without uncertainty the expected best is the larger of the best and the mean.
    assert expected_best_output(0.7, 0.0, 0.5) == pytest.approx(0.7)
    assert expected_best_output(0.2, 0.0, 0.5) == pytest.approx(0.5)


def test_append_weekly_batch_bootstraps_and_is_idempotent(tmp_path):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(tmp_path / "week_1")

    appended = append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
    repeated = append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)

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
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)

    updated = append_observation(tmp_path, 1, [0.4, 0.5], 13.0, week=1, std=0.25)
    assert updated.expected_outputs[-2] == 11.5
    assert np.isnan(updated.expected_outputs[-1])
    np.testing.assert_array_equal(updated.stds[-2:], [0.5, 0.25])
    np.testing.assert_array_equal(load_observations(tmp_path, 1).stds[-2:], [0.5, 0.25])

    outputs_path.write_text("[12.0]\n", encoding="utf-8")
    predictions_path.write_text(
        '{"expected_outputs": [12.5], "stds": [0.5], "expected_best_outputs": [12.6]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different expected output"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)


def test_predictions_backfill_legacy_aggregate_and_reject_conflicts(tmp_path):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(tmp_path / "week_1")
    imported = append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)[1]

    aggregate = tmp_path / "function_1" / "all_observations.npz"
    np.savez(
        aggregate, inputs=imported.inputs, outputs=imported.outputs,
        expected_outputs=imported.expected_outputs, weeks=imported.weeks,
    )
    assert np.isnan(load_observations(tmp_path, 1).stds).all()
    updated = append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)[1]
    assert len(updated.outputs) == len(imported.outputs)
    assert updated.stds[-1] == 0.5
    assert np.isnan(updated.stds[:-1]).all()
    before = aggregate.read_bytes()
    append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
    assert aggregate.read_bytes() == before

    predictions_path.write_text(
        '{"expected_outputs": [11.5], "stds": [0.6], "expected_best_outputs": [12.1]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different standard deviation"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
    assert aggregate.read_bytes() == before


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ('{"expected_outputs": [11.5], "stds": [], "expected_best_outputs": [12.1]}', "equally sized"),
        ('{"expected_outputs": [11.5], "stds": [-0.1], "expected_best_outputs": [12.1]}', "Standard deviations"),
        ('{"expected_outputs": [11.5], "stds": [NaN], "expected_best_outputs": [12.1]}', "Standard deviations"),
        ('{"expected_outputs": [11.5], "stds": [Infinity], "expected_best_outputs": [12.1]}', "Standard deviations"),
        ('{"expected_outputs": [11.5], "stds": [[0.5]], "expected_best_outputs": [12.1]}', "list of numbers"),
        ('{"expected_outputs": [11.5], "stds": [0.5]}', "must contain exactly"),
        ("[]", "must contain exactly"),
    ],
)
def test_predictions_reject_invalid_values_before_saving(tmp_path, predictions, message):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(
        tmp_path / "week_1", predictions=predictions,
    )

    with pytest.raises(ValueError, match=message):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
    assert not (tmp_path / "function_1" / "all_observations.npz").exists()


def test_weekly_batch_rejects_mismatched_prediction_length(tmp_path):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(
        tmp_path / "week_1", inputs="[[0.2, 0.3], [0.4, 0.5]]", outputs="[12.0, 13.0]",
    )

    with pytest.raises(ValueError, match="same nonzero length"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
    assert not (tmp_path / "function_1" / "all_observations.npz").exists()


@pytest.mark.parametrize("stds", [np.zeros(9), np.full(10, -0.1), np.full(10, np.inf)])
def test_observations_validate_stds(tmp_path, stds):
    inputs, outputs = initial_data(tmp_path)
    with pytest.raises(ValueError, match="Standard deviations"):
        Observations(inputs, outputs, 2, stds=stds)


@pytest.mark.parametrize("contents", ["[array([0.2, 0.3])]", "[[0.2, 0.3],]", "{}"])
def test_weekly_batch_rejects_invalid_json(tmp_path, contents):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(tmp_path / "week_1", inputs=contents)

    with pytest.raises(ValueError, match="weekly data file|Weekly data file"):
        append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path, week=1)
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


def test_up_to_week_filters_rows(tmp_path):
    initial_data(tmp_path)
    append_observation(tmp_path, 1, [0.4, 0.5], 13.0, week=1)
    append_observation(tmp_path, 1, [0.6, 0.7], 14.0, week=2)

    everything = load_observations(tmp_path, 1)
    assert set(everything.weeks.tolist()) == {0, 1, 2}
    assert len(load_observations(tmp_path, 1, up_to_week=2).inputs) == 12
    through_one = load_observations(tmp_path, 1, up_to_week=1)
    assert len(through_one.inputs) == 11
    assert through_one.weeks.max() == 1
    assert len(load_observations(tmp_path, 1, up_to_week=0).inputs) == 10


def test_weekly_folder_infers_week(tmp_path):
    initial_data(tmp_path)
    inputs_path, outputs_path, predictions_path = write_week(tmp_path / "week_1")

    imported = append_weekly_batch(tmp_path, inputs_path, outputs_path, predictions_path)[1]

    assert imported.weeks[-1] == 1
    assert load_observations(tmp_path, 1).weeks[-1] == 1


def test_missing_week_labels_are_rejected(tmp_path):
    initial_data(tmp_path)
    np.savez(
        tmp_path / "function_1" / "all_observations.npz",
        inputs=np.array([[0.2, 0.3]]), outputs=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="Missing week labels"):
        load_observations(tmp_path, 1)



def test_wrong_count_and_invalid_append(tmp_path):
    initial_data(tmp_path)
    directory = tmp_path / "function_1"
    np.save(directory / "initial_inputs.npy", np.zeros((9, 2)))
    np.save(directory / "initial_outputs.npy", np.zeros(9))
    observations = load_observations(tmp_path, 1)
    assert len(observations.inputs) == 9
    initial_data_arrays = Observations(
        np.zeros((2, 2)), np.array([1, 2]), 2, weeks=np.zeros(2, dtype=int),
    )
    with pytest.raises(ValueError, match="dimension"):
        initial_data_arrays.with_observation([0.1], 3, week=1)
    with pytest.raises(ValueError, match="within"):
        initial_data_arrays.with_observation([0.1, 1], 3, week=1)
    with pytest.raises(ValueError, match="finite"):
        initial_data_arrays.with_observation([0.1, 0.2], float("nan"), week=1)
    with pytest.raises(ValueError, match="Week must be a nonnegative integer"):
        initial_data_arrays.with_observation([0.1, 0.2], 3, week=-1)


def test_initial_observations_and_explicit_week_zero_are_supported(tmp_path):
    initial_data(tmp_path)
    observations = load_observations(tmp_path, 1, up_to_week=0)
    np.testing.assert_array_equal(observations.weeks, np.zeros(10, dtype=int))
    updated = observations.with_observation([0.4, 0.5], 13., week=0)
    assert updated.weeks[-1] == 0
    assert len(updated.through_week(0).inputs) == 11


@pytest.mark.parametrize("week", [-1, True, 0.5])
def test_invalid_week_labels_are_rejected(week):
    observations = Observations([[0.1, 0.2]], [1.], 2)
    with pytest.raises(ValueError, match="Week must be a nonnegative integer"):
        observations.through_week(week)
    with pytest.raises(ValueError, match="Week must be a nonnegative integer"):
        observations.with_observation([0.3, 0.4], 2., week=week)


def test_negative_stored_week_labels_are_rejected():
    with pytest.raises(ValueError, match="Weeks must contain one nonnegative integer"):
        Observations([[0.1, 0.2]], [1.], 2, weeks=np.array([-1]))
