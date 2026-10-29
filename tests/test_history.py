import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_bbo.data import FUNCTION_SPECS, load_observations, load_predictions
from bayesian_bbo.history import load_weekly_history


ROOT = Path(__file__).resolve().parents[1] / "data"


def test_history_matches_existing_weekly_calculations():
    history = load_weekly_history(ROOT, current_week=3)
    assert len(history.records) == 24
    for row in history.records.itertuples():
        before = load_observations(ROOT, row.function_id, up_to_week=row.week - 1)
        through = load_observations(ROOT, row.function_id, up_to_week=row.week)
        selected = through.weeks == row.week
        payload = load_predictions(ROOT / "weekly_updates" / f"week_{row.week}" / "predictions.json")
        assert row.previous_best == before.best_output
        assert row.observed_best == through.best_output
        assert row.expected_best == payload["expected_best_outputs"][row.function_id - 1]
        assert row.residual == pytest.approx(
            ((through.outputs[selected] - through.expected_outputs[selected]) / through.stds[selected])[0]
        )
    assert history.records.iloc[0].acquisition_settings["name"] == "ucb"
    assert history.observations[1].through_week(2).weeks.max() == 2
    assert history.observations[1].weeks.max() == 3


def test_history_subsets_preserve_function_mapping_and_order():
    full = load_weekly_history(ROOT, current_week=3)
    subset = load_weekly_history(ROOT, current_week=3, function_ids=[7, 2])
    assert subset.records.function_id.tolist() == [7, 2] * 3
    for row in subset.records.itertuples():
        expected = full.records.loc[(full.records.week == row.week) & (full.records.function_id == row.function_id)].iloc[0]
        assert row.candidate == expected.candidate
        assert row.expected_mean == expected.expected_mean
    assert len(subset.records.loc[subset.records.week == 3]) == 2


@pytest.fixture
def historical_root(tmp_path):
    directory = tmp_path / "function_1"
    directory.mkdir()
    np.savez(directory / "all_observations.npz", inputs=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
             outputs=[0.0, 1.0, 100.0], expected_outputs=[np.nan, 0.5, 0.0],
             stds=[np.nan, 0.2, 0.1], weeks=[0, 1, 2])
    week = tmp_path / "weekly_updates" / "week_1"
    week.mkdir(parents=True)
    (week / "inputs.json").write_text(json.dumps([[0.2] * dimension for dimension in FUNCTION_SPECS.values()]))
    payload = dict(expected_outputs=[0.5] * 8, stds=[0.2] * 8, expected_best_outputs=[0.6] * 8)
    (week / "predictions.json").write_text(json.dumps(payload))
    return tmp_path


def test_history_optional_metadata_and_future_exclusion(historical_root):
    history = load_weekly_history(historical_root, current_week=1, function_ids=[1])
    row = history.records.iloc[0]
    assert row.acquisition_settings is None
    assert row.acquisition_score is None
    assert row.previous_best == 0
    assert row.observed_best == 1
    assert row.residual == 2.5
    assert history.observations[1].weeks.max() == 1


@pytest.mark.parametrize("field,value,message", [
    ("stds", [0.0] * 8, "positive"),
    ("expected_outputs", [0.6] * 8, "prediction differs"),
    ("expected_best_outputs", [-1.0] * 8, "below"),
    ("stds", [0.3] * 8, "prediction differs"),
])
def test_history_rejects_mismatched_predictions(historical_root, field, value, message):
    path = historical_root / "weekly_updates" / "week_1" / "predictions.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_weekly_history(historical_root, current_week=1, function_ids=[1])


@pytest.mark.parametrize("week,ids", [(0, None), (-1, None), (True, None), (1, []), (1, [1, 1]), (1, [9]), (1, [1.0])])
def test_history_rejects_invalid_requests(week, ids):
    with pytest.raises(ValueError):
        load_weekly_history(ROOT, current_week=week, function_ids=ids)


@pytest.mark.parametrize("points", [[], [[0.2, 0.2]] * 7, [[0.3, 0.2]] * 8, [[1.0, 0.2]] * 8, [[0.2]] * 8])
def test_history_rejects_invalid_inputs(historical_root, points):
    path = historical_root / "weekly_updates" / "week_1" / "inputs.json"
    path.write_text(json.dumps(points))
    with pytest.raises(ValueError, match="inputs must|input differs|invalid input"):
        load_weekly_history(historical_root, current_week=1, function_ids=[1])


def test_history_accepts_negative_ucb_scores(historical_root):
    path = historical_root / "weekly_updates" / "week_1" / "predictions.json"
    payload = json.loads(path.read_text())
    payload["acquisition_scores"] = [-0.5] * 8
    payload["acquisition_settings"] = [dict(name="ucb", kappa=1.96) for _ in range(8)]
    path.write_text(json.dumps(payload))
    row = load_weekly_history(historical_root, current_week=1, function_ids=[1]).records.iloc[0]
    assert row.acquisition_score == -0.5
    assert row.acquisition_settings == dict(name="ucb", kappa=1.96)


def test_history_rejects_incomplete_prediction_arrays(historical_root):
    path = historical_root / "weekly_updates" / "week_1" / "predictions.json"
    path.write_text(json.dumps(dict(expected_outputs=[0.5], stds=[0.2], expected_best_outputs=[0.6])))
    with pytest.raises(ValueError, match="one entry per function"):
        load_weekly_history(historical_root, current_week=1, function_ids=[1])


def test_history_rejects_missing_files(historical_root):
    (historical_root / "weekly_updates" / "week_1" / "predictions.json").unlink()
    with pytest.raises(ValueError, match="Unable to parse predictions"):
        load_weekly_history(historical_root, current_week=1, function_ids=[1])


@pytest.mark.parametrize("weeks", [[0, 2, 2], [0, 1, 1]])
def test_history_rejects_missing_or_duplicate_week_rows(historical_root, weeks):
    path = historical_root / "function_1" / "all_observations.npz"
    with np.load(path) as arrays:
        recorded = {key: arrays[key] for key in arrays.files}
    recorded["weeks"] = weeks
    np.savez(path, **recorded)
    with pytest.raises(ValueError, match="exactly one"):
        load_weekly_history(historical_root, current_week=1, function_ids=[1])