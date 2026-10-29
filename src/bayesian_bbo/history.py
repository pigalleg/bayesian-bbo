"""Validated weekly prediction and outcome history for analysis notebooks."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FUNCTION_SPECS, Observations, load_observations, load_predictions


@dataclass(frozen=True)
class WeeklyHistory:
    records: pd.DataFrame
    observations: dict[int, Observations]


def load_weekly_history(
    data_root: Path,
    *,
    current_week: int,
    function_ids: Sequence[int] | None = None,
) -> WeeklyHistory:
    """Collect weeks 1..current_week; week 0 supplies the initial incumbent.

    Weekly JSON rows use FUNCTION_SPECS order, even when selecting a subset.
    Observations are cached through current_week; callers must apply their own
    earlier training cutoff when reconstructing historical models.
    """
    if isinstance(current_week, bool) or not isinstance(current_week, (int, np.integer)) or current_week < 1:
        raise ValueError("current_week must be a positive integer")
    ids = tuple(FUNCTION_SPECS) if function_ids is None else tuple(function_ids)
    if not ids or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in FUNCTION_SPECS
        for value in ids
    ):
        raise ValueError("Provide supported function IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Function IDs must be unique")
    root = Path(data_root)
    observations = {
        function_id: load_observations(root, function_id, up_to_week=current_week)
        for function_id in ids
    }
    positions = {function_id: index for index, function_id in enumerate(FUNCTION_SPECS)}
    records = []
    for week in range(1, current_week + 1):
        directory = root / "weekly_updates" / f"week_{week}"
        predictions = load_predictions(directory / "predictions.json")
        try:
            inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Week {week}: unable to read inputs") from error
        if not isinstance(inputs, list) or len(inputs) != len(FUNCTION_SPECS):
            raise ValueError(f"Week {week}: inputs must contain one point per function")
        if len(predictions["expected_outputs"]) != len(FUNCTION_SPECS):
            raise ValueError(f"Week {week}: predictions must contain one entry per function")
        for function_id in ids:
            index = positions[function_id]
            context = f"Function {function_id}, week {week}"
            through = observations[function_id]
            if through.weeks is None or through.expected_outputs is None or through.stds is None:
                raise ValueError(f"{context}: missing historical predictions or weeks")
            this_week = through.weeks == week
            if np.count_nonzero(this_week) != 1:
                raise ValueError(f"{context}: expected exactly one evaluated observation")
            point = np.asarray(inputs[index], dtype=float)
            if (
                point.shape != (through.dimension,)
                or not np.all(np.isfinite(point))
                or np.any((point < 0) | (point >= 1))
            ):
                raise ValueError(f"{context}: invalid input coordinates")
            if not np.allclose(point, through.inputs[this_week][0], rtol=0, atol=1e-12):
                raise ValueError(f"{context}: input differs from observation record")
            mean = predictions["expected_outputs"][index]
            std = predictions["stds"][index]
            if std <= 0:
                raise ValueError(f"{context}: standard deviation must be positive for residuals")
            if not np.isclose(mean, through.expected_outputs[this_week][0]) or not np.isclose(
                std, through.stds[this_week][0],
            ):
                raise ValueError(f"{context}: prediction differs from observation record")
            previous_best = through.through_week(week - 1).best_output
            expected_best = predictions["expected_best_outputs"][index]
            if expected_best < previous_best - 1e-9 * (1 + abs(previous_best)):
                raise ValueError(f"{context}: expected best is below the previous best")
            observed_output = float(through.outputs[this_week][0])
            scores = predictions.get("acquisition_scores")
            settings = predictions.get("acquisition_settings")
            records.append(dict(
                function_id=function_id, week=week, candidate=tuple(point),
                previous_best=previous_best, expected_best=expected_best,
                observed_best=max(previous_best, observed_output),
                expected_mean=mean, expected_std=std, observed_output=observed_output,
                residual=(observed_output - mean) / std,
                acquisition_score=None if scores is None else scores[index],
                acquisition_settings=None if settings is None else dict(settings[index]),
            ))
    return WeeklyHistory(pd.DataFrame(records), observations)