import numpy as np
import pytest

from bayesian_bbo.acquisition import expected_improvement
from bayesian_bbo.data import Observations
from bayesian_bbo.optimisation import BayesianOptimizer, OptimisationConfig
from bayesian_bbo.submission import format_submission


@pytest.mark.parametrize(
    "settings",
    [
        {"budget": -1},
        {"budget": 1.5},
        {"candidates": 0},
        {"acquisition": "unknown"},
        {"xi": -1},
        {"kappa": float("nan")},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        OptimisationConfig(**settings)


def test_history_and_budget():
    initial = Observations(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([1., 2.]), 2)
    optimizer = BayesianOptimizer(initial, OptimisationConfig(budget=1, candidates=32, seed=4))
    point = optimizer.suggest()
    assert point.shape == (2,)
    assert np.all((point >= 0) & (point < 1))
    assert format_submission(point).count("-") == 1
    history = optimizer.observe(point, 3.)
    assert optimizer.remaining == 0
    assert len(history.inputs) == 3
    np.testing.assert_array_equal(history.best_input, point)
    assert history.best_output == 3
    assert len(initial.inputs) == 2
    with pytest.raises(RuntimeError, match="budget"):
        optimizer.suggest()


def test_run_refits_using_all_observations(monkeypatch):
    from bayesian_bbo import optimisation

    sizes = []
    original_fit = optimisation.fit_surrogate

    def tracked_fit(observations, seed):
        sizes.append(len(observations.inputs))
        return original_fit(observations, seed)

    monkeypatch.setattr(optimisation, "fit_surrogate", tracked_fit)
    initial = Observations(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0., 1.]), 2)
    optimizer = BayesianOptimizer(initial, OptimisationConfig(budget=2, candidates=16, acquisition="ucb", seed=1))
    history = optimizer.run(lambda point: float(np.sum(point)))
    assert sizes == [2, 3]
    assert len(history.outputs) == 4


def test_acquisition_zero_uncertainty():
    assert expected_improvement(np.array([5.]), np.array([0.]), 1.)[0] == 0


@pytest.mark.parametrize("point", [[-0.1], [1.0], [0.9999999], [float("nan")], [[0.1]]])
def test_submission_rejects_invalid_points(point):
    with pytest.raises(ValueError):
        format_submission(point)


def test_submission_precision():
    assert format_submission([0., 0.1234564, 0.999999]) == "0.000000-0.123456-0.999999"
