import numpy as np
import pytest

from bayesian_bbo.acquisition import expected_improvement
from bayesian_bbo import CONFIG
from bayesian_bbo.data import Observations
from bayesian_bbo.optimisation import BayesianOptimizer, OptimisationConfig, remaining_budget
from bayesian_bbo.settings import CONFIG_PATH, load_config
from bayesian_bbo.submission import format_submission
from bayesian_bbo.surrogate import fit_surrogate
from sklearn.gaussian_process.kernels import RBF


def test_shared_config_loads_yaml_values(tmp_path):
    assert CONFIG_PATH.is_file()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "normalize_y: false\nn_restarts_optimizer: 3\nrandom_state: 7\ncandidates: 16\n",
        encoding="utf-8",
    )

    assert load_config(config_path) == {
        "normalize_y": False,
        "n_restarts_optimizer": 3,
        "random_state": 7,
        "candidates": 16,
    }
    assert CONFIG == {
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "random_state": 42,
        "candidates": 2048,
    }


def test_optimizer_and_surrogate_defaults_use_shared_config():
    observations = Observations(np.array([[0.1], [0.3]]), np.array([0., 1.]), 1)
    optimizer = BayesianOptimizer(observations)
    model = fit_surrogate(
        observations,
        kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
    )

    assert optimizer.config.seed == CONFIG["random_state"]
    assert optimizer.config.candidates == CONFIG["candidates"]
    assert model.normalize_y is CONFIG["normalize_y"]
    assert model.n_restarts_optimizer == CONFIG["n_restarts_optimizer"]


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

    def tracked_fit(observations, seed, kernel=None):
        sizes.append(len(observations.inputs))
        return original_fit(observations, seed, kernel)

    monkeypatch.setattr(optimisation, "fit_surrogate", tracked_fit)
    initial = Observations(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0., 1.]), 2)
    optimizer = BayesianOptimizer(initial, OptimisationConfig(budget=2, candidates=16, acquisition="ucb", seed=1))
    history = optimizer.run(lambda point: float(np.sum(point)))
    assert sizes == [2, 3]
    assert len(history.outputs) == 4


def test_custom_kernel_can_be_injected():
    observations = Observations(np.array([[0.1], [0.3]]), np.array([0., 1.]), 1)

    model = fit_surrogate(observations, kernel=lambda dimension: RBF(length_scale=np.ones(dimension)))
    assert isinstance(model.kernel_, RBF)

    optimizer = BayesianOptimizer(
        observations,
        OptimisationConfig(budget=1, candidates=16, seed=1, kernel=RBF(length_scale=0.2)),
    )
    assert optimizer.suggest().shape == (1,)


def test_suggest_with_prediction_returns_selected_point_and_prediction():
    observations = Observations(np.array([[0.1], [0.3]]), np.array([0., 1.]), 1)
    optimizer = BayesianOptimizer(
        observations,
        OptimisationConfig(
            budget=1,
            candidates=32,
            seed=1,
            acquisition="ucb",
            kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
        ),
    )

    point, mean, std = optimizer.suggest_with_prediction()

    assert point.shape == (1,)
    assert 0 <= point[0] < 1
    assert np.isfinite(mean)
    assert np.isfinite(std) and std >= 0


def test_config_accepts_finite_expected_improvement_threshold():
    assert OptimisationConfig(best=3.5).best == 3.5
    with pytest.raises(ValueError, match="Best must be finite"):
        OptimisationConfig(best=float("nan"))


def test_acquisition_zero_uncertainty():
    assert expected_improvement(np.array([5.]), np.array([0.]), 1.)[0] == 0


def test_remaining_budget_after_restart():
    x = np.arange(24).reshape(12, 2) / 100
    observations = Observations(x, np.arange(12), 2)
    assert remaining_budget(observations, 1) == 8
    assert remaining_budget(observations, 1, total_budget=2) == 0
    with pytest.raises(ValueError, match="match"):
        remaining_budget(observations, 3)


def test_remaining_budget_uses_actual_initial_count():
    observations = Observations(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([1., 2.]), 2)
    assert remaining_budget(observations, 1, total_budget=5) == 3


@pytest.mark.parametrize("point", [[-0.1], [1.0], [0.9999999], [float("nan")], [[0.1]]])
def test_submission_rejects_invalid_points(point):
    with pytest.raises(ValueError):
        format_submission(point)


def test_submission_precision():
    assert format_submission([0., 0.1234564, 0.999999]) == "0.000000-0.123456-0.999999"
