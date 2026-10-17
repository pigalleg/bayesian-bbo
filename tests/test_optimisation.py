import numpy as np
import pytest
import yaml

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
        "normalize_y: false\nn_restarts_optimizer: 3\nrandom_state: 7\ncandidates: 16\n"
        + yaml.safe_dump({"length_caps": dict.fromkeys(range(1, 9), 5.0)}),
        encoding="utf-8",
    )

    assert load_config(config_path) == {
        "normalize_y": False,
        "n_restarts_optimizer": 3,
        "random_state": 7,
        "candidates": 16,
        "length_caps": dict.fromkeys(range(1, 9), 5.0),
    }
    assert CONFIG == {
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "random_state": 42,
        "candidates": 1000000,
        "length_caps": {
            1: 31.622776601683793, 2: 31.622776601683793, 3: 31.622776601683793,
            4: 10.0, 5: 100.0, 6: 31.622776601683793, 7: 31.622776601683793, 8: 100.0,
        },
    }


@pytest.mark.parametrize(
    "caps",
    [
        None, [], {}, {1: 5.0},
        {**dict.fromkeys(range(1, 9), 5.0), "1": 5.0},
        {**dict.fromkeys(range(1, 9), 5.0), 0: 5.0},
        {True: 5.0, **dict.fromkeys(range(2, 9), 5.0)},
        *[{**dict.fromkeys(range(1, 9), 5.0), 1: cap}
          for cap in (0, -1, float("inf"), float("nan"), True, "31.6")],
    ],
)
def test_shared_config_rejects_invalid_length_caps(tmp_path, caps):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({**CONFIG, "length_caps": caps}), encoding="utf-8")
    with pytest.raises(ValueError, match="length_caps"):
        load_config(config_path)


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

    point, mean, std, score = optimizer.suggest_with_prediction()

    assert point.shape == (1,)
    assert 0 <= point[0] < 1
    assert np.isfinite(mean)
    assert np.isfinite(std) and std >= 0
    assert score == pytest.approx(mean + optimizer.config.kappa * std)


def test_config_accepts_finite_expected_improvement_threshold():
    assert OptimisationConfig(best=3.5).best == 3.5
    with pytest.raises(ValueError, match="Best must be finite"):
        OptimisationConfig(best=float("nan"))


@pytest.mark.parametrize("best", [None, 0.0, 1.5])
def test_acquisition_settings_resolves_incumbent_and_tracks_observations(best):
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    optimizer = BayesianOptimizer(observations, OptimisationConfig(budget=1, best=best, xi=0.2))
    settings = optimizer.acquisition_settings
    assert settings == {"name": "ei", "xi": 0.2, "best": 1.0 if best is None else best}
    optimizer.observe(np.array([0.5]), 2.0)
    assert optimizer.acquisition_settings == {
        "name": "ei", "xi": 0.2, "best": 2.0 if best is None else best,
    }
    assert settings["best"] == (1.0 if best is None else best)
    settings["xi"] = 99.0
    assert optimizer.acquisition_settings["xi"] == 0.2


def test_acquisition_settings_tracks_configuration_without_fitting_or_spending_budget(monkeypatch):
    from bayesian_bbo import optimisation

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Reading acquisition settings must not fit a model")

    monkeypatch.setattr(optimisation, "fit_surrogate", unexpected_fit)
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    optimizer = BayesianOptimizer(observations, OptimisationConfig(budget=1))
    assert optimizer.acquisition_settings == {"name": "ei", "xi": 0.01, "best": 1.0}
    optimizer.config = OptimisationConfig(budget=1, acquisition="ucb", kappa=1.96)
    assert optimizer.acquisition_settings == {"name": "ucb", "kappa": 1.96}
    assert optimizer.remaining == 1


@pytest.mark.parametrize("acquisition", ["ei", "ucb"])
def test_predict_and_suggest_share_predictions_and_scores(acquisition):
    from bayesian_bbo.acquisition import upper_confidence_bound

    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    config = OptimisationConfig(
        budget=1, candidates=32, seed=1, acquisition=acquisition, best=1.5, xi=0.2,
        kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
    )
    optimizer = BayesianOptimizer(observations, config)
    pool = np.random.default_rng(config.seed).integers(
        0, 1_000_000, size=(config.candidates, 1),
    ) / 1_000_000
    mean, std, scores = optimizer.predict(pool)
    expected = (
        expected_improvement(mean, std, config.best, config.xi)
        if acquisition == "ei"
        else upper_confidence_bound(mean, std, config.kappa)
    )
    np.testing.assert_allclose(scores, expected)
    point, selected_mean, selected_std, selected_score = optimizer.suggest_with_prediction()
    selected = np.argmax(scores)
    np.testing.assert_array_equal(point, pool[selected])
    assert selected_mean == mean[selected]
    assert selected_std == std[selected]
    assert selected_score == scores[selected]
    assert optimizer.remaining == 1


def test_predict_reuses_model_and_refits_after_observation_or_config_change(monkeypatch):
    from bayesian_bbo import optimisation

    fits = []
    original_fit = optimisation.fit_surrogate

    def tracked_fit(observations, seed, kernel):
        fits.append(len(observations.outputs))
        return original_fit(observations, seed, kernel)

    monkeypatch.setattr(optimisation, "fit_surrogate", tracked_fit)
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    config = OptimisationConfig(
        budget=1, candidates=16, kernel=RBF(length_scale_bounds="fixed"),
    )
    optimizer = BayesianOptimizer(observations, config)
    optimizer.predict(np.array([[0.5]]))
    point, _, _, _ = optimizer.suggest_with_prediction()
    assert fits == [2]
    optimizer.observe(point, 2.)
    optimizer.predict(np.array([[0.5]]))
    assert fits == [2, 3]
    optimizer.config = OptimisationConfig(
        kernel=RBF(length_scale=0.5, length_scale_bounds="fixed"),
    )
    optimizer.predict(np.array([[0.5]]))
    assert fits == [2, 3, 3]


def test_acquisition_zero_uncertainty():
    assert expected_improvement(np.array([5.]), np.array([0.]), 1.)[0] == 0


def test_ei_sweep_matches_individual_suggestions_and_fits_once(monkeypatch):
    from bayesian_bbo import optimisation

    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    kernel = RBF(length_scale=0.2, length_scale_bounds="fixed")
    xis = [0.3, 0., 0.1, 0.3]
    config = OptimisationConfig(budget=1, candidates=32, seed=4, kernel=kernel, best=1.5)
    expected = [
        BayesianOptimizer(observations, OptimisationConfig(
            budget=1, candidates=32, seed=4, kernel=kernel, best=1.5, xi=xi,
        )).suggest_with_prediction()
        for xi in xis
    ]
    fits = []
    original_fit = optimisation.fit_surrogate

    def tracked_fit(observations, seed, kernel):
        fits.append(len(observations.outputs))
        return original_fit(observations, seed, kernel)

    monkeypatch.setattr(optimisation, "fit_surrogate", tracked_fit)
    optimizer = BayesianOptimizer(observations, config)
    actual = optimizer.suggest_ei_sweep(xis, batch_size=7)
    assert len(actual) == len(xis)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result[0], reference[0])
        np.testing.assert_allclose(result[1:], reference[1:], rtol=1e-10, atol=1e-12)
    assert fits == [2]
    assert optimizer.remaining == 1
    assert optimizer.history is observations
    assert optimizer.config is config
    random = np.random.default_rng(config.seed)
    random.integers(0, 1_000_000, size=(config.candidates, 1))
    np.testing.assert_array_equal(optimizer._random.integers(1000, size=8), random.integers(1000, size=8))


@pytest.mark.parametrize("xis", [[], [[0.1]], [-1], [float("nan")], [float("inf")]])
def test_ei_sweep_rejects_invalid_xis(xis):
    optimizer = BayesianOptimizer(Observations([[0.1], [0.3]], [0., 1.], 1))
    with pytest.raises(ValueError, match="xi_values"):
        optimizer.suggest_ei_sweep(xis)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_ei_sweep_rejects_invalid_batch_sizes(batch_size):
    optimizer = BayesianOptimizer(Observations([[0.1], [0.3]], [0., 1.], 1))
    with pytest.raises(ValueError, match="batch_size"):
        optimizer.suggest_ei_sweep([0.1], batch_size=batch_size)


def test_ei_sweep_requires_ei_and_remaining_budget():
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    optimizer = BayesianOptimizer(observations, OptimisationConfig(acquisition="ucb"))
    with pytest.raises(ValueError, match="acquisition='ei'"):
        optimizer.suggest_ei_sweep([0.1])
    optimizer = BayesianOptimizer(observations, OptimisationConfig(budget=0))
    with pytest.raises(RuntimeError, match="budget"):
        optimizer.suggest_ei_sweep([0.1])


def test_candidate_pool_excludes_observed_points():
    pool = np.random.default_rng(4).integers(0, 1_000_000, size=(8, 1)) / 1_000_000
    observations = Observations(pool[:2], [0., 1.], 1)
    optimizer = BayesianOptimizer(observations, OptimisationConfig(candidates=8, seed=4))
    np.testing.assert_array_equal(optimizer._candidate_pool(), pool[2:])


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
