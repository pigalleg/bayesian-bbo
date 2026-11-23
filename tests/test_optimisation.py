import numpy as np
import pytest
import yaml

from bayesian_bbo.acquisition import expected_improvement, log_expected_improvement
from bayesian_bbo import CONFIG
from bayesian_bbo.data import Observations
from bayesian_bbo.optimisation import BayesianOptimizer, OptimisationConfig, find_prediction_regions, remaining_budget
from bayesian_bbo.settings import CONFIG_PATH, load_config
from bayesian_bbo.submission import format_submission
from bayesian_bbo.surrogate import fit_surrogate
from sklearn.gaussian_process.kernels import RBF


def test_prediction_regions_are_ranked_separated_and_reproducible():
    from scipy.stats import qmc

    calls = []

    def predict(points):
        calls.append(len(points))
        return points[:, 0] + 2 * points[:, 1]

    settings = dict(candidate_power=7, region_count=3, min_distance=0.4, batch_size=17, seed=9)
    points, scores = find_prediction_regions(predict, 2, **settings)
    assert max(calls) <= 17
    assert sum(calls) == 128
    pool = qmc.Sobol(d=2, scramble=True, seed=9).random_base2(7)
    pool_scores = pool[:, 0] + 2 * pool[:, 1]
    np.testing.assert_array_equal(points[0], pool[pool_scores.argmax()])
    np.testing.assert_allclose(scores, points[:, 0] + 2 * points[:, 1])
    assert len(points) == 3
    assert np.all(np.diff(scores) <= 0)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    assert np.all(distances[np.triu_indices(3, 1)] >= 0.4)
    repeated_points, repeated_scores = find_prediction_regions(predict, 2, **settings)
    np.testing.assert_array_equal(points, repeated_points)
    np.testing.assert_array_equal(scores, repeated_scores)


def test_prediction_regions_allow_fewer_representatives():
    points, scores = find_prediction_regions(
        lambda points: points[:, 0], 1, candidate_power=3, min_distance=2,
    )
    assert points.shape == (1, 1)
    assert scores.shape == (1,)


@pytest.mark.parametrize("settings", [{"dimension": 0}, {"candidate_power": -1},
                                     {"region_count": 0}, {"batch_size": 0},
                                     {"min_distance": 0}, {"min_distance": float("nan")}])
def test_prediction_regions_reject_invalid_settings(settings):
    with pytest.raises(ValueError):
        find_prediction_regions(lambda points: points[:, 0], **{"dimension": 2, **settings})


@pytest.mark.parametrize("predict", [lambda points: np.zeros((len(points), 1)),
                                    lambda points: np.full(len(points), np.nan)])
def test_prediction_regions_reject_invalid_predictions(predict):
    with pytest.raises(ValueError, match="finite scalar"):
        find_prediction_regions(predict, 2, candidate_power=2)


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


@pytest.mark.parametrize(
    "settings",
    [
        {"search_strategy": "unknown"},
        {"lbfgs_restarts": 0},
        {"lbfgs_restarts": True},
        {"local_candidates": -1},
        {"local_candidates": True},
    ],
)
def test_invalid_search_config(settings):
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
    assert optimizer.config.search_strategy == "random"
    np.testing.assert_array_equal(optimizer._candidate_pool(), pool[2:])


def test_sobol_lbfgs_pool_and_suggestion_are_reproducible():
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    config = OptimisationConfig(
        candidates=17, seed=4, search_strategy="sobol_lbfgs", local_candidates=12,
        lbfgs_restarts=3, kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
    )
    first = BayesianOptimizer(observations, config)
    second = BayesianOptimizer(observations, config)
    np.testing.assert_array_equal(first._candidate_pool(), second._candidate_pool())
    first = BayesianOptimizer(observations, config).suggest_with_prediction()
    second = BayesianOptimizer(observations, config).suggest_with_prediction()
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_sobol_lbfgs_candidates_are_snapped_bounded_and_unused():
    observations = Observations([[0.1], [0.3]], [0., 1.], 1)
    optimizer = BayesianOptimizer(
        observations,
        OptimisationConfig(
            candidates=32, seed=7, search_strategy="sobol_lbfgs", local_candidates=20,
            lbfgs_restarts=4, kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
        ),
    )
    pool = optimizer._candidate_pool()
    assert np.all((pool >= 0.0) & (pool < 1.0))
    np.testing.assert_array_equal(pool, np.round(pool, 6))
    assert not np.any(np.all(pool[:, None] == observations.inputs[None, :, :], axis=2))

    point, _, _, _ = optimizer.suggest_with_prediction()
    assert np.all((point >= 0.0) & (point < 1.0))
    np.testing.assert_array_equal(point, np.round(point, 6))
    assert not np.any(np.all(observations.inputs == point, axis=1))


def test_sobol_lbfgs_polishing_matches_or_improves_stage_one_ei():
    observations = Observations(
        [[0.05], [0.25], [0.5], [0.75], [0.95]],
        [0.05, 0.45, 0.9, 0.45, 0.05],
        1,
    )
    config = OptimisationConfig(
        candidates=64, seed=13, search_strategy="sobol_lbfgs", local_candidates=0,
        lbfgs_restarts=4, kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
    )
    optimizer = BayesianOptimizer(observations, config)
    pool = optimizer._candidate_pool()
    _, _, stage_one_scores = optimizer.predict(pool)

    point, mean, std, score = optimizer.suggest_with_prediction()

    assert score >= np.max(stage_one_scores) - 1e-12
    assert score == pytest.approx(expected_improvement(
        np.array([mean]), np.array([std]), observations.best_output, config.xi,
    )[0])
    assert not np.any(np.all(observations.inputs == point, axis=1))


@pytest.mark.parametrize("acquisition", ["ei", "ucb"])
def test_sobol_lbfgs_suggest_supports_ei_and_ucb(acquisition):
    observations = Observations([[0.1], [0.3], [0.7]], [0., 1., 0.2], 1)
    config = OptimisationConfig(
        candidates=32, seed=5, acquisition=acquisition, search_strategy="sobol_lbfgs",
        local_candidates=8, lbfgs_restarts=3,
        kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
    )
    optimizer = BayesianOptimizer(observations, config)

    point, mean, std, score = optimizer.suggest_with_prediction()

    assert point.shape == (1,)
    assert 0.0 <= point[0] < 1.0
    np.testing.assert_array_equal(point, np.round(point, 6))
    assert not np.any(np.all(observations.inputs == point, axis=1))
    if acquisition == "ei":
        assert score == pytest.approx(expected_improvement(
            np.array([mean]), np.array([std]), observations.best_output, config.xi,
        )[0])
    else:
        assert score == pytest.approx(mean + config.kappa * std)


def test_sobol_lbfgs_ei_sweep_polishes_each_xi():
    from bayesian_bbo import optimisation

    observations = Observations([[0.1], [0.3], [0.7]], [0., 1., 0.2], 1)
    fits = []
    original_fit = optimisation.fit_surrogate

    def tracked_fit(observations, seed, kernel):
        fits.append(len(observations.inputs))
        return original_fit(observations, seed, kernel)

    optimizer = BayesianOptimizer(
        observations,
        OptimisationConfig(
            candidates=32, seed=5, search_strategy="sobol_lbfgs", local_candidates=0,
            lbfgs_restarts=3, kernel=RBF(length_scale=0.2, length_scale_bounds="fixed"),
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(optimisation, "fit_surrogate", tracked_fit)
        suggestions = optimizer.suggest_ei_sweep([0.0, 0.1, 0.3], batch_size=7)

    assert len(suggestions) == 3
    assert fits == [3]
    for (point, mean, std, score), xi in zip(suggestions, [0.0, 0.1, 0.3]):
        assert 0.0 <= point[0] < 1.0
        np.testing.assert_array_equal(point, np.round(point, 6))
        assert not np.any(np.all(observations.inputs == point, axis=1))
        assert score == pytest.approx(expected_improvement(
            np.array([mean]), np.array([std]), observations.best_output, xi,
        )[0])


def test_log_expected_improvement_matches_ei_ranking_and_stays_finite_in_tail():
    z = np.array([-8.0, -5.0, -2.0, 0.0, 1.0, 3.0])
    std = np.ones_like(z)
    mean = z + 0.01
    raw = expected_improvement(mean, std, 0.0, xi=0.01)
    logged = log_expected_improvement(mean, std, 0.0, xi=0.01)
    np.testing.assert_array_equal(np.argsort(raw), np.argsort(logged))
    np.testing.assert_allclose(np.exp(logged[1:]), raw[1:], rtol=1e-12, atol=1e-30)

    extreme_mean = np.array([-40.0, -1000.0])
    extreme_raw = expected_improvement(extreme_mean, np.ones(2), 0.0, xi=0.0)
    extreme_log = log_expected_improvement(extreme_mean, np.ones(2), 0.0, xi=0.0)
    np.testing.assert_array_equal(extreme_raw, np.zeros(2))
    assert np.all(np.isfinite(extreme_log))
    assert extreme_log[0] > extreme_log[1]


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
