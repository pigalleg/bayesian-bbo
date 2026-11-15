import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold

from bayesian_bbo.data import Observations
from bayesian_bbo.kernels import make_comparison_kernels
from bayesian_bbo.validation import cross_validate_kernels, evaluate_kernel_splits


@pytest.mark.parametrize("dimension", [2, 2, 3, 4, 4, 5, 6, 8])
def test_cross_validation_supports_function_input_dimensions(dimension):
    random = np.random.default_rng(dimension)
    inputs = random.uniform(0.0, 1.0, size=(6, dimension))
    outputs = np.sum(inputs**2, axis=1)
    kernel_factories = {
        "RBF": lambda: RBF(length_scale=np.ones(dimension)),
    }

    results = cross_validate_kernels(
        inputs,
        outputs,
        kernel_factories,
        n_splits=3,
        n_restarts_optimizer=0,
    )

    assert list(results.columns) == [
        "kernel",
        "fold",
        "RMSE",
        "NRMSE",
        "MAE",
        "negative_log_predictive_density",
        "coverage_90",
    ]
    assert results["fold"].tolist() == [1, 2, 3]
    assert results["kernel"].tolist() == ["RBF"] * 3
    np.testing.assert_allclose(results["NRMSE"], results["RMSE"] / np.std(outputs))
    assert np.isfinite(results.select_dtypes(include="number")).all().all()


def test_cross_validation_rejects_mismatched_outputs():
    with pytest.raises(ValueError, match="one scalar output per row"):
        cross_validate_kernels(
            np.ones((4, 2)),
            np.ones(3),
            {"RBF": lambda: RBF()},
            n_splits=2,
            n_restarts_optimizer=0,
        )


def diagnostic_observations():
    return Observations(
        np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]),
        np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]),
        dimension=1,
        weeks=np.array([0, 0, 0, 1, 2, 3]),
    )


def test_detailed_predictions_share_folds_and_cover_every_row_once():
    observations = diagnostic_observations()
    splits = dict(enumerate(KFold(3, shuffle=True, random_state=42).split(observations.inputs), 1))
    kernels = {"RBF": RBF(0.3, length_scale_bounds="fixed")}
    results = evaluate_kernel_splits(observations, kernels, splits, n_restarts_optimizer=0)
    for name, rows in results.groupby("kernel"):
        assert sorted(rows.row_index) == list(range(6))
    baseline = results.loc[results.kernel == "Training mean"].sort_values("row_index")
    gp = results.loc[results.kernel == "RBF"].sort_values("row_index")
    np.testing.assert_array_equal(baseline["split"], gp["split"])
    np.testing.assert_allclose(results.width_95, 2 * 1.96 * results["std"])
    np.testing.assert_allclose(results.z, results.error / results.scoring_std)
    np.testing.assert_allclose(
        results.nlpd,
        np.log(results.scoring_std) + 0.5 * np.log(2 * np.pi) + 0.5 * results.z**2,
    )
    assert kernels["RBF"].length_scale == 0.3


def test_historical_splits_fit_only_past_rows_and_baseline_uses_training_stats(monkeypatch):
    observations = diagnostic_observations()
    trained = []

    class Model:
        kernel_ = RBF(1.0)

        def predict(self, inputs, return_std):
            assert return_std
            return np.full(len(inputs), 4.0), np.full(len(inputs), 0.5)

    def fit(training, **kwargs):
        trained.append(training.outputs.copy())
        return Model()

    monkeypatch.setattr("bayesian_bbo.validation.fit_surrogate", fit)
    splits = {
        week: (np.flatnonzero(observations.weeks < week), np.flatnonzero(observations.weeks == week))
        for week in (1, 2, 3)
    }
    results = evaluate_kernel_splits(observations, {"RBF": RBF()}, splits)
    assert (results.training_max_week < results.week).all()
    for index, week in enumerate((1, 2, 3)):
        training_y = observations.outputs[observations.weeks < week]
        np.testing.assert_array_equal(trained[index], training_y)
        baseline = results.loc[(results.kernel == "Training mean") & (results.week == week)].iloc[0]
        assert baseline["mean"] == pytest.approx(training_y.mean())
        assert baseline["std"] == pytest.approx(training_y.std())
        assert baseline.actual == observations.outputs[observations.weeks == week][0]


@pytest.mark.parametrize("train, test, message", [
    ([0, 1], [1, 2], "overlap"),
    ([], [2], "training"),
    ([0, 1], [], "validation"),
    ([-1, 0], [2], "training"),
    ([0, 1], [6], "validation"),
    ([0, 0], [2], "training"),
    ([0.0, 1.0], [2], "training"),
])
def test_detailed_predictions_reject_invalid_splits(train, test, message):
    with pytest.raises(ValueError, match=message):
        evaluate_kernel_splits(diagnostic_observations(), {"RBF": RBF()}, {1: (train, test)})


def test_comparison_kernels_match_existing_candidates():
    kernels = make_comparison_kernels(4, 100.0)
    assert list(kernels) == ["Matérn 3/2", "Matérn 5/2", "RBF", "Rational Quadratic", "Exp-Sine-Squared"]
    assert kernels["Matérn 3/2"].k1.k2.nu == 1.5
    assert kernels["Matérn 5/2"].k1.k2.nu == 2.5
    for kernel in kernels.values():
        assert kernel.k1.k2.length_scale_bounds == (1e-3, 100.0)
        assert kernel.k2.noise_level_bounds == (1e-8, 1e1)


@pytest.mark.parametrize("std", [-1.0, np.nan, np.inf])
def test_detailed_predictions_reject_invalid_predictive_std(monkeypatch, std):
    class Model:
        kernel_ = RBF()

        def predict(self, inputs, return_std):
            return np.ones(len(inputs)), np.full(len(inputs), std)

    monkeypatch.setattr("bayesian_bbo.validation.fit_surrogate", lambda *args, **kwargs: Model())
    with pytest.raises(ValueError, match="invalid predictive"):
        evaluate_kernel_splits(
            diagnostic_observations(), {"RBF": RBF()}, {1: ([0, 1], [2])},
        )


def test_zero_std_scoring_floor_does_not_widen_intervals(monkeypatch):
    class Model:
        kernel_ = RBF()

        def predict(self, inputs, return_std):
            return np.ones(len(inputs)), np.zeros(len(inputs))

    monkeypatch.setattr("bayesian_bbo.validation.fit_surrogate", lambda *args, **kwargs: Model())
    results = evaluate_kernel_splits(
        diagnostic_observations(), {"RBF": RBF()}, {1: ([0, 1], [2])},
    )
    gp = results.loc[results.kernel == "RBF"].iloc[0]
    assert gp["std"] == 0
    assert gp.scoring_std == 1e-9
    assert gp.lower_95 == gp.upper_95 == gp["mean"]
    assert not gp.covered_95
    assert np.isfinite(gp.nlpd)