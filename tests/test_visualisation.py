import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from unittest.mock import Mock  # noqa: E402

from bayesian_bbo.data import Observations  # noqa: E402
from bayesian_bbo.optimisation import BayesianOptimizer, OptimisationConfig  # noqa: E402
from bayesian_bbo.visualisation import (  # noqa: E402
    plot_calibration,
    plot_convergence,
    plot_gp_2d,
    plot_gp_slices,
    plot_optimum_progress,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def progress_arrays():
    previous = np.array([[1.0, 2.0], [-3.0, -2.0]])
    expected = np.array([[1.5, 2.5], [-2.5, -2.0]])
    observed = np.array([[1.0, 3.0], [-2.5, -1.0]])
    return previous, expected, observed


def test_convergence_plots_observed_running_best():
    observations = Observations(inputs=[[0.1], [0.2]], outputs=[0.5, 1.0], dimension=1)
    _, ax = plot_convergence(observations, 1)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.5, 1.0])


def test_optimum_progress_draws_three_series_per_function():
    previous, expected, observed = progress_arrays()
    _, axes = plot_optimum_progress([2, 3], [1, 2], previous, expected, observed)
    panels = axes.ravel()
    assert len(panels) == 2
    assert all(len(ax.lines) == 3 for ax in panels)


def test_optimum_progress_hides_unused_panels():
    previous = np.ones((3, 2))
    _, axes = plot_optimum_progress([2, 3], [1, 2, 3], previous, previous + 0.5, previous + 1)
    panels = axes.ravel()
    assert len(panels) == 4
    assert not panels[3].get_visible()


def test_optimum_progress_rejects_expected_below_previous():
    previous, expected, observed = progress_arrays()
    expected[0, 0] = 0.5
    with pytest.raises(ValueError, match="Expected best"):
        plot_optimum_progress([2, 3], [1, 2], previous, expected, observed)


def test_optimum_progress_rejects_observed_below_previous():
    previous, expected, observed = progress_arrays()
    observed[1, 1] = -5.0
    with pytest.raises(ValueError, match="Observed best"):
        plot_optimum_progress([2, 3], [1, 2], previous, expected, observed)


def test_optimum_progress_rejects_wrong_shape():
    previous, expected, observed = progress_arrays()
    with pytest.raises(ValueError, match="shape"):
        plot_optimum_progress([2], [1, 2], previous, expected, observed)


def test_calibration_plots_each_function():
    _, ax = plot_calibration({1: [0.1, -1.2], 2: [0.5]})
    assert len(ax.collections) == 2


def test_calibration_rejects_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        plot_calibration({1: [np.nan]})


def diagnostic_optimizer(dimension):
    observations = Observations(
        inputs=np.full((1, dimension), 0.1), outputs=[1.0], dimension=dimension,
    )
    optimizer = BayesianOptimizer(observations, OptimisationConfig(budget=1))

    def predict(points):
        mean = np.sum(points, axis=1)
        return mean, np.full(len(points), 0.25), mean + 0.5

    optimizer.predict = Mock(side_effect=predict)
    optimizer.suggest = Mock(side_effect=AssertionError("Plotting must not suggest a new point"))
    return optimizer


def test_gp_2d_marks_exact_stored_candidate_with_saved_predictions():
    optimizer = diagnostic_optimizer(2)
    candidate = np.array([0.37, 0.63])
    _, axes = plot_gp_2d(optimizer, n=5, candidate=candidate, expected_mean=9.0, expected_std=2.0)
    for ax in axes:
        np.testing.assert_allclose(ax.collections[-1].get_offsets(), candidate[None, :])
        assert "Grid EI max" not in ax.get_legend_handles_labels()[1]
    assert "Saved mean: 9" in axes[0].collections[-1].get_label()
    assert "Saved std: 2" in axes[1].collections[-1].get_label()
    assert "Model EI: 1.5" in axes[2].collections[-1].get_label()
    np.testing.assert_array_equal(optimizer.predict.call_args_list[0].args[0], candidate[None, :])
    assert optimizer.remaining == 1


def test_gp_slices_pass_through_candidate_and_plot_saved_prediction():
    optimizer = diagnostic_optimizer(3)
    candidate = np.array([0.37, 0.63, 0.42])
    _, axes = plot_gp_slices(optimizer, n=5, candidate=candidate, expected_mean=9.0, expected_std=2.0)
    for j in range(3):
        points = optimizer.predict.call_args_list[j + 1].args[0]
        other = np.arange(3) != j
        np.testing.assert_allclose(points[:, other], np.tile(candidate[other], (5, 1)))
        top, bottom = axes[:, j]
        errorbar = top.containers[0]
        np.testing.assert_allclose(errorbar.lines[0].get_xdata(orig=False), [candidate[j]])
        np.testing.assert_allclose(errorbar.lines[0].get_ydata(orig=False), [9.0])
        np.testing.assert_allclose(
            errorbar.lines[2][0].get_segments()[0],
            [[candidate[j], 9.0 - 1.96 * 2.0], [candidate[j], 9.0 + 1.96 * 2.0]],
        )
        np.testing.assert_allclose(bottom.collections[0].get_offsets(), [[candidate[j], 1.92]])
        assert "Slice EI max" not in bottom.get_legend_handles_labels()[1]


@pytest.mark.parametrize("ei_level", [0.0, 0.025, 287.0])
@pytest.mark.parametrize("stored", [False, True])
def test_gp_slices_keep_flat_ei_and_candidate_visible(ei_level, stored):
    optimizer = diagnostic_optimizer(3)

    def predict(points):
        return np.zeros(len(points)), np.ones(len(points)), np.full(len(points), ei_level)

    optimizer.predict = Mock(side_effect=predict)
    kwargs = {"candidate": np.array([0.0, 0.5, 0.99])} if stored else {}
    _, axes = plot_gp_slices(optimizer, n=5, **kwargs)
    for top, bottom in axes.T:
        low, high = bottom.get_ylim()
        assert low == 0
        assert high >= 1.09 * ei_level
        assert high > 0
        marker = bottom.collections[0]
        assert marker.get_zorder() > bottom.lines[-1].get_zorder()
        assert not marker.get_clip_on()
        if stored:
            assert top.containers[0].lines[0].get_zorder() > top.lines[-1].get_zorder()
            assert "Stored candidate input" in bottom.get_legend_handles_labels()[1]


@pytest.mark.parametrize("stored", [False, True])
def test_gp_slices_share_y_scales_and_include_all_slices(stored):
    optimizer = diagnostic_optimizer(3)

    def predict(points):
        mean = points @ np.array([-10.0, 2.0, 1.0])
        return mean, np.full(len(points), 0.25), np.abs(mean)

    optimizer.predict = Mock(side_effect=predict)
    kwargs = (
        {"candidate": [0.3, 0.6, 0.4], "expected_mean": 30.0, "expected_std": 2.0}
        if stored else {}
    )
    figure, axes = plot_gp_slices(optimizer, n=5, **kwargs)
    figure.canvas.draw()
    for row in axes:
        for ax in row:
            assert ax.get_shared_y_axes().joined(row[0], ax)
            np.testing.assert_allclose(ax.get_ylim(), row[0].get_ylim())
    top_low, top_high = axes[0, 0].get_ylim()
    max_ei = 0.0
    for top, bottom in axes.T:
        mean = top.lines[0].get_ydata()
        assert top_low < np.min(mean) - 1.96 * 0.25
        assert top_high > np.max(mean) + 1.96 * 0.25
        max_ei = max(max_ei, float(np.max(bottom.lines[0].get_ydata())))
    low, high = axes[1, 0].get_ylim()
    assert low == 0
    assert high >= 1.09 * max_ei
    if stored:
        assert top_high > 30.0 + 1.96 * 2.0
    assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])


@pytest.mark.parametrize("plot,dimension", [(plot_gp_2d, 2), (plot_gp_slices, 3)])
@pytest.mark.parametrize("expected_ei", [0.0, 40.0])
def test_gp_saved_ei_skips_candidate_prediction_and_marks_saved_score(plot, dimension, expected_ei):
    optimizer = diagnostic_optimizer(dimension)
    candidate = np.full(dimension, 0.3)
    _, axes = plot(
        optimizer, n=5, candidate=candidate,
        expected_mean=9.0, expected_std=2.0, expected_ei=expected_ei,
    )
    assert isinstance(optimizer.predict, Mock)
    assert optimizer.predict.call_count == (1 if dimension == 2 else dimension)
    assert all(call.args[0].shape[0] > 1 for call in optimizer.predict.call_args_list)
    if dimension == 2:
        assert f"Saved EI: {expected_ei:g}" in axes[2].collections[-1].get_label()
    else:
        for coordinate, bottom in zip(candidate, axes[1]):
            np.testing.assert_allclose(bottom.collections[0].get_offsets(), [[coordinate, expected_ei]])
            assert bottom.collections[0].get_label() == "Saved candidate EI"
            assert bottom.get_ylim()[1] > expected_ei


@pytest.mark.parametrize("plot,dimension", [(plot_gp_2d, 2), (plot_gp_slices, 3)])
def test_gp_saved_ei_can_use_model_mean_and_std(plot, dimension):
    optimizer = diagnostic_optimizer(dimension)
    _, axes = plot(optimizer, n=5, candidate=np.full(dimension, 0.3), expected_ei=40.0)
    assert isinstance(optimizer.predict, Mock)
    assert optimizer.predict.call_count == (2 if dimension == 2 else dimension + 1)
    if dimension == 2:
        assert "Predicted mean: 0.6" in axes[0].collections[-1].get_label()
        assert "Saved EI: 40" in axes[2].collections[-1].get_label()
    else:
        np.testing.assert_allclose(axes[0, 0].containers[0].lines[0].get_ydata(orig=False), [0.9])
        assert axes[1, 0].collections[0].get_offsets()[0, 1] == 40.0


@pytest.mark.parametrize("plot,dimension", [(plot_gp_2d, 2), (plot_gp_slices, 3)])
def test_gp_candidate_can_use_model_predictions(plot, dimension):
    optimizer = diagnostic_optimizer(dimension)
    candidate = np.full(dimension, 0.3)
    _, axes = plot(optimizer, n=5, candidate=candidate)
    if dimension == 2:
        assert "Predicted mean: 0.6" in axes[0].collections[-1].get_label()
        assert "Predicted std: 0.25" in axes[1].collections[-1].get_label()
    else:
        np.testing.assert_allclose(axes[0, 0].containers[0].lines[0].get_ydata(orig=False), [0.9])


@pytest.mark.parametrize("plot,dimension", [(plot_gp_2d, 2), (plot_gp_slices, 3)])
def test_gp_default_keeps_grid_or_slice_maximum(plot, dimension):
    optimizer = diagnostic_optimizer(dimension)
    _, axes = plot(optimizer, n=5)
    if dimension == 2:
        assert "Grid EI max" in axes[2].get_legend_handles_labels()[1]
    else:
        assert "Slice EI max" in axes[1, 0].get_legend_handles_labels()[1]
        points = optimizer.predict.call_args_list[0].args[0]
        np.testing.assert_allclose(points[:, 1:], 0.1)


@pytest.mark.parametrize("plot,dimension", [(plot_gp_2d, 2), (plot_gp_slices, 3)])
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"expected_mean": 1.0, "expected_std": 0.2}, "require candidate"),
        ({"expected_ei": 0.1}, "require candidate"),
        ({"candidate": [0.2]}, "one finite coordinate"),
        ({"candidate": [np.nan, 0.2]}, "one finite coordinate"),
        ({"candidate": [1.0, 0.2]}, "one finite coordinate"),
        ({"expected_mean": 1.0}, "both saved"),
        ({"expected_mean": 1.0, "expected_std": -0.2}, "nonnegative"),
        ({"expected_mean": np.inf, "expected_std": 0.2}, "finite"),
        ({"expected_mean": 1.0, "expected_std": np.nan}, "finite"),
        ({"expected_ei": -0.1}, "nonnegative"),
        ({"expected_ei": np.nan}, "finite"),
        ({"expected_ei": np.inf}, "finite"),
    ],
)
def test_gp_saved_candidate_validation(plot, dimension, kwargs, message):
    optimizer = diagnostic_optimizer(dimension)
    if message not in {"require candidate", "one finite coordinate"}:
        kwargs = {"candidate": np.full(dimension, 0.3), **kwargs}
    with pytest.raises(ValueError, match=message):
        plot(optimizer, n=5, **kwargs)
    optimizer.predict.assert_not_called()
