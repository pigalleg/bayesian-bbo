"""Plot measured best-so-far values; never infer unmeasured outcomes."""

import math
from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import Observations
from .optimisation import BayesianOptimizer


def plot_prediction_planes(
    predict: Callable[[np.ndarray], np.ndarray],
    reference_points: np.ndarray,
    *,
    input_pair: tuple[int, int] = (0, 1),
    reference_labels: Sequence[str] | None = None,
    quantity_labels: Sequence[str] = ("Predicted output",),
    grid_size: int = 50,
):
    """Plot unit-cube conditional planes with one reference per row.

    predict returns shape (rows, quantities). Each quantity has a color scale
    shared across references. Stars mark reference coordinates, not outcomes.
    All coordinates outside input_pair remain fixed at each row's reference.
    """
    references = np.asarray(reference_points, dtype=float)
    if (references.ndim != 2 or not len(references) or references.shape[1] < 2
            or not np.all(np.isfinite(references)) or np.any((references < 0) | (references > 1))):
        raise ValueError("reference_points must be nonempty finite unit-cube rows with dimension >= 2")
    if (len(input_pair) != 2 or len(set(input_pair)) != 2
            or any(isinstance(index, bool) or not isinstance(index, int)
                   or not 0 <= index < references.shape[1] for index in input_pair)):
        raise ValueError("input_pair must contain two distinct valid input indices")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    labels = list(reference_labels) if reference_labels is not None else [
        f"Reference {index + 1}" for index in range(len(references))
    ]
    if len(labels) != len(references) or not len(quantity_labels):
        raise ValueError("Provide one reference label per row and at least one quantity label")
    horizontal, vertical = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    plane_points = np.repeat(references[:, None, :], grid_size ** 2, axis=1)
    plane_points[:, :, input_pair[0]] = horizontal.ravel()
    plane_points[:, :, input_pair[1]] = vertical.ravel()
    predictions = np.asarray(predict(plane_points.reshape(-1, references.shape[1])), dtype=float)
    expected_shape = (len(references) * grid_size ** 2, len(quantity_labels))
    if predictions.shape != expected_shape or not np.all(np.isfinite(predictions)):
        raise ValueError("predict must return finite values with shape (rows, quantities)")
    surfaces = predictions.reshape(len(references), grid_size, grid_size, len(quantity_labels))
    figure, axes = plt.subplots(
        len(references), len(quantity_labels), squeeze=False,
        figsize=(4.5 * len(quantity_labels), 3.4 * len(references)),
        sharex=True, sharey=True, layout="constrained",
    )
    for column, quantity in enumerate(quantity_labels):
        lower = float(surfaces[:, :, :, column].min())
        upper = float(surfaces[:, :, :, column].max())
        if lower == upper:
            padding = max(abs(lower), 1.0) * 1e-6
            lower, upper = lower - padding, upper + padding
        levels = np.linspace(lower, upper, 21)
        for row, reference in enumerate(references):
            axis = axes[row, column]
            contours = axis.contourf(horizontal, vertical, surfaces[row, :, :, column], levels=levels)
            axis.scatter(*reference[list(input_pair)], marker="*", s=110,
                         color="white", edgecolors="black", label="Reference")
            axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
            if row == 0:
                axis.set_title(quantity)
            if column == 0:
                axis.set_ylabel(f"{labels[row]}\nInput {input_pair[1] + 1}")
            if row == len(references) - 1:
                axis.set_xlabel(f"Input {input_pair[0] + 1}")
        figure.colorbar(contours, ax=list(axes[:, column]), label=quantity, shrink=0.8)
    figure.suptitle("Conditional planes: other inputs fixed at each reference; stars mark references")
    return figure, axes


def plot_convergence(observations: Observations, initial_count: int):
    if not 1 <= initial_count <= len(observations.outputs):
        raise ValueError("Initial count must be within the observation history")
    best = np.maximum.accumulate(observations.outputs)
    fig, ax = plt.subplots()
    ax.plot(np.arange(initial_count, len(best) + 1), best[initial_count - 1 :])
    ax.set(xlabel="Total evaluations", ylabel="Best observed output", title="Observed convergence")
    return fig, ax


def plot_optimum_progress(
    weeks: Sequence[int],
    function_ids: Sequence[int],
    actual_outputs: np.ndarray,
    expected_best: np.ndarray,
    observed_best: np.ndarray,
    *,
    expected_outputs: np.ndarray | None = None,
    stds: np.ndarray | None = None,
):
    """Plot actual outputs and expected/observed best-so-far per function.

    The arrays have shape (functions, weeks). Observed-best labels use six
    significant digits; actual outputs may decrease from one week to the next.
    Supply saved expected_outputs and stds together to draw the approximate
    Gaussian 95% predictive band (mean +/- 1.96 std), not a band around best-so-far.
    """
    if not weeks or not function_ids:
        raise ValueError("Provide at least one week and one function")
    shape = (len(function_ids), len(weeks))
    actual = np.asarray(actual_outputs, dtype=float)
    expected = np.asarray(expected_best, dtype=float)
    observed = np.asarray(observed_best, dtype=float)
    if {actual.shape, expected.shape, observed.shape} != {shape}:
        raise ValueError("Output arrays must have shape (functions, weeks)")
    if not np.all(np.isfinite(actual) & np.isfinite(expected) & np.isfinite(observed)):
        raise ValueError("Output values must be finite")
    tolerance = 1e-9 * (1 + np.abs(actual))
    if np.any(observed < actual - tolerance):
        raise ValueError("Observed best must not be below the actual output")
    if (expected_outputs is None) != (stds is None):
        raise ValueError("Provide expected outputs and standard deviations together")
    mean = uncertainty = None
    if expected_outputs is not None:
        mean = np.asarray(expected_outputs, dtype=float)
        uncertainty = np.asarray(stds, dtype=float)
        if mean.shape != shape or uncertainty.shape != shape:
            raise ValueError("Prediction arrays must have shape (functions, weeks)")
        if not np.all(np.isfinite(mean) & np.isfinite(uncertainty)) or np.any(uncertainty < 0):
            raise ValueError("Predictions must be finite and standard deviations nonnegative")

    rows = math.ceil(len(function_ids) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), squeeze=False)
    panels = axes.ravel()
    for index, (function_id, ax) in enumerate(zip(function_ids, panels)):
        ax.plot(
            weeks,
            actual[index],
            linestyle="--",
            color="gray",
            linewidth=2.5,
            marker=".",
            label="Actual output",
        )
        ax.plot(weeks, expected[index], marker="o", label="Expected best")
        observed_line, = ax.plot(weeks, observed[index], marker="s", label="Observed best")
        if mean is not None and uncertainty is not None:
            mean_line, = ax.plot(
                weeks, mean[index], linestyle=":", color="C2", label="Predicted mean",
            )
            ax.fill_between(
                weeks,
                mean[index] - 1.96 * uncertainty[index],
                mean[index] + 1.96 * uncertainty[index],
                color=mean_line.get_color(),
                alpha=0.18,
                label="Approx. 95% predictive band",
            )
        for week, value in zip(weeks, observed[index]):
            ax.annotate(
                f"{value:.6g}",
                xy=(week, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=observed_line.get_color(),
            )
        ax.set(title=f"Function {function_id}", xlabel="Week", ylabel="Output", xticks=list(weeks))
        ax.grid(True, alpha=0.3)
    for ax in panels[len(function_ids):]:
        ax.set_visible(False)
    panels[0].legend()
    figure.suptitle("Actual output and best-so-far: expected and observed")
    figure.tight_layout()
    return figure, axes


def plot_calibration(standardised_residuals: Mapping[int, Sequence[float]]):
    """Plot each function's standardised residuals on its own y-axis."""
    if not standardised_residuals:
        raise ValueError("Provide residuals for at least one function")
    function_ids = list(standardised_residuals)
    residuals = []
    for function_id in function_ids:
        values = np.asarray(standardised_residuals[function_id], dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Residuals for function {function_id} must be a nonempty finite list")
        residuals.append(values)

    figure, axes = plt.subplots(
        1, len(function_ids), figsize=(4.5 * len(function_ids), 5), squeeze=False,
        sharey=False,
    )
    axes = axes[0]
    for function_id, values, ax in zip(function_ids, residuals, axes):
        observations = np.arange(1, len(values) + 1)
        ax.scatter(observations, values, color="C0", alpha=0.7)
        for level, style in ((0, "-"), (1, ":"), (-1, ":"), (2, "--"), (-2, "--")):
            ax.axhline(level, color="gray", linestyle=style, linewidth=1)
        ax.set(
            title=f"Function {function_id}", xlabel="Observation",
            ylabel="(observed - expected) / std",
            xticks=observations,
        )
        ax.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    return figure, axes

def _candidate_prediction(optimizer, candidate, expected_mean, expected_std, expected_ei=None):
    if candidate is None:
        if expected_mean is not None or expected_std is not None or expected_ei is not None:
            raise ValueError("Saved predictions require candidate coordinates")
        return None
    point = np.asarray(candidate, dtype=float)
    if (
        point.shape != (optimizer.observations.dimension,)
        or not np.all(np.isfinite(point))
        or np.any((point < 0) | (point >= 1))
    ):
        raise ValueError("Candidate must have one finite coordinate per dimension within [0, 1)")
    if (expected_mean is None) != (expected_std is None):
        raise ValueError("Provide both saved mean and standard deviation")
    if expected_mean is not None and (
        not np.isfinite(expected_mean) or not np.isfinite(expected_std) or expected_std < 0
    ):
        raise ValueError("Saved mean must be finite and standard deviation finite and nonnegative")
    if expected_ei is not None and (not np.isfinite(expected_ei) or expected_ei < 0):
        raise ValueError("Saved EI must be finite and nonnegative")
    if expected_mean is None or expected_ei is None:
        mean, std, ei = optimizer.predict(point[None, :])
        if expected_mean is None:
            expected_mean, expected_std = mean[0], std[0]
        if expected_ei is None:
            expected_ei = ei[0]
    return point, float(expected_mean), float(expected_std), float(expected_ei)


def plot_gp_2d(
    optimizer: BayesianOptimizer, n=48, *, candidate=None, expected_mean=None, expected_std=None,
    expected_ei=None,
):
    """Plot GP surfaces, optionally marking a stored candidate and its saved prediction.

    Predict missing candidate mean/std or EI with optimizer.predict(). When all
    three saved values are supplied, only the grid requires model predictions.
    Without a candidate, mark the grid maximum as before.
    """
    observations = optimizer.observations
    prediction = _candidate_prediction(optimizer, candidate, expected_mean, expected_std, expected_ei)
    grid = np.linspace(0, np.nextafter(1.0, 0.0), n)
    x1, x2 = np.meshgrid(grid, grid)
    points = np.column_stack((x1.ravel(), x2.ravel()))
    mean, std, ei = optimizer.predict(points)
    best = observations.best_input
    grid_candidate = points[np.argmax(ei)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, values, title in zip(axes, (mean, std, ei), ("GP mean", "Predictive std", "EI")):
        image = ax.pcolormesh(x1, x2, values.reshape(x1.shape), shading="auto")
        fig.colorbar(image, ax=ax)
        ax.scatter(*observations.inputs.T, c="black", edgecolors="white", s=30, label="Observations")
        ax.scatter(*best, c="red", marker="*", s=130, label="Best observed")
        ax.set(title=title, xlabel="x1", ylabel="x2", xlim=(0, 1), ylim=(0, 1))
    if prediction is None:
        axes[2].scatter(*grid_candidate, c="red", marker="v", s=70, label="Grid EI max")
    else:
        point, candidate_mean, candidate_std, candidate_ei = prediction
        source = "Saved" if expected_mean is not None else "Predicted"
        for ax, value, name in zip(axes, (candidate_mean, candidate_std, candidate_ei), ("mean", "std", "EI")):
            ei_source = "Saved" if expected_ei is not None else "Model"
            label = f"{source} {name}" if name != "EI" else f"{ei_source} EI"
            ax.scatter(*point, c="red", marker="v", s=70, label=f"Stored candidate ({label}: {value:.6g})")
    for ax in axes:
        ax.legend(fontsize=8)
    return fig, axes


def plot_gp_slices(
    optimizer: BayesianOptimizer, n=48, *, candidate=None, expected_mean=None, expected_std=None,
    expected_ei=None,
):
    """Slice through a stored candidate, or the best observed input when omitted.

    Saved mean/std are shown separately from the current GP curve and band.
    Predict only missing candidate values; use saved EI when supplied.
    Each row shares a y-scale across all coordinate slices.
    """
    observations = optimizer.observations
    prediction = _candidate_prediction(optimizer, candidate, expected_mean, expected_std, expected_ei)
    reference = observations.best_input if prediction is None else prediction[0]
    dimension = observations.dimension
    grid = np.linspace(0, np.nextafter(1.0, 0.0), n)
    fig, axes = plt.subplots(
        2, dimension, figsize=(3.5 * dimension, 6), squeeze=False,
        sharex="col", sharey="row", constrained_layout=True,
    )
    max_ei = 0.0 if prediction is None else prediction[3]
    for j in range(dimension):
        points = np.tile(reference, (n, 1))
        points[:, j] = grid
        mean, std, ei = optimizer.predict(points)
        other = np.arange(dimension) != j
        on_slice = np.all(np.isclose(observations.inputs[:, other], reference[other]), axis=1)
        top, bottom = axes[:, j]
        top.plot(grid, mean, label="GP mean")
        top.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std, alpha=0.2, label="95% predictive band")
        top.scatter(observations.inputs[on_slice, j], observations.outputs[on_slice], c="black", label="On-slice observations")
        top.axhline(observations.best_output, color="gray", linestyle=":", label="Best observed")
        top.set(title=f"Vary x{j + 1}", ylabel="Output")
        bottom.plot(grid, ei, color="green")
        if prediction is None:
            peak = np.argmax(ei)
            bottom.scatter(
                grid[peak], ei[peak], c="red", marker="v", s=70,
                zorder=4, clip_on=False, label="Slice EI max",
            )
        else:
            point, candidate_mean, candidate_std, candidate_ei = prediction
            source = "Saved" if expected_mean is not None else "Predicted"
            top.errorbar(
                point[j], candidate_mean, yerr=1.96 * candidate_std,
                fmt="v", color="red", capsize=4, markersize=8, zorder=4,
                clip_on=False, label=f"{source} candidate mean +/- 1.96 std",
            )
            bottom.scatter(
                point[j], candidate_ei, c="red", marker="v", s=70,
                zorder=4, clip_on=False,
                label="Saved candidate EI" if expected_ei is not None else "Stored candidate EI",
            )
        max_ei = max(max_ei, float(np.max(ei)))
        bottom.set(xlabel=f"x{j + 1}", ylabel="EI")
        reference_label = "Reference input" if prediction is None else "Stored candidate input"
        for ax in (top, bottom):
            ax.axvline(reference[j], color="gray", linestyle="--", zorder=1, label=reference_label)
            ax.set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1.1 * max_ei if max_ei > 0 else 1)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    return fig, axes
