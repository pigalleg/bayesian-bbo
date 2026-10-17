"""Plot measured best-so-far values; never infer unmeasured outcomes."""

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import Observations
from .optimisation import BayesianOptimizer


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
    previous_best: np.ndarray,
    expected_best: np.ndarray,
    observed_best: np.ndarray,
):
    """Plot best-so-far per function; the arrays have shape (functions, weeks)."""
    if not weeks or not function_ids:
        raise ValueError("Provide at least one week and one function")
    shape = (len(function_ids), len(weeks))
    previous = np.asarray(previous_best, dtype=float)
    expected = np.asarray(expected_best, dtype=float)
    observed = np.asarray(observed_best, dtype=float)
    if {previous.shape, expected.shape, observed.shape} != {shape}:
        raise ValueError("Best-so-far arrays must have shape (functions, weeks)")
    if not np.all(np.isfinite(previous) & np.isfinite(expected) & np.isfinite(observed)):
        raise ValueError("Best-so-far values must be finite")
    tolerance = 1e-9 * (1 + np.abs(previous))
    if np.any(expected < previous - tolerance):
        raise ValueError("Expected best must not be below the previous best")
    if np.any(observed < previous - tolerance):
        raise ValueError("Observed best must not be below the previous best")

    rows = math.ceil(len(function_ids) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), squeeze=False)
    panels = axes.ravel()
    for index, (function_id, ax) in enumerate(zip(function_ids, panels)):
        ax.plot(
            weeks,
            previous[index],
            linestyle="--",
            color="gray",
            linewidth=2.5,
            label="Previous best",
        )
        ax.plot(weeks, expected[index], marker="o", label="Expected best")
        ax.plot(weeks, observed[index], marker="s", label="Observed best")
        ax.set(title=f"Function {function_id}", xlabel="Week", ylabel="Best-so-far", xticks=list(weeks))
        ax.grid(True, alpha=0.3)
    for ax in panels[len(function_ids):]:
        ax.set_visible(False)
    panels[0].legend()
    figure.suptitle("Best-so-far: previous, expected, and observed")
    figure.tight_layout()
    return figure, axes


def plot_calibration(standardised_residuals: Mapping[int, Sequence[float]]):
    """Strip plot of standardised residuals per function, with reference lines at 0, +/-1, and +/-2."""
    if not standardised_residuals:
        raise ValueError("Provide residuals for at least one function")
    function_ids = list(standardised_residuals)
    columns = []
    for function_id in function_ids:
        values = np.asarray(standardised_residuals[function_id], dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Residuals for function {function_id} must be a nonempty finite list")
        columns.append(values)

    figure, ax = plt.subplots(figsize=(10, 5))
    for position, values in enumerate(columns, start=1):
        ax.scatter(np.full(len(values), position), values, color="C0", alpha=0.7)
    for level, style in ((0, "-"), (1, ":"), (-1, ":"), (2, "--"), (-2, "--")):
        ax.axhline(level, color="gray", linestyle=style, linewidth=1)
    ax.set_xticks(range(1, len(function_ids) + 1), labels=[f"Function {function_id}" for function_id in function_ids])
    ax.set(ylabel="(observed - expected) / std", title="Calibration of expected outputs")
    ax.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    return figure, ax

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
