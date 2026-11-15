"""Cross-validation utilities for comparing Gaussian-process kernels."""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import KFold
from sklearn.base import clone

from .data import Observations
from .settings import CONFIG
from .surrogate import fit_surrogate


KernelFactory = Callable[[], Kernel]


def evaluate_kernel_splits(
    observations: Observations,
    kernels: Mapping[str, Kernel],
    splits: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int = CONFIG["random_state"],
    n_restarts_optimizer: int = CONFIG["n_restarts_optimizer"],
    normalize_y: bool = CONFIG["normalize_y"],
) -> pd.DataFrame:
    """Return held-out predictions and scores on explicit shared splits.

    Fits only the training rows of each split, including target normalization.
    Adds a training-mean Gaussian baseline with population training std.
    Predictive std is retained for intervals; a 1e-9 scoring floor prevents
    division by zero in residual/NLPD calculations and is reported separately.
    """
    if not kernels or not splits:
        raise ValueError("Provide at least one kernel and one split")
    if "Training mean" in kernels:
        raise ValueError("'Training mean' is reserved for the baseline")
    if any(not isinstance(kernel, Kernel) for kernel in kernels.values()):
        raise TypeError("Each kernel must be a scikit-learn Kernel")
    if isinstance(n_restarts_optimizer, bool) or not isinstance(n_restarts_optimizer, int) or n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer must be a nonnegative integer")
    checked_splits = {}
    for split_id, (train, test) in splits.items():
        indices = []
        for name, values in (("training", train), ("validation", test)):
            array = np.asarray(values)
            if (
                array.ndim != 1 or not len(array)
                or not np.issubdtype(array.dtype, np.integer)
                or np.any(array < 0) or np.any(array >= len(observations.outputs))
                or len(np.unique(array)) != len(array)
            ):
                raise ValueError(f"Split {split_id}: invalid {name} indices")
            indices.append(array)
        if np.intersect1d(*indices).size:
            raise ValueError(f"Split {split_id}: training and validation indices overlap")
        checked_splits[split_id] = indices

    records = []
    for split_id, (train, test) in checked_splits.items():
        training = Observations(
            observations.inputs[train], observations.outputs[train], observations.dimension,
        )
        train_mean = float(training.outputs.mean())
        train_std = float(training.outputs.std())
        predictions = {
            "Training mean": (
                np.full(len(test), train_mean), np.full(len(test), train_std), "Training mean",
            ),
        }
        for name, kernel in kernels.items():
            model = fit_surrogate(
                training, kernel=clone(kernel), random_state=random_state,
                n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y,
            )
            mean, std = model.predict(observations.inputs[test], return_std=True)
            predictions[name] = (mean, std, str(model.kernel_))
        for name, (mean, std, fitted_kernel) in predictions.items():
            if (
                np.shape(mean) != (len(test),) or np.shape(std) != (len(test),)
                or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
                or np.any(std < 0)
            ):
                raise ValueError(f"Split {split_id}, {name}: invalid predictive mean or std")
            scoring_std = np.maximum(std, 1e-9)
            actual = observations.outputs[test]
            error = actual - mean
            z = error / scoring_std
            nlpd = np.log(scoring_std) + 0.5 * np.log(2 * np.pi) + 0.5 * z**2
            if not np.all(np.isfinite(nlpd)):
                raise ValueError(f"Split {split_id}, {name}: predictive scores overflowed")
            for position, row_index in enumerate(test):
                records.append({
                    "kernel": name, "split": split_id, "row_index": int(row_index),
                    "week": int(observations.weeks[row_index]),
                    "training_max_week": int(observations.weeks[train].max()),
                    "training_count": len(train), "training_mean": train_mean,
                    "training_std": train_std,
                    "actual": float(actual[position]), "mean": float(mean[position]),
                    "std": float(std[position]), "scoring_std": float(scoring_std[position]),
                    "error": float(error[position]), "z": float(z[position]),
                    "absolute_error": float(abs(error[position])),
                    "squared_error": float(error[position] ** 2),
                    "nlpd": float(nlpd[position]),
                    "lower_90": float(mean[position] - 1.645 * std[position]),
                    "upper_90": float(mean[position] + 1.645 * std[position]),
                    "lower_95": float(mean[position] - 1.96 * std[position]),
                    "upper_95": float(mean[position] + 1.96 * std[position]),
                    "covered_90": bool(abs(error[position]) <= 1.645 * std[position]),
                    "covered_95": bool(abs(error[position]) <= 1.96 * std[position]),
                    "width_90": float(2 * 1.645 * std[position]),
                    "width_95": float(2 * 1.96 * std[position]),
                    "fitted_kernel": fitted_kernel,
                })
    return pd.DataFrame(records)


def cross_validate_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernel_factories: Mapping[str, KernelFactory],
    *,
    n_splits: int = 5,
    random_state: int = CONFIG["random_state"],
    n_restarts_optimizer: int = CONFIG["n_restarts_optimizer"],
    normalize_y: bool = CONFIG["normalize_y"],
) -> pd.DataFrame:
    """Score each kernel on the same shuffled K-fold splits."""
    inputs = np.asarray(X, dtype=float)
    outputs = np.asarray(y, dtype=float)
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs[:, 0]
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional array")
    if outputs.ndim != 1 or len(outputs) != len(inputs):
        raise ValueError("y must contain one scalar output per row in X")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(outputs)):
        raise ValueError("X and y must contain only finite values")
    output_scale = max(float(np.std(outputs)), np.finfo(float).eps)
    if not isinstance(n_splits, int) or n_splits < 2 or n_splits > len(inputs):
        raise ValueError("n_splits must be an integer between 2 and the number of samples")
    if not isinstance(n_restarts_optimizer, int) or n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer must be a nonnegative integer")
    if not kernel_factories:
        raise ValueError("kernel_factories must contain at least one kernel")

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(inputs)
    fold_indices = list(folds)
    records = []

    for kernel_name, kernel_factory in kernel_factories.items():
        if not callable(kernel_factory):
            raise TypeError(f"Kernel factory for {kernel_name!r} must be callable")
        for fold_number, (training_indices, validation_indices) in enumerate(fold_indices, start=1):
            model = GaussianProcessRegressor(
                kernel=kernel_factory(),
                normalize_y=normalize_y,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state,
            )
            model.fit(inputs[training_indices], outputs[training_indices])
            prediction, predictive_std = model.predict(inputs[validation_indices], return_std=True)
            predictive_std = np.maximum(predictive_std, 1e-9)
            residual = outputs[validation_indices] - prediction
            variance = predictive_std**2
            rmse = float(np.sqrt(np.mean(residual**2)))

            records.append({
                "kernel": kernel_name,
                "fold": fold_number,
                "RMSE": rmse,
                "NRMSE": rmse / output_scale,
                "MAE": float(np.mean(np.abs(residual))),
                "negative_log_predictive_density": float(np.mean(
                    0.5 * np.log(2.0 * np.pi * variance) + residual**2 / (2.0 * variance)
                )),
                "coverage_90": float(np.mean(np.abs(residual) <= 1.645 * predictive_std)),
            })

    return pd.DataFrame(records)