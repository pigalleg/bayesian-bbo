"""Cross-validation utilities for comparing Gaussian-process kernels."""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import KFold

from .settings import CONFIG


KernelFactory = Callable[[], Kernel]


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