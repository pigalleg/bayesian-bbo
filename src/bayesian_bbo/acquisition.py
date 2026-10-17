"""Acquisition scores for maximisation."""

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float | np.ndarray, xi: float = 0.01
) -> np.ndarray:
    improvement = np.asarray(mean) - best - xi
    std = np.asarray(std)
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
    return np.where(std > 0, improvement * norm.cdf(z) + std * norm.pdf(z), 0.0)


def expected_best_output(
    mean: np.ndarray, std: np.ndarray, best: float | np.ndarray
) -> np.ndarray:
    """Expected best-so-far after one evaluation: best plus EI with xi = 0."""
    mean_values = np.asarray(mean, dtype=float)
    std_values = np.asarray(std, dtype=float)
    gain = expected_improvement(mean_values, std_values, best, xi=0.0)
    return np.where(std_values > 0, best + gain, np.maximum(best, mean_values))


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return np.asarray(mean) + kappa * np.asarray(std)
