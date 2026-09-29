"""Acquisition scores for maximisation."""

import numpy as np
from scipy.stats import norm


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    improvement = np.asarray(mean) - best - xi
    std = np.asarray(std)
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
    return np.where(std > 0, improvement * norm.cdf(z) + std * norm.pdf(z), 0.0)


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return np.asarray(mean) + kappa * np.asarray(std)
