"""Acquisition scores for maximisation."""

import numpy as np
from scipy.special import erfcx, log_ndtr
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float | np.ndarray, xi: float = 0.01
) -> np.ndarray:
    improvement = np.asarray(mean) - best - xi
    std = np.asarray(std)
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
    return np.where(std > 0, improvement * norm.cdf(z) + std * norm.pdf(z), 0.0)


def log_expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float | np.ndarray, xi: float = 0.01
) -> np.ndarray:
    """Return log EI without underflowing for large negative standardised gains."""
    mean_values, std_values, best_values = np.broadcast_arrays(
        np.asarray(mean, dtype=float),
        np.asarray(std, dtype=float),
        np.asarray(best, dtype=float),
    )
    improvement = mean_values - best_values - xi
    z = np.divide(improvement, std_values, out=np.zeros_like(improvement), where=std_values > 0)
    result = np.full(z.shape, -np.inf, dtype=float)
    valid = std_values > 0
    if not np.any(valid):
        return result

    z_valid = z[valid]
    log_density = -0.5 * z_valid**2 - 0.5 * np.log(2.0 * np.pi)
    log_kernel = np.empty_like(z_valid)
    nonnegative = z_valid >= 0
    positive = z_valid > 0
    log_kernel[nonnegative] = log_density[nonnegative]
    log_kernel[nonnegative & positive] = np.logaddexp(
        log_density[nonnegative & positive],
        np.log(z_valid[nonnegative & positive]) + log_ndtr(z_valid[nonnegative & positive]),
    )

    negative = ~nonnegative
    if np.any(negative):
        tail = -z_valid[negative]
        factor = np.empty_like(tail)
        asymptotic = tail >= 8.0
        if np.any(asymptotic):
            inverse_square = 1.0 / tail[asymptotic] ** 2
            factor[asymptotic] = inverse_square * (
                1.0 + inverse_square * (
                    -3.0 + inverse_square * (
                        15.0 + inverse_square * (-105.0 + 945.0 * inverse_square)
                    )
                )
            )
        if np.any(~asymptotic):
            moderate_tail = tail[~asymptotic]
            factor[~asymptotic] = 1.0 - moderate_tail * np.sqrt(np.pi / 2.0) * erfcx(
                moderate_tail / np.sqrt(2.0)
            )
        log_kernel[negative] = log_density[negative] + np.log(factor)

    result[valid] = np.log(std_values[valid]) + log_kernel
    return result


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
