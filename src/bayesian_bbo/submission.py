"""Format candidate inputs for the external evaluation service."""

import numpy as np


def format_submission(point: np.ndarray) -> str:
    values = np.asarray(point, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Submission must be a finite one-dimensional vector")
    if np.any((values < 0) | (values >= 1)):
        raise ValueError("Submission inputs must be within [0, 1)")
    rounded = np.round(values, 6)
    if np.any(rounded >= 1):
        raise ValueError("Rounding to six decimals would exceed the valid input range")
    return "-".join(f"{value:.6f}" for value in rounded)
