"""Shared kernel constructors for weekly optimization workflows."""

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel


def make_matern_32_kernel(dimension: int, length_cap: float) -> Kernel:
    """Return a fresh ARD Matern 3/2 kernel with the weekly noise settings."""
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("Dimension must be a positive integer")
    if isinstance(length_cap, bool) or not np.isfinite(length_cap) or length_cap < 1.0:
        raise ValueError("Length cap must be finite and at least the initial length scale, 1.0")
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(np.ones(dimension), length_scale_bounds=(1e-3, length_cap), nu=1.5)
        + WhiteKernel(1e-5, (1e-8, 1e1))
    )