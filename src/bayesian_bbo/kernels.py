"""Shared kernel constructors for weekly optimization workflows."""

import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel, ExpSineSquared, Kernel, Matern, RBF, RationalQuadratic, WhiteKernel,
)


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


def make_rational_quadratic_kernel(dimension: int, length_cap: float) -> Kernel:
    """Return a fresh isotropic Rational Quadratic kernel with weekly noise settings."""
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("Dimension must be a positive integer")
    if isinstance(length_cap, bool) or not np.isfinite(length_cap) or length_cap < 1.0:
        raise ValueError("Length cap must be finite and at least the initial length scale, 1.0")
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, length_cap))
        + WhiteKernel(1e-5, (1e-8, 1e1))
    )


def make_comparison_kernels(dimension: int, length_cap: float) -> dict[str, Kernel]:
    """Construct the five kernel candidates used by the cross-validation notebooks."""
    matern32 = make_matern_32_kernel(dimension, length_cap)
    bounds = (1e-3, length_cap)
    signals = {
        "Matérn 5/2": Matern(np.ones(dimension), length_scale_bounds=bounds, nu=2.5),
        "RBF": RBF(np.ones(dimension), length_scale_bounds=bounds),
        "Rational Quadratic": RationalQuadratic(length_scale_bounds=bounds),
        "Exp-Sine-Squared": ExpSineSquared(
            length_scale_bounds=bounds, periodicity_bounds=(1e-2, 1e2),
        ),
    }
    return {
        "Matérn 3/2": matern32,
        **{
            name: ConstantKernel(1.0, (1e-3, 1e3)) * signal + WhiteKernel(1e-5, (1e-8, 1e1))
            for name, signal in signals.items()
        },
    }