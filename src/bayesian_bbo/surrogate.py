"""Gaussian-process surrogate for scalar black-box observations."""

from collections.abc import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel

from .data import Observations


KernelFactory = Callable[[int], Kernel]


def fit_surrogate(
    observations: Observations,
    random_state: int | None = None,
    kernel: Kernel | KernelFactory | None = None,
) -> GaussianProcessRegressor:
    if kernel is None:
        selected_kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(observations.dimension), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
    elif isinstance(kernel, Kernel):
        selected_kernel = kernel
    else:
        selected_kernel = kernel(observations.dimension)
    model = GaussianProcessRegressor(kernel=selected_kernel, normalize_y=True, random_state=random_state)
    model.fit(observations.inputs, observations.outputs)
    return model
