"""Gaussian-process surrogate for scalar black-box observations."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .data import Observations


def fit_surrogate(observations: Observations, random_state: int | None = None) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(observations.dimension), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
    )
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
    model.fit(observations.inputs, observations.outputs)
    return model
