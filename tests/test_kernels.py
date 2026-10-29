import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bayesian_bbo.kernels import make_matern_32_kernel


@pytest.mark.parametrize("dimension,length_cap", [(2, 31.622776601683793), (4, 10.0), (8, 100.0)])
def test_matern_32_matches_notebook_kernel(dimension, length_cap):
    original = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(np.ones(dimension), length_scale_bounds=(1e-3, length_cap), nu=1.5)
        + WhiteKernel(1e-5, (1e-8, 1e1))
    )
    shared = make_matern_32_kernel(dimension, length_cap)
    assert str(original) == str(shared)
    np.testing.assert_array_equal(shared.theta, original.theta)
    np.testing.assert_array_equal(shared.bounds, original.bounds)
    points = np.random.default_rng(42).random((6, dimension))
    outputs = points.sum(axis=1)
    models = [GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(points, outputs)
              for kernel in (original, shared)]
    for actual, expected in zip(models[0].predict(points, return_std=True), models[1].predict(points, return_std=True)):
        np.testing.assert_array_equal(actual, expected)
    another = make_matern_32_kernel(dimension, length_cap)
    shared.set_params(k1__k2__length_scale=np.full(dimension, 0.5))
    np.testing.assert_array_equal(another.k1.k2.length_scale, np.ones(dimension))


@pytest.mark.parametrize("dimension,cap", [(0, 10), (True, 10), (1.5, 10), (2, 0.5), (2, np.inf), (2, np.nan)])
def test_matern_32_rejects_invalid_parameters(dimension, cap):
    with pytest.raises(ValueError):
        make_matern_32_kernel(dimension, cap)