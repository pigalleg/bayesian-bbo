import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF

from bayesian_bbo.validation import cross_validate_kernels


@pytest.mark.parametrize("dimension", [2, 2, 3, 4, 4, 5, 6, 8])
def test_cross_validation_supports_function_input_dimensions(dimension):
    random = np.random.default_rng(dimension)
    inputs = random.uniform(0.0, 1.0, size=(6, dimension))
    outputs = np.sum(inputs**2, axis=1)
    kernel_factories = {
        "RBF": lambda: RBF(length_scale=np.ones(dimension)),
    }

    results = cross_validate_kernels(
        inputs,
        outputs,
        kernel_factories,
        n_splits=3,
        n_restarts_optimizer=0,
    )

    assert list(results.columns) == [
        "kernel",
        "fold",
        "RMSE",
        "NRMSE",
        "MAE",
        "negative_log_predictive_density",
        "coverage_90",
    ]
    assert results["fold"].tolist() == [1, 2, 3]
    assert results["kernel"].tolist() == ["RBF"] * 3
    np.testing.assert_allclose(results["NRMSE"], results["RMSE"] / np.std(outputs))
    assert np.isfinite(results.select_dtypes(include="number")).all().all()


def test_cross_validation_rejects_mismatched_outputs():
    with pytest.raises(ValueError, match="one scalar output per row"):
        cross_validate_kernels(
            np.ones((4, 2)),
            np.ones(3),
            {"RBF": lambda: RBF()},
            n_splits=2,
            n_restarts_optimizer=0,
        )