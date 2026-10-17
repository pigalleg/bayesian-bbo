"""Suggest a Bayesian-optimisation candidate and return its GP prediction."""

from typing import Literal

import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from bayesian_bbo.data import Observations
from bayesian_bbo.optimisation import BayesianOptimizer, OptimisationConfig
from bayesian_bbo.settings import CONFIG
from bayesian_bbo.surrogate import KernelFactory


def suggest_candidate(
    observations: Observations,
    kernel: Kernel | KernelFactory | None,
    random_state: int | None = CONFIG["random_state"],
    acquisition: Literal["ei", "ucb"] = "ei",
    best: float | None = None,
    xi: float = 0.01,
    kappa: float = 2.0,
) -> tuple[np.ndarray, float, float, float]:
    """Return the selected point, predictive mean, std, and acquisition score.

    Candidates are sampled from a uniform ``1e-6`` grid over ``[0, 1)``.
    Kernel hyperparameters are optimized unless fixed in the supplied kernel.
    For EI, ``best`` defaults to the best observed output.
    """
    if acquisition not in {"ei", "ucb"}:
        raise ValueError("Acquisition must be 'ei' or 'ucb'")
    if acquisition == "ei":
        best = observations.best_output if best is None else best
        if not np.isfinite(best) or not np.isfinite(xi) or xi < 0:
            raise ValueError("EI best and xi must be finite, and xi must be nonnegative")
    elif not np.isfinite(kappa) or kappa < 0:
        raise ValueError("UCB kappa must be finite and nonnegative")

    optimizer = BayesianOptimizer(
        observations,
        OptimisationConfig(
            budget=1,
            acquisition=acquisition,
            xi=xi,
            kappa=kappa,
            seed=random_state,
            kernel=kernel,
            best=best,
        ),
    )
    return optimizer.suggest_with_prediction()