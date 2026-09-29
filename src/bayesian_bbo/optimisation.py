"""Sequential, budgeted Bayesian optimisation without assuming an objective formula."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .acquisition import expected_improvement, upper_confidence_bound
from .data import Observations
from .surrogate import fit_surrogate


@dataclass(frozen=True)
class OptimisationConfig:
    budget: int = 20
    candidates: int = 2048
    acquisition: str = "ei"
    xi: float = 0.01
    kappa: float = 2.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("Budget must be a nonnegative integer")
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int) or self.candidates < 1:
            raise ValueError("Candidates must be a positive integer")
        if self.acquisition not in {"ei", "ucb"}:
            raise ValueError("Acquisition must be 'ei' or 'ucb'")
        if not np.isfinite(self.xi) or self.xi < 0 or not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("Acquisition parameters must be finite and nonnegative")


class BayesianOptimizer:
    """Call suggest, evaluate the real objective, then observe; repeat within budget."""

    def __init__(self, observations: Observations, config: OptimisationConfig = OptimisationConfig()):
        self.observations = observations
        self.config = config
        self.initial_count = len(observations.inputs)
        self._random = np.random.default_rng(config.seed)

    @property
    def remaining(self) -> int:
        return self.config.budget - (len(self.observations.inputs) - self.initial_count)

    @property
    def history(self) -> Observations:
        return self.observations

    def suggest(self) -> np.ndarray:
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted")
        model = fit_surrogate(self.observations, self.config.seed)
        for _ in range(10):
            pool = self._random.integers(0, 1_000_000, size=(self.config.candidates, self.observations.dimension)) / 1_000_000
            unused = ~np.any(np.all(pool[:, None, :] == self.observations.inputs[None, :, :], axis=2), axis=1)
            pool = pool[unused]
            if len(pool):
                mean, std = model.predict(pool, return_std=True)
                if self.config.acquisition == "ei":
                    scores = expected_improvement(mean, std, self.observations.best_output, self.config.xi)
                else:
                    scores = upper_confidence_bound(mean, std, self.config.kappa)
                return pool[int(np.argmax(scores))].copy()
        raise RuntimeError("Could not find an unused candidate")

    def observe(self, point: np.ndarray, value: float) -> Observations:
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted")
        self.observations = self.observations.with_observation(point, value)
        return self.observations

    def run(self, evaluate: Callable[[np.ndarray], float]) -> Observations:
        """Evaluate and refit sequentially; persist real observations separately if needed."""
        while self.remaining:
            point = self.suggest()
            value = evaluate(point.copy())
            self.observe(point, value)
        return self.history
