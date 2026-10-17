"""Sequential, budgeted Bayesian optimisation without assuming an objective formula."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from .acquisition import expected_improvement, upper_confidence_bound
from .data import FUNCTION_SPECS, Observations
from .settings import CONFIG
from .surrogate import KernelFactory, fit_surrogate


@dataclass(frozen=True)
class OptimisationConfig:
    budget: int = 20
    candidates: int = CONFIG["candidates"]
    acquisition: str = "ei"
    xi: float = 0.01
    kappa: float = 2.0
    seed: int | None = CONFIG["random_state"]
    kernel: Kernel | KernelFactory | None = None
    best: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("Budget must be a nonnegative integer")
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int) or self.candidates < 1:
            raise ValueError("Candidates must be a positive integer")
        if self.acquisition not in {"ei", "ucb"}:
            raise ValueError("Acquisition must be 'ei' or 'ucb'")
        if not np.isfinite(self.xi) or self.xi < 0 or not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("Acquisition parameters must be finite and nonnegative")
        if self.best is not None and not np.isfinite(self.best):
            raise ValueError("Best must be finite when provided")


def remaining_budget(observations: Observations, function_id: int, total_budget: int = 20) -> int:
    """Number of new queries left when total_budget includes initial observations."""
    if function_id not in FUNCTION_SPECS:
        raise ValueError("Function ID must be between 1 and 8")
    if observations.dimension != FUNCTION_SPECS[function_id]:
        raise ValueError("Observations do not match the specified function")
    if isinstance(total_budget, bool) or not isinstance(total_budget, int) or total_budget < 0:
        raise ValueError("Total budget must be a nonnegative integer")
    return max(0, total_budget - len(observations.inputs))


class BayesianOptimizer:
    """Call suggest, evaluate the real objective, then observe; repeat within budget."""

    def __init__(self, observations: Observations, config: OptimisationConfig = OptimisationConfig()):
        self.observations = observations
        self.config = config
        self.initial_count = len(observations.inputs)
        self._random = np.random.default_rng(config.seed)
        self._model: GaussianProcessRegressor | None = None
        self._fitted_observations: Observations | None = None
        self._fitted_config: OptimisationConfig | None = None

    @property
    def remaining(self) -> int:
        return self.config.budget - (len(self.observations.inputs) - self.initial_count)

    @property
    def history(self) -> Observations:
        return self.observations

    @property
    def acquisition_settings(self) -> dict[str, str | float]:
        """Current selection settings, including the resolved EI incumbent."""
        if self.config.acquisition == "ei":
            best = self.config.best
            if best is None:
                best = self.observations.best_output
            return {"name": "ei", "xi": float(self.config.xi), "best": float(best)}
        return {"name": "ucb", "kappa": float(self.config.kappa)}

    def suggest(self) -> np.ndarray:
        point, _, _, _ = self.suggest_with_prediction()
        return point

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictive mean, std, and configured acquisition.

        Reuse the fitted GP until observations or configuration are replaced.
        Prediction does not consume candidate randomness or evaluation budget.
        """
        if (
            self._model is None
            or self._fitted_observations is not self.observations
            or self._fitted_config is not self.config
        ):
            self._model = fit_surrogate(self.observations, self.config.seed, self.config.kernel)
            self._fitted_observations = self.observations
            self._fitted_config = self.config
        prediction = self._model.predict(points, return_std=True, return_cov=False)
        assert isinstance(prediction, tuple) and len(prediction) == 2
        mean, std = prediction
        settings = self.acquisition_settings
        if settings["name"] == "ei":
            acquisition = expected_improvement(
                mean, std, float(settings["best"]), float(settings["xi"]),
            )
        else:
            acquisition = upper_confidence_bound(mean, std, float(settings["kappa"]))
        return mean, std, acquisition

    def suggest_with_prediction(self) -> tuple[np.ndarray, float, float, float]:
        """Return the suggested point, predictive mean, std, and configured acquisition score."""
        pool = self._candidate_pool()
        mean, std, acquisition = self.predict(pool)
        selected = int(np.argmax(acquisition))
        return (
            pool[selected].copy(), float(mean[selected]), float(std[selected]),
            float(acquisition[selected]),
        )

    def suggest_ei_sweep(
        self, xi_values: Sequence[float] | np.ndarray, *, batch_size: int = 10_000,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        """Return a suggestion per xi using one shared pool and cached, batched predictions.

        Results follow xi_values order. This diagnostic does not change configuration
        or observations, but advances the candidate RNG once, like a suggestion.
        """
        if self.config.acquisition != "ei":
            raise ValueError("An EI sweep requires acquisition='ei'")
        xis = np.asarray(xi_values, dtype=float)
        if xis.ndim != 1 or not len(xis) or not np.all(np.isfinite(xis)) or np.any(xis < 0):
            raise ValueError("xi_values must be a nonempty sequence of finite nonnegative values")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        pool = self._candidate_pool()
        mean, std = np.empty(len(pool)), np.empty(len(pool))
        for start in range(0, len(pool), batch_size):
            batch = slice(start, start + batch_size)
            mean[batch], std[batch], _ = self.predict(pool[batch])
        best = float(self.acquisition_settings["best"])
        suggestions = []
        for xi in xis:
            scores = expected_improvement(mean, std, best, float(xi))
            selected = int(np.argmax(scores))
            suggestions.append((
                pool[selected].copy(), float(mean[selected]), float(std[selected]),
                float(scores[selected]),
            ))
        return suggestions

    def _candidate_pool(self) -> np.ndarray:
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted")
        for _ in range(10):
            pool = self._random.integers(0, 1_000_000, size=(self.config.candidates, self.observations.dimension)) / 1_000_000
            unused = np.ones(len(pool), dtype=bool)
            for point in self.observations.inputs:
                unused &= ~np.all(pool == point, axis=1)
            pool = pool[unused]
            if len(pool):
                return pool
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
