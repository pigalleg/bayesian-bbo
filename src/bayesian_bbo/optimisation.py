"""Sequential, budgeted Bayesian optimisation without assuming an objective formula."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from scipy.stats import qmc

from .acquisition import expected_improvement, log_expected_improvement, upper_confidence_bound
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
    search_strategy: str = "random"
    lbfgs_restarts: int = 20
    local_candidates: int = 1024

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
        if not isinstance(self.search_strategy, str) or self.search_strategy not in {
            "random", "sobol_lbfgs",
        }:
            raise ValueError("Search strategy must be 'random' or 'sobol_lbfgs'")
        if (
            isinstance(self.lbfgs_restarts, bool)
            or not isinstance(self.lbfgs_restarts, int)
            or self.lbfgs_restarts < 1
        ):
            raise ValueError("lbfgs_restarts must be a positive integer")
        if (
            isinstance(self.local_candidates, bool)
            or not isinstance(self.local_candidates, int)
            or self.local_candidates < 0
        ):
            raise ValueError("local_candidates must be a nonnegative integer")


def remaining_budget(observations: Observations, function_id: int, total_budget: int = 20) -> int:
    """Number of new queries left when total_budget includes initial observations."""
    if function_id not in FUNCTION_SPECS:
        raise ValueError("Function ID must be between 1 and 8")
    if observations.dimension != FUNCTION_SPECS[function_id]:
        raise ValueError("Observations do not match the specified function")
    if isinstance(total_budget, bool) or not isinstance(total_budget, int) or total_budget < 0:
        raise ValueError("Total budget must be a nonnegative integer")
    return max(0, total_budget - len(observations.inputs))


def find_prediction_regions(
    predict: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    *,
    candidate_power: int = 15,
    region_count: int = 3,
    min_distance: float = 0.5,
    batch_size: int = 4096,
    seed: int | None = CONFIG["random_state"],
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily select separated, high-prediction Sobol representatives in [0, 1).

    Distance is Euclidean in the full unit cube. These are finite-pool model
    diagnostics, not identified basins, certified maxima, or measured outcomes.
    Fewer than region_count points may satisfy the requested separation.
    """
    for name, value, minimum in (
        ("dimension", dimension, 1), ("candidate_power", candidate_power, 0),
        ("region_count", region_count, 1), ("batch_size", batch_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError("min_distance must be finite and positive")
    pool = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(candidate_power)
    scores = np.empty(len(pool))
    for start in range(0, len(pool), batch_size):
        batch = pool[start:start + batch_size]
        values = np.asarray(predict(batch), dtype=float)
        if values.shape != (len(batch),) or not np.all(np.isfinite(values)):
            raise ValueError("predict must return one finite scalar per input row")
        scores[start:start + len(batch)] = values
    ranked = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    for index in ranked:
        if not selected or np.all(np.linalg.norm(pool[selected] - pool[index], axis=1) >= min_distance):
            selected.append(int(index))
            if len(selected) == region_count:
                break
    return pool[selected].copy(), scores[selected].copy()


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
        if self.config.search_strategy == "sobol_lbfgs":
            xi = float(self.acquisition_settings["xi"]) if self.config.acquisition == "ei" else None
            return self._polish_candidates(pool, mean, std, xi=xi)
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
            if self.config.search_strategy == "sobol_lbfgs":
                suggestions.append(self._polish_candidates(pool, mean, std, xi=float(xi)))
                continue
            selected = int(np.argmax(scores))
            suggestions.append((
                pool[selected].copy(), float(mean[selected]), float(std[selected]),
                float(scores[selected]),
            ))
        return suggestions

    def _candidate_pool(self) -> np.ndarray:
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted")
        if self.config.search_strategy == "sobol_lbfgs":
            exponent = (self.config.candidates - 1).bit_length()
            pool = qmc.Sobol(
                d=self.observations.dimension, scramble=True,
                rng=np.random.default_rng(self.config.seed),
            ).random_base2(exponent)[:self.config.candidates]
            if self.config.local_candidates:
                count = self.config.local_candidates
                scales = np.full(count, 0.15)
                scales[:count // 2] = 0.05
                local = self._random.normal(
                    loc=self.observations.best_input, scale=scales[:, None],
                )
                pool = np.vstack((pool, local))
            pool = np.clip(np.round(np.clip(pool, 0.0, 0.999999), 6), 0.0, 0.999999)
            unused = np.ones(len(pool), dtype=bool)
            for point in self.observations.inputs:
                unused &= ~np.all(pool == point, axis=1)
            pool = pool[unused]
            if len(pool):
                return pool
            raise RuntimeError("Could not find an unused candidate")
        for _ in range(10):
            pool = self._random.integers(0, 1_000_000, size=(self.config.candidates, self.observations.dimension)) / 1_000_000
            unused = np.ones(len(pool), dtype=bool)
            for point in self.observations.inputs:
                unused &= ~np.all(pool == point, axis=1)
            pool = pool[unused]
            if len(pool):
                return pool
        raise RuntimeError("Could not find an unused candidate")

    def _polish_candidates(
        self,
        pool: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        xi: float | None = None,
    ) -> tuple[np.ndarray, float, float, float]:
        """Polish top screening points and return the best unused snapped candidate."""
        settings = self.acquisition_settings
        if self.config.acquisition == "ei":
            best = float(settings["best"])
            threshold = float(settings["xi"]) if xi is None else xi
            ranking = log_expected_improvement(mean, std, best, threshold)
        else:
            best = 0.0
            threshold = 0.0
            ranking = upper_confidence_bound(mean, std, float(settings["kappa"]))
        starts = np.argsort(-ranking, kind="stable")[:self.config.lbfgs_restarts]
        bounds = [(0.0, 0.999999)] * self.observations.dimension

        def objective(point: np.ndarray) -> float:
            predicted_mean, predicted_std, _ = self.predict(np.asarray(point).reshape(1, -1))
            if self.config.acquisition == "ei":
                value = -float(log_expected_improvement(
                    predicted_mean[0], predicted_std[0], best, threshold,
                ))
            else:
                value = -float(upper_confidence_bound(
                    predicted_mean[0], predicted_std[0], float(settings["kappa"]),
                ))
            return value if np.isfinite(value) else 1e100

        candidates = [pool[int(starts[0])].copy()]
        for index in starts:
            result = minimize(
                objective, pool[int(index)], method="L-BFGS-B", bounds=bounds,
            )
            if result.x.shape == (self.observations.dimension,) and np.all(np.isfinite(result.x)):
                candidates.append(np.clip(np.round(result.x, 6), 0.0, 0.999999))

        unique: list[np.ndarray] = []
        for point in candidates:
            snapped = np.clip(np.round(point, 6), 0.0, 0.999999)
            if (
                not np.any(np.all(self.observations.inputs == snapped, axis=1))
                and not any(np.array_equal(snapped, existing) for existing in unique)
            ):
                unique.append(snapped)
        if not unique:
            raise RuntimeError("Could not find an unused polished candidate")

        candidate_array = np.asarray(unique)
        final_mean, final_std, configured_scores = self.predict(candidate_array)
        if self.config.acquisition == "ei":
            scores = expected_improvement(final_mean, final_std, best, threshold)
            if np.max(scores) == 0.0:
                tie_break = log_expected_improvement(final_mean, final_std, best, threshold)
                selected = int(np.argmax(tie_break))
            else:
                selected = int(np.argmax(scores))
        else:
            scores = upper_confidence_bound(final_mean, final_std, float(settings["kappa"]))
            selected = int(np.argmax(scores))
        score = float(scores[selected])
        if self.config.acquisition == "ei" and xi is None:
            score = float(configured_scores[selected])
        return (
            candidate_array[selected].copy(), float(final_mean[selected]),
            float(final_std[selected]), score,
        )

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
