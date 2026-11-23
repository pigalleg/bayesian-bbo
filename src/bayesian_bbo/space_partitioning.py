"""Opt-in Space-Partitioning Local Bayesian Optimisation (adapted from SPBOpt / TuRBO).

Adapts the core ideas of JetBrains Research's NeurIPS 2020 BBO Challenge entry
(Sazanovich et al., arXiv:2012.10335) for scalar **maximisation** using the
existing NumPy, SciPy and scikit-learn stack:

1. Recursively split observed points into higher- and lower-output clusters via
   2-means clustering on standardised ``[x, y]`` features and fit an SVM boundary
   per split (stopping whenever a split would drop below ``min_region_points``).
2. Fit a local Gaussian process on the active partition's observations (optionally
   rank-Gaussianising ``y`` so extreme scales or outliers do not distort the fit).
3. Build a trust-region box around the active region's best observed point
   (scaled by fitted ARD length scales when available), sample Sobol candidates
   inside the trust region, filter them through the learned SVM region, and select
   a six-decimal unused candidate by Thompson sampling or local Expected Improvement.

This module is an opt-in diagnostic/experimental alternative and does not alter
``BayesianOptimizer`` or historical weekly exports.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, rankdata
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.svm import SVC

from .acquisition import expected_improvement
from .data import Observations
from .kernels import make_matern_32_kernel
from .settings import CONFIG
from .surrogate import fit_surrogate


@dataclass(frozen=True)
class SpacePartitioningConfig:
    """Configuration for single-step space-partitioning local BO."""

    min_region_points: int = 8
    max_depth: int = 3
    trust_length: float = 0.4
    candidates: int = 2048
    acquisition: str = "thompson"
    xi: float = 0.0
    use_rank_transform: bool = True
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    length_cap: float = 31.622776601683793
    seed: int | None = CONFIG["random_state"]
    kernel: Kernel | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("min_region_points", self.min_region_points, 2),
            ("max_depth", self.max_depth, 0),
            ("candidates", self.candidates, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.trust_length) or not (0.0 < self.trust_length <= 1.0):
            raise ValueError("trust_length must be finite and in (0, 1]")
        if self.acquisition not in {"thompson", "ei"}:
            raise ValueError("acquisition must be 'thompson' or 'ei'")
        if not np.isfinite(self.xi) or self.xi < 0.0:
            raise ValueError("xi must be finite and nonnegative")
        if not isinstance(self.use_rank_transform, bool):
            raise ValueError("use_rank_transform must be a bool")
        if self.svm_kernel not in {"rbf", "linear", "poly"}:
            raise ValueError("svm_kernel must be 'rbf', 'linear', or 'poly'")
        if not np.isfinite(self.svm_c) or self.svm_c <= 0.0:
            raise ValueError("svm_c must be finite and positive")
        if not np.isfinite(self.length_cap) or self.length_cap < 1.0:
            raise ValueError("length_cap must be finite and >= 1.0")


@dataclass(frozen=True)
class PartitionSplit:
    """Record of one K-Means + SVM split along the path to the promising leaf."""

    depth: int
    svm: SVC
    good_label: int
    good_count: int
    bad_count: int
    good_mean: float
    bad_mean: float


@dataclass(frozen=True)
class SpacePartition:
    """Learned sequence of SVM splits and the resulting promising observation subset."""

    splits: tuple[PartitionSplit, ...]
    active_indices: np.ndarray

    def predict_good(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of points classified into the promising leaf."""
        coords = np.asarray(points, dtype=float)
        if coords.ndim != 2:
            raise ValueError("points must be a 2D array")
        mask = np.ones(len(coords), dtype=bool)
        for split in self.splits:
            if not np.any(mask):
                break
            mask[mask] = split.svm.predict(coords[mask]) == split.good_label
        return mask


@dataclass(frozen=True)
class SpacePartitioningResult:
    """Proposed candidate and diagnostic artifacts from one local-BO step."""

    candidate: np.ndarray
    local_mean: float
    local_std: float
    local_score: float
    target_space: str
    trust_lower: np.ndarray
    trust_upper: np.ndarray
    partition: SpacePartition
    local_gp: GaussianProcessRegressor
    candidate_pool_total: int
    candidate_pool_in_region: int
    used_region_filter: bool


def rank_gaussian_transform(values: np.ndarray) -> np.ndarray:
    """Map arbitrary real outputs monotonically to approximate standard-normal scores."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a nonempty 1D array")
    if len(arr) == 1 or np.all(arr == arr[0]):
        return np.zeros_like(arr, dtype=float)
    ranks = rankdata(arr, method="average")
    quantiles = (ranks - 0.5) / len(arr)
    return np.asarray(ndtri(quantiles), dtype=float)


def fit_space_partition(
    observations: Observations,
    config: SpacePartitioningConfig = SpacePartitioningConfig(),
) -> SpacePartition:
    """Recursively partition the unit cube with K-Means + SVM toward higher outputs."""
    n_obs = len(observations.outputs)
    active = np.arange(n_obs, dtype=int)
    splits: list[PartitionSplit] = []
    seed = 0 if config.seed is None else int(config.seed)

    for depth in range(config.max_depth):
        if len(active) < 2 * config.min_region_points:
            break
        sub_x = observations.inputs[active]
        sub_y = observations.outputs[active]
        y_feat = rank_gaussian_transform(sub_y) if config.use_rank_transform else sub_y
        y_std = float(np.std(y_feat))
        if y_std == 0.0:
            break
        x_std = np.std(sub_x, axis=0)
        x_scaled = np.divide(
            sub_x - np.mean(sub_x, axis=0),
            np.where(x_std > 0.0, x_std, 1.0),
        )
        y_scaled = (y_feat - np.mean(y_feat)) / y_std
        features = np.column_stack((x_scaled, y_scaled))

        labels = KMeans(
            n_clusters=2, n_init=10, random_state=seed + depth,
        ).fit_predict(features)
        counts = (int(np.sum(labels == 0)), int(np.sum(labels == 1)))
        if min(counts) < config.min_region_points:
            break

        mean_0 = float(np.mean(sub_y[labels == 0]))
        mean_1 = float(np.mean(sub_y[labels == 1]))
        good_label = 1 if mean_1 >= mean_0 else 0
        bad_label = 1 - good_label

        svm = SVC(kernel=config.svm_kernel, C=config.svm_c, random_state=seed + depth)
        svm.fit(sub_x, labels)
        splits.append(
            PartitionSplit(
                depth=depth,
                svm=svm,
                good_label=good_label,
                good_count=int(np.sum(labels == good_label)),
                bad_count=int(np.sum(labels == bad_label)),
                good_mean=max(mean_0, mean_1),
                bad_mean=min(mean_0, mean_1),
            )
        )
        active = active[labels == good_label]

    return SpacePartition(splits=tuple(splits), active_indices=active)


def _extract_length_scales(gp: GaussianProcessRegressor, dimension: int) -> np.ndarray | None:
    params = gp.kernel_.get_params()
    for key, value in params.items():
        if key.endswith("length_scale"):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 1 and len(arr) == dimension and np.all(arr > 0) and np.all(np.isfinite(arr)):
                return arr
    return None


def compute_trust_region(
    center: np.ndarray,
    trust_length: float,
    length_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a unit-cube trust-region box, optionally shaped by ARD length scales."""
    point = np.asarray(center, dtype=float)
    dimension = len(point)
    if length_scales is not None:
        weights = length_scales / float(np.mean(length_scales))
        weights = weights / float(np.prod(weights) ** (1.0 / dimension))
        half_widths = 0.5 * trust_length * weights
    else:
        half_widths = np.full(dimension, 0.5 * trust_length, dtype=float)
    lower = np.clip(point - half_widths, 0.0, 0.999999)
    upper = np.clip(point + half_widths, 0.0, 0.999999)
    degenerate = upper <= lower
    if np.any(degenerate):
        lower = np.where(degenerate, np.maximum(0.0, point - 1e-3), lower)
        upper = np.where(degenerate, np.minimum(0.999999, point + 1e-3), upper)
    return lower, upper


class SpacePartitioningOptimizer:
    """Single-step Space-Partitioning Local BO optimizer for maximisation."""

    def __init__(self, config: SpacePartitioningConfig = SpacePartitioningConfig()):
        self.config = config

    def suggest(self, observations: Observations) -> SpacePartitioningResult:
        """Partition the space, fit a local GP, and return one unused 6-decimal candidate."""
        cfg = self.config
        partition = fit_space_partition(observations, cfg)
        local_inputs = observations.inputs[partition.active_indices]
        local_raw_outputs = observations.outputs[partition.active_indices]
        if cfg.use_rank_transform:
            local_targets = rank_gaussian_transform(local_raw_outputs)
            target_space = "rank_gaussian"
        else:
            local_targets = local_raw_outputs
            target_space = "original"

        local_obs = Observations(local_inputs, local_targets, observations.dimension)
        kernel = (
            cfg.kernel
            if cfg.kernel is not None
            else make_matern_32_kernel(observations.dimension, cfg.length_cap)
        )
        local_gp = fit_surrogate(
            local_obs,
            random_state=cfg.seed,
            kernel=kernel,
            n_restarts_optimizer=CONFIG["n_restarts_optimizer"],
        )

        best_local_idx = int(np.argmax(local_raw_outputs))
        center = local_inputs[best_local_idx]
        length_scales = _extract_length_scales(local_gp, observations.dimension)
        trust_lower, trust_upper = compute_trust_region(center, cfg.trust_length, length_scales)

        exponent = (cfg.candidates - 1).bit_length()
        raw_sobol = qmc.Sobol(
            d=observations.dimension,
            scramble=True,
            rng=np.random.default_rng(cfg.seed),
        ).random_base2(exponent)[: cfg.candidates]
        pool = trust_lower + raw_sobol * (trust_upper - trust_lower)
        pool = np.clip(np.round(pool, 6), 0.0, 0.999999)

        pool = np.unique(pool, axis=0)
        unused = np.ones(len(pool), dtype=bool)
        for existing in observations.inputs:
            unused &= ~np.all(pool == existing, axis=1)
        pool = pool[unused]
        if not len(pool):
            raise RuntimeError("Could not find an unused candidate inside the trust region")

        in_region_mask = partition.predict_good(pool)
        in_region_count = int(np.sum(in_region_mask))
        used_region_filter = in_region_count > 0
        scoring_pool = pool[in_region_mask] if used_region_filter else pool

        means, stds = local_gp.predict(scoring_pool, return_std=True)
        if cfg.acquisition == "thompson":
            sample_seed = 0 if cfg.seed is None else int(cfg.seed) + 101
            try:
                sampled = local_gp.sample_y(
                    scoring_pool, n_samples=1, random_state=sample_seed,
                ).ravel()
            except np.linalg.LinAlgError:
                rng = np.random.default_rng(sample_seed)
                sampled = means + stds * rng.standard_normal(len(means))
            scores = sampled
        else:
            best_target = float(np.max(local_targets))
            scores = expected_improvement(means, stds, best_target, cfg.xi)

        selected = int(np.argmax(scores))
        return SpacePartitioningResult(
            candidate=scoring_pool[selected].copy(),
            local_mean=float(means[selected]),
            local_std=float(stds[selected]),
            local_score=float(scores[selected]),
            target_space=target_space,
            trust_lower=trust_lower,
            trust_upper=trust_upper,
            partition=partition,
            local_gp=local_gp,
            candidate_pool_total=int(len(pool)),
            candidate_pool_in_region=in_region_count,
            used_region_filter=used_region_filter,
        )
