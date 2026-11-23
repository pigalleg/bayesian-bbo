"""Unit tests for the opt-in space-partitioning local BO module."""

import numpy as np
import pytest

from bayesian_bbo.data import Observations
from bayesian_bbo.space_partitioning import (
    SpacePartitioningConfig,
    SpacePartitioningOptimizer,
    compute_trust_region,
    fit_space_partition,
    rank_gaussian_transform,
)


def _bimodal_observations() -> Observations:
    rng = np.random.default_rng(7)
    low_x = rng.uniform(0.05, 0.35, size=(12, 2))
    high_x = rng.uniform(0.65, 0.95, size=(12, 2))
    inputs = np.round(np.vstack((low_x, high_x)), 6)
    low_y = -2.0 + 0.1 * rng.standard_normal(12)
    high_y = 5.0 + 0.1 * rng.standard_normal(12)
    outputs = np.concatenate((low_y, high_y))
    return Observations(inputs, outputs, dimension=2)


def test_config_validation():
    with pytest.raises(ValueError):
        SpacePartitioningConfig(min_region_points=1)
    with pytest.raises(ValueError):
        SpacePartitioningConfig(max_depth=-1)
    with pytest.raises(ValueError):
        SpacePartitioningConfig(trust_length=0.0)
    with pytest.raises(ValueError):
        SpacePartitioningConfig(trust_length=1.5)
    with pytest.raises(ValueError):
        SpacePartitioningConfig(acquisition="ucb")
    with pytest.raises(ValueError):
        SpacePartitioningConfig(svm_kernel="sigmoid")


def test_rank_gaussian_transform_preserves_order_on_extreme_scales():
    raw = np.array([-3.6e-3, -1e-40, 1e-120, 6e-7, 5.85e-6])
    transformed = rank_gaussian_transform(raw)
    assert np.all(np.isfinite(transformed))
    assert np.all(np.diff(transformed) > 0)
    assert np.isclose(np.mean(transformed), 0.0, atol=1e-12)


def test_fit_space_partition_selects_higher_output_cluster():
    obs = _bimodal_observations()
    cfg = SpacePartitioningConfig(min_region_points=8, max_depth=2, seed=42)
    partition = fit_space_partition(obs, cfg)
    assert len(partition.splits) >= 1
    assert len(partition.active_indices) >= cfg.min_region_points
    assert np.mean(obs.outputs[partition.active_indices]) > 4.0
    mask = partition.predict_good(np.array([[0.8, 0.8], [0.2, 0.2]]))
    assert mask.tolist() == [True, False]


def test_fit_space_partition_stops_when_too_few_points():
    obs = _bimodal_observations()
    cfg = SpacePartitioningConfig(min_region_points=15, max_depth=3, seed=42)
    partition = fit_space_partition(obs, cfg)
    assert len(partition.splits) == 0
    assert len(partition.active_indices) == len(obs.outputs)


def test_compute_trust_region_respects_bounds_and_ard_scaling():
    low, high = compute_trust_region(
        np.array([0.98, 0.50]), trust_length=0.4, length_scales=np.array([0.2, 0.8]),
    )
    assert np.all(low >= 0.0) and np.all(high <= 0.999999)
    assert np.all(high > low)
    assert (high[1] - low[1]) > (high[0] - low[0])


@pytest.mark.parametrize("acquisition", ["thompson", "ei"])
@pytest.mark.parametrize("use_rank_transform", [True, False])
def test_optimizer_suggest_is_reproducible_valid_and_unused(acquisition, use_rank_transform):
    obs = _bimodal_observations()
    cfg = SpacePartitioningConfig(
        min_region_points=8,
        max_depth=2,
        trust_length=0.4,
        candidates=256,
        acquisition=acquisition,
        use_rank_transform=use_rank_transform,
        seed=42,
    )
    r1 = SpacePartitioningOptimizer(cfg).suggest(obs)
    r2 = SpacePartitioningOptimizer(cfg).suggest(obs)
    np.testing.assert_array_equal(r1.candidate, r2.candidate)
    assert r1.candidate.shape == (2,)
    assert np.all((r1.candidate >= 0.0) & (r1.candidate < 1.0))
    np.testing.assert_allclose(r1.candidate, np.round(r1.candidate, 6))
    assert not np.any(np.all(obs.inputs == r1.candidate, axis=1))
    assert np.isfinite(r1.local_mean) and np.isfinite(r1.local_std) and r1.local_std >= 0.0
    assert r1.candidate[0] > 0.5 and r1.candidate[1] > 0.5
