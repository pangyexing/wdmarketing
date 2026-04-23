"""Ranking metric correctness."""
import numpy as np
import pytest

from wdm.metrics.binned_lift import compute_binned_lift
from wdm.metrics.ks import ks_stat
from wdm.metrics.ranking import lift_at_k, precision_at_k, recall_at_k


def test_precision_at_k_perfect_ranking():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    assert precision_at_k(y, score, 0.3) == 1.0
    assert lift_at_k(y, score, 0.3) == pytest.approx(1.0 / 0.3, rel=1e-6)
    assert recall_at_k(y, score, 0.3) == 1.0


def test_precision_at_k_fractional_vs_int():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    # Top-2 → 2 positives, precision 1.0; 0.2 should be equivalent
    assert precision_at_k(y, score, 2) == 1.0
    assert precision_at_k(y, score, 0.2) == 1.0


def test_ks_is_between_0_and_1():
    rng = np.random.RandomState(0)
    y = (rng.rand(1000) < 0.3).astype(int)
    score = rng.rand(1000)
    ks = ks_stat(y, score)
    assert 0.0 <= ks <= 1.0


def test_binned_lift_sums_to_total_positives():
    rng = np.random.RandomState(1)
    y = (rng.rand(500) < 0.2).astype(int)
    score = rng.rand(500)
    df = compute_binned_lift(y, score, n_bins=10)
    assert df["n_positives"].sum() == int(y.sum())
    assert df["n_samples"].sum() == y.size
    assert df["cum_recall"].iloc[-1] == pytest.approx(1.0, rel=1e-6)
