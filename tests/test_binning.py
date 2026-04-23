"""Tests for equal_freq_edges small-bin merging.

The merge guarantees every bin holds ≥ min_samples; without it, sparse
features (90%+ zeros) produce tiny non-zero bins whose IV/PSI is noise.
"""
import numpy as np
import pytest

from wdm.utils.binning import equal_freq_edges


def _bin_counts(values, edges):
    """Per-bin sample count given edges (interior-digitize convention)."""
    bins = np.digitize(values, edges[1:-1], right=False)
    return np.bincount(bins, minlength=edges.size - 1)


def test_merge_collapses_tiny_tail_bins():
    """90% zeros + 10% spread → merging yields few bins, each ≥ min_samples."""
    rng = np.random.RandomState(0)
    zeros = np.zeros(900)
    tail = rng.uniform(1.0, 10.0, size=100)
    values = np.concatenate([zeros, tail])

    edges = equal_freq_edges(values, n_bins=10, min_samples_per_bin=50)
    counts = _bin_counts(values, edges)
    assert (counts >= 50).all(), "all bins must meet min_samples after merging"
    # The giant zero mass forces merging down to very few bins.
    assert edges.size - 1 <= 3, "expected ≤ 3 bins for 90%-zero feature"


def test_disable_merging_preserves_legacy_edges():
    """min_samples_per_bin=0 or 1 keeps the raw quantile edges."""
    rng = np.random.RandomState(0)
    values = rng.randn(10000)
    legacy = equal_freq_edges(values, n_bins=10, min_samples_per_bin=0)
    # 10 bins → 11 unique quantile edges on well-distributed data.
    assert legacy.size == 11


def test_auto_default_does_not_over_merge_dense_data():
    """With N=10000 uniform data and auto default (max(50, 100)=100), all
    10 quantile bins already have ~1000 samples → no merging happens.
    """
    rng = np.random.RandomState(0)
    values = rng.randn(10000)
    edges = equal_freq_edges(values, n_bins=10)  # auto default
    counts = _bin_counts(values, edges)
    assert edges.size == 11, "dense data should keep all 10 bins"
    assert (counts >= 100).all()


def test_merge_respects_left_and_right_edge_cases():
    """Smallest bin at the left and right edges must merge with the only
    available neighbor, not error out.
    """
    # Put one rare value below the rest — it forms bin 0 with count 1.
    values = np.concatenate([np.array([-100.0]), np.arange(1000)])
    edges = equal_freq_edges(values, n_bins=10, min_samples_per_bin=50)
    counts = _bin_counts(values, edges)
    assert (counts >= 50).all()

    # One rare value above the rest — forms the last bin.
    values2 = np.concatenate([np.arange(1000), np.array([1e6])])
    edges2 = equal_freq_edges(values2, n_bins=10, min_samples_per_bin=50)
    counts2 = _bin_counts(values2, edges2)
    assert (counts2 >= 50).all()


def test_low_cardinality_branch_also_merges_rare_levels():
    """When unique count ≤ n_bins, each unique gets its own bin — but very
    rare levels (e.g., 3 samples out of 10000) should still merge.
    """
    # 4 levels: 0 (5000), 1 (4997), 2 (2), 3 (1). Levels 2 and 3 are noise.
    values = np.concatenate([
        np.zeros(5000), np.ones(4997),
        np.full(2, 2.0), np.full(1, 3.0),
    ])
    edges = equal_freq_edges(values, n_bins=10, min_samples_per_bin=50)
    counts = _bin_counts(values, edges)
    assert (counts >= 50).all()
    # 0 and 1 survive; 2 and 3 get absorbed.
    assert edges.size - 1 <= 3


def test_degenerate_small_input_collapses_to_single_bin():
    """N < min_samples → merge loop collapses to 1 bin rather than erroring."""
    values = np.arange(30.0)
    edges = equal_freq_edges(values, n_bins=10, min_samples_per_bin=50)
    # Can't satisfy 50-per-bin; should collapse to 1 bin (2 edges).
    assert edges.size == 2
