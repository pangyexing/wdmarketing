"""Shared binning utilities: equal_freq and tree-based.

All binners accept pre-NaN-replaced arrays (NaN = missing).
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _merge_small_bins(edges, values, min_samples):
    """Merge adjacent bins until every bin holds ≥ min_samples (or only 1 bin
    remains). Each iteration finds the smallest bin and drops the edge between
    it and its smaller neighbor — this preserves larger, more stable bins.

    Critical for sparse features: equal-frequency quantiles on a 90%-zero
    column produce a huge zero bin and a handful of tiny non-zero bins whose
    IV contribution is pure noise (log(p/n) on <50 samples).
    """
    if edges.size < 3:
        return edges
    values = np.asarray(values, dtype=np.float64)
    while edges.size >= 3:
        bins = np.digitize(values, edges[1:-1], right=False)
        counts = np.bincount(bins, minlength=edges.size - 1)
        if (counts >= min_samples).all():
            break
        smallest = int(np.argmin(counts))
        if smallest == 0:
            drop = 1
        elif smallest == counts.size - 1:
            drop = edges.size - 2
        else:
            # Merge with the neighbor that has fewer samples, so the resulting
            # bin stays as balanced as possible with the rest.
            left = counts[smallest - 1]
            right = counts[smallest + 1]
            drop = smallest if left <= right else smallest + 1
        edges = np.concatenate([edges[:drop], edges[drop + 1:]])
    return edges


def equal_freq_edges(values, n_bins=10, min_samples_per_bin=None):
    """Compute equal-frequency (quantile) edges from non-NaN values.

    Returns: np.ndarray of edges with len = n_effective_bins + 1.

    Low-cardinality branch: when unique values ≤ n_bins (typical for label-encoded
    categoricals like poutcome=0/1/2/3), use each unique value as its own bin.
    This prevents categorical features from collapsing to 1 bin under naive
    quantile binning when one category dominates.

    min_samples_per_bin: int or None
        Minimum samples per bin; adjacent bins under this threshold are merged
        into neighbors so small bins cannot inflate downstream IV/PSI/lift with
        noise. None → auto = max(50, int(0.01 * N_non_nan)). Pass 0 or 1 to
        disable merging (preserves pre-2026-04 behavior).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return np.array([], dtype=np.float64)
    uniques = np.unique(values)
    if uniques.size == 1:
        v = float(uniques[0])
        return np.array([v, v + 1e-12], dtype=np.float64)

    if uniques.size <= n_bins:
        # Put each unique value in its own bin by placing edges midway between
        # consecutive uniques. Edges = [min, mid_1, mid_2, ..., mid_{k-1}, max+eps]
        mids = (uniques[:-1] + uniques[1:]) / 2.0
        edges = np.concatenate([[uniques[0]], mids, [uniques[-1]]])
        edges[-1] = np.nextafter(edges[-1], np.inf)
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, qs, interpolation="linear")
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([values.min(), values.max() + 1e-12], dtype=np.float64)
        edges[-1] = np.nextafter(edges[-1], np.inf)

    if min_samples_per_bin is None:
        min_samples_per_bin = max(50, int(0.01 * n))
    if min_samples_per_bin > 1 and edges.size >= 3:
        edges = _merge_small_bins(edges, values, int(min_samples_per_bin))
    return edges.astype(np.float64)


def tree_edges(values, y, n_bins=10, min_samples_leaf=None, random_state=0):
    """Decision-tree based monotonic-friendly binning on (values, y).

    Uses sklearn DecisionTreeClassifier leaves as bins. Missing values
    must be pre-filtered by caller.
    """
    from sklearn.tree import DecisionTreeClassifier
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y)
    mask = ~np.isnan(values.ravel())
    x = values[mask]
    yy = y[mask]
    if x.size == 0 or len(np.unique(yy)) < 2:
        return equal_freq_edges(values.ravel(), n_bins=n_bins)
    leaf = min_samples_leaf or max(int(0.05 * x.size), 50)
    tree = DecisionTreeClassifier(
        max_leaf_nodes=n_bins,
        min_samples_leaf=leaf,
        random_state=random_state,
    )
    tree.fit(x, yy)
    thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
    edges = np.concatenate([[x.min()], np.sort(thresholds), [x.max()]])
    edges = np.unique(edges)
    if edges.size < 2:
        return equal_freq_edges(values.ravel(), n_bins=n_bins)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges.astype(np.float64)


def digitize_with_missing(values, edges, missing_bin=-1):
    """Digitize values against edges; return int array.

    - NaN → missing_bin (default -1, meaning "the missing bin")
    - values < edges[0] → 0
    - values >= edges[-1] → len(edges)-2 (clipped into last real bin)
    - otherwise bin index in [0, len(edges)-2]
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty(arr.shape, dtype=np.int64)
    nan_mask = np.isnan(arr)
    if edges.size < 2:
        out[:] = missing_bin
        out[~nan_mask] = 0
        return out
    non_nan = arr[~nan_mask]
    # np.digitize returns bins in 1..len(edges); shift to 0-based, clip to [0, n_bins-1]
    idx = np.digitize(non_nan, edges[1:-1], right=False)
    out[~nan_mask] = idx
    out[nan_mask] = missing_bin
    return out


def bin_counts(bins, n_bins, missing_bin=-1, y=None):
    """Per-bin count table.

    Returns a dict with:
      - 'n' : counts per bin index 0..n_bins-1 (arr of len n_bins)
      - 'n_missing' : count of missing bin
      - 'pos': positives per bin (only if y given)
      - 'pos_missing': positives in missing bin (only if y given)
    """
    bins = np.asarray(bins, dtype=np.int64)
    miss_mask = bins == missing_bin
    n_missing = int(miss_mask.sum())
    non_miss = bins[~miss_mask]
    n = np.bincount(non_miss, minlength=n_bins).astype(np.int64)[:n_bins]
    out = {"n": n, "n_missing": n_missing}
    if y is not None:
        y = np.asarray(y)
        pos = np.bincount(non_miss, weights=(y[~miss_mask] == 1).astype(np.int64),
                          minlength=n_bins).astype(np.int64)[:n_bins]
        pos_missing = int(((y[miss_mask] == 1).astype(np.int64)).sum())
        out["pos"] = pos
        out["pos_missing"] = pos_missing
    return out
