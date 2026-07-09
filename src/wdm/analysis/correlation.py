"""Pearson correlation across features with memory-safe block-wise computation.

Algorithm (two passes, mathematically equivalent to whole-table Pearson):

Pass 1: single iteration over column chunks → per-column mean, sum_sq,
        non-nan count (this is what "global mean/std" means here).

Pass 2: iterate over PAIRS of chunks (A, B). For each pair, load A and B
        together (still memory-safe; chunk_size × chunk_size = 2500 columns
        in the worst case, fits easily). Compute 50×50 covariance matrix
        using the GLOBAL means from Pass 1 — never block-local means.
        Store only edges with |r| >= threshold, to keep the edge list small.

NaN handling: pairwise complete — for each (i, j) pair, only rows where both
columns are non-NaN contribute. We track `n_pairs` per edge so the selector
can down-weight edges with low overlap.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _column_stats(arr_2d):
    """Given 2d float array (rows × features) with NaN for missing,
    return per-column (count, sum, sum_sq) arrays.
    """
    mask = ~np.isnan(arr_2d)
    arr0 = np.where(mask, arr_2d, 0.0)
    count = mask.sum(axis=0).astype(np.int64)
    s = arr0.sum(axis=0)
    s2 = (arr0 * arr0).sum(axis=0)
    return count, s, s2


def _pairwise_cov_block(A, B, muA, muB):
    """Compute cross-covariance between columns of A and columns of B using
    the provided per-column means.

    NaN entries in either A or B suppress that row for that column-pair.
    Returns (cov, n_pairs) — both 2d arrays of shape (nA, nB).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mA = ~np.isnan(A)
    mB = ~np.isnan(B)
    # Centered values, zeroed where missing. Because missing entries are 0 in
    # A0/B0, (A0.T @ B0)[i,j] sums (A[:,i]-muA[i])·(B[:,j]-muB[j]) over exactly
    # the rows where both mA[:,i] and mB[:,j] are True — the per-pair mask
    # never needs to be materialized.
    A0 = np.where(mA, A - muA[np.newaxis, :], 0.0)
    B0 = np.where(mB, B - muB[np.newaxis, :], 0.0)
    mA_i = mA.astype(np.float64)
    mB_i = mB.astype(np.float64)
    pair_counts = mA_i.T @ mB_i        # (nA, nB)
    cov_sum = A0.T @ B0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(pair_counts > 0, cov_sum / pair_counts, 0.0)
    return cov, pair_counts


def finalize_column_stats(count, sum_, sum_sq):
    """Derive per-column (mean, std) from accumulated Pass-1 statistics."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sum_ / count, 0.0)
        var = np.where(count > 0, sum_sq / count - mean * mean, 0.0)
        std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std


def _collect_block_edges(r, pairs, blockA, blockB, same_block,
                         threshold, total_rows, min_overlap_frac):
    """Vectorized edge collection for one block-pair correlation matrix.

    np.argwhere scans row-major, so the emitted edge order matches the
    original i-outer / j-inner element loop exactly.
    """
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(r) & (np.abs(r) >= threshold)
    if same_block:
        valid &= np.triu(np.ones(r.shape, dtype=bool), k=1)
    edges = []
    for i, j in np.argwhere(valid):
        np_pairs = int(pairs[i, j])
        frac = np_pairs / float(total_rows) if total_rows else 0.0
        edges.append({
            "f1": blockA[i],
            "f2": blockB[j],
            "r": float(r[i, j]),
            "n_pairs": np_pairs,
            "low_overlap": bool(frac < min_overlap_frac),
        })
    return edges


def compute_correlation_edges(features,
                              path,
                              always,
                              spec_map,
                              get_spec_fn,
                              chunk_size=50,
                              threshold=0.95,
                              min_overlap_frac=0.10,
                              row_mask=None):
    """Two-pass block-wise Pearson correlation.

    Returns a DataFrame of edges with columns:
        f1, f2, r, n_pairs, low_overlap
    Only includes |r| >= threshold. `low_overlap` marks pairs where the
    pairwise-complete overlap fraction is below min_overlap_frac — the
    selector should ignore these.

    row_mask: optional boolean mask; when given only those rows enter both
    passes — the same semantics as the scan-cache path, where the cached
    blocks are saved pre-masked (analysis.unsupervised_stats_split).
    """
    from wdm.io.chunked_reader import iter_column_chunks
    from wdm.preprocess.missing import to_nan_array
    from wdm.utils.progress import ProgressCounter, track

    features = list(features)
    n_chunks = (len(features) + chunk_size - 1) // chunk_size
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.all():
            row_mask = None

    # ---- Pass 1: global means + sum_sq ----
    feat_idx = {f: i for i, f in enumerate(features)}
    count = np.zeros(len(features), dtype=np.int64)
    sum_ = np.zeros(len(features), dtype=np.float64)
    sum_sq = np.zeros(len(features), dtype=np.float64)
    total_rows = None

    for df_chunk, block in track(
            iter_column_chunks(path, features, always=always, chunk_size=chunk_size),
            total=n_chunks, label="correlation pass1 chunks"):
        n_rows = len(df_chunk) if row_mask is None else int(row_mask.sum())
        if total_rows is None:
            total_rows = n_rows
        # Convert each column in block to NaN-aware
        cols = np.empty((n_rows, len(block)), dtype=np.float64)
        for j, feat in enumerate(block):
            spec = get_spec_fn(spec_map, feat)
            arr, _ = to_nan_array(df_chunk[feat], spec, analysis=True)
            cols[:, j] = arr if row_mask is None else arr[row_mask]
        c, s, s2 = _column_stats(cols)
        for j, feat in enumerate(block):
            i = feat_idx[feat]
            count[i] += c[j]
            sum_[i] += s[j]
            sum_sq[i] += s2[j]

    mean, std = finalize_column_stats(count, sum_, sum_sq)

    # ---- Pass 2: block-pair covariance ----
    blocks = [features[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    pair_prog = ProgressCounter("correlation pass2 block-pairs",
                                total=n_chunks * (n_chunks + 1) // 2)

    edges = []
    for bi in range(n_chunks):
        blockA = blocks[bi]
        idxA = np.array([feat_idx[f] for f in blockA])
        muA = mean[idxA]
        stdA = std[idxA]
        # Load blockA once
        always_set = list(set(always) | set(blockA))
        # Single-chunk read for efficiency; reuse pd.read_csv
        dfA = pd.read_csv(path, usecols=always_set)
        dfA = dfA[always_set]  # ensure order
        logger.info("[corr pass 2 cov] block %d/%d loaded (%d features); pairing with %d remaining blocks",
                    bi + 1, n_chunks, len(blockA), n_chunks - bi)
        # Build nan-aware matrix for blockA
        n_a = len(dfA) if row_mask is None else int(row_mask.sum())
        A = np.empty((n_a, len(blockA)), dtype=np.float64)
        for j, feat in enumerate(blockA):
            spec = get_spec_fn(spec_map, feat)
            arr, _ = to_nan_array(dfA[feat], spec, analysis=True)
            A[:, j] = arr if row_mask is None else arr[row_mask]

        for bj in range(bi, n_chunks):  # symmetric; include diagonal for within-block pairs
            blockB = blocks[bj]
            idxB = np.array([feat_idx[f] for f in blockB])
            muB = mean[idxB]
            stdB = std[idxB]
            if bi == bj:
                B = A
            else:
                always_set_b = list(set(always) | set(blockB))
                dfB = pd.read_csv(path, usecols=always_set_b)
                dfB = dfB[always_set_b]
                n_b = len(dfB) if row_mask is None else int(row_mask.sum())
                B = np.empty((n_b, len(blockB)), dtype=np.float64)
                for j, feat in enumerate(blockB):
                    spec = get_spec_fn(spec_map, feat)
                    arr, _ = to_nan_array(dfB[feat], spec, analysis=True)
                    B[:, j] = arr if row_mask is None else arr[row_mask]

            cov, pairs = _pairwise_cov_block(A, B, muA, muB)
            denom = np.outer(stdA, stdB)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(denom > 0, cov / denom, 0.0)
            edges.extend(_collect_block_edges(
                r, pairs, blockA, blockB, bi == bj,
                threshold, total_rows, min_overlap_frac))
            pair_prog.tick(extra="{0} edges so far".format(len(edges)))
    df = pd.DataFrame(edges, columns=["f1", "f2", "r", "n_pairs", "low_overlap"])
    if not df.empty:
        df = df.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df


def _validate_cache_manifest(cache_dir, blocks, n_rows):
    from wdm.analysis.feature_scan import MANIFEST_NAME

    manifest_path = Path(cache_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "scan cache manifest missing: {0}".format(manifest_path))
    with open(str(manifest_path), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if (manifest.get("blocks") != [list(b) for b in blocks]
            or int(manifest.get("n_rows", -1)) != int(n_rows)):
        raise ValueError(
            "scan cache manifest does not match the requested feature blocks "
            "(stale cache?): {0}".format(cache_dir))


def compute_edges_from_cache(features, blocks, cache_dir, count, sum_, sum_sq,
                             n_rows, threshold=0.95, min_overlap_frac=0.10,
                             mmap=True):
    """Correlation Pass-2 fed by the single-pass scan's .npy block cache.

    Numerically identical to compute_correlation_edges: same Pass-1 statistics
    (accumulated by feature_scan and passed in), same block-pair covariance,
    same edge collection — only the data source differs (np.load of the cached
    NaN-aware float64 blocks instead of re-parsing the CSV per block pair).
    """
    from wdm.analysis.feature_scan import block_cache_path
    from wdm.utils.progress import ProgressCounter

    features = list(features)
    cache_dir = Path(cache_dir)
    _validate_cache_manifest(cache_dir, blocks, n_rows)

    feat_idx = {f: i for i, f in enumerate(features)}
    mean, std = finalize_column_stats(count, sum_, sum_sq)
    n_chunks = len(blocks)
    mode = "r" if mmap else None
    total_rows = int(n_rows)
    pair_prog = ProgressCounter("correlation pass2 block-pairs (cached)",
                                total=n_chunks * (n_chunks + 1) // 2)

    edges = []
    for bi in range(n_chunks):
        blockA = blocks[bi]
        idxA = np.array([feat_idx[f] for f in blockA])
        muA = mean[idxA]
        stdA = std[idxA]
        A = np.load(str(block_cache_path(cache_dir, bi)), mmap_mode=mode)
        for bj in range(bi, n_chunks):
            blockB = blocks[bj]
            idxB = np.array([feat_idx[f] for f in blockB])
            muB = mean[idxB]
            stdB = std[idxB]
            if bi == bj:
                B = A
            else:
                B = np.load(str(block_cache_path(cache_dir, bj)), mmap_mode=mode)
            cov, pairs = _pairwise_cov_block(A, B, muA, muB)
            denom = np.outer(stdA, stdB)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(denom > 0, cov / denom, 0.0)
            edges.extend(_collect_block_edges(
                r, pairs, blockA, blockB, bi == bj,
                threshold, total_rows, min_overlap_frac))
            pair_prog.tick(extra="{0} edges so far".format(len(edges)))
    df = pd.DataFrame(edges, columns=["f1", "f2", "r", "n_pairs", "low_overlap"])
    if not df.empty:
        df = df.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df


def cluster_correlated(edges_df, features):
    """Union-find clustering of features connected by high-correlation edges.

    `edges_df` rows with low_overlap=True are ignored.
    Returns Dict[cluster_id, List[feature]] — singleton clusters are OK.
    """
    parent = {f: f for f in features}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if edges_df is not None and not edges_df.empty:
        for _, row in edges_df.iterrows():
            if row.get("low_overlap", False):
                continue
            if row["f1"] in parent and row["f2"] in parent:
                union(row["f1"], row["f2"])

    clusters = {}
    for f in features:
        root = find(f)
        clusters.setdefault(root, []).append(f)
    # Reindex
    return {i: members for i, members in enumerate(clusters.values())}


def cluster_id_per_feature(clusters):
    """Invert the cluster dict into {feature: cluster_id}."""
    out = {}
    for cid, members in clusters.items():
        for f in members:
            out[f] = cid
    return out
