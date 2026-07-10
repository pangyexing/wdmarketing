"""Single-pass Stage-1 feature scan.

One iteration over column chunks computes, per feature:
  * the IV/WOE report row + BinSpec        (iv_woe.iv_row_from_array)
  * the missing-stats row                  (missing.missing_row_from_array)
  * the lift/gini row                      (lift.lift_row_from_array)
  * the PSI row                            (psi.psi_row_from_array)
and accumulates the correlation Pass-1 statistics (per-column non-NaN
count/sum/sum_sq). With a cache_dir, each block's NaN-aware float64 matrix is
also saved as block_NNNN.npy so correlation Pass-2
(correlation.compute_edges_from_cache) never re-parses the CSV.

This replaces four independent chunk iterations (IV, missing, lift, PSI) plus
correlation Pass-1 — the CSV is parsed once per chunk instead of 5+ times, and
to_nan_array runs once per feature instead of ~6 times.

Equivalence contract: the four DataFrames are constructed in the same row
order and sorted exactly like the legacy per-signal functions, so outputs are
numerically identical (guarded by tests/test_feature_scan_equivalence.py and
tests/test_stage1_golden.py).
"""
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from wdm.analysis.correlation import _column_stats
from wdm.analysis.iv_woe import iv_row_from_array
from wdm.analysis.lift import lift_row_from_array
from wdm.analysis.missing_stats import missing_row_from_array
from wdm.analysis.psi import flag_thresholds, psi_row_from_array

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclasses.dataclass
class ScanResult:
    iv_df: pd.DataFrame
    bin_specs: Dict
    miss_df: pd.DataFrame
    lift_df: pd.DataFrame
    psi_df: pd.DataFrame
    col_count: np.ndarray          # correlation Pass-1, aligned to `features`
    col_sum: np.ndarray
    col_sum_sq: np.ndarray
    n_rows: int
    blocks: List[List[str]]        # chunk layout, aligned to block_NNNN.npy
    cache_dir: Optional[Path]      # None when block caching is disabled


def block_cache_path(cache_dir, block_index):
    return Path(cache_dir) / "block_{0:04d}.npy".format(block_index)


def run_feature_scan(path, features, y_series, mask_expected, mask_actual,
                     spec_map, get_spec_fn, cfg, cache_dir=None,
                     supervised_mask=None, unsupervised_mask=None):
    """One pass over iter_column_chunks → ScanResult.

    cache_dir, when given, must be an existing (preferably empty, run-private)
    directory; it receives block_NNNN.npy files + a manifest.json.

    supervised_mask: optional boolean row mask. When given, the label-driven
    statistics (IV/WOE, bin edges, Lift@K, Gini) are fit on the masked rows
    only — pass the train-split mask so valid/OOT labels never influence
    feature selection.

    unsupervised_mask: optional boolean row mask for the label-free selection
    statistics (missing rate, correlation Pass-1 stats AND the cached .npy
    blocks Pass-2 reads — the two must see the same rows). Pass the
    train-split mask so valid/OOT feature distributions never influence the
    missing-rate gate or which cluster member survives de-duplication. PSI
    keeps its own mask_expected/mask_actual partition.
    """
    from wdm.io.chunked_reader import iter_column_chunks
    from wdm.preprocess.missing import to_nan_array
    from wdm.utils.progress import track

    features = list(features)
    chunk_size = int(cfg["io"]["column_chunk_size"])
    n_chunks = (len(features) + chunk_size - 1) // chunk_size
    blocks = [features[i * chunk_size:(i + 1) * chunk_size]
              for i in range(n_chunks)]

    n_expected_rows = y_series.values.shape[0]
    sup = None
    if supervised_mask is not None:
        sup = np.asarray(supervised_mask, dtype=bool)
        if sup.shape[0] != n_expected_rows:
            raise ValueError("supervised_mask length != y length")
    unsup = None
    if unsupervised_mask is not None:
        unsup = np.asarray(unsupervised_mask, dtype=bool)
        if unsup.shape[0] != n_expected_rows:
            raise ValueError("unsupervised_mask length != y length")
        if unsup.all():
            unsup = None  # full mask — take the cheaper unmasked path

    # Same y views as the legacy per-signal functions (optionally restricted
    # to the supervised rows).
    y_iv = y_series.values if sup is None else y_series.values[sup]
    y_lift = y_iv.astype(np.int64)
    total_n = y_lift.size
    total_pos = int(y_lift.sum())

    n_bins_cfg = int(cfg["analysis"].get("n_bins", 10))
    strategy = str(cfg["analysis"].get("binning", "equal_freq"))
    top_k_pct = float(cfg["training"].get("top_k_pct", 0.10))
    shift_t, broken_t = flag_thresholds(cfg)
    m_e = np.asarray(mask_expected, dtype=bool)
    m_a = np.asarray(mask_actual, dtype=bool)

    feat_idx = {f: i for i, f in enumerate(features)}
    col_count = np.zeros(len(features), dtype=np.int64)
    col_sum = np.zeros(len(features), dtype=np.float64)
    col_sum_sq = np.zeros(len(features), dtype=np.float64)

    cache_dir = Path(cache_dir) if cache_dir is not None else None

    bin_specs = {}
    iv_rows = []
    miss_rows = []
    lift_rows = []
    psi_rows = []
    seen = set()
    n_rows = None

    chunk_iter = iter_column_chunks(path, features, always=[],
                                    chunk_size=chunk_size)
    for chunk_i, (df_chunk, block) in enumerate(
            track(chunk_iter, total=n_chunks, label="single-pass scan chunks")):
        n_total = len(df_chunk)
        # Rows entering the label-free selection stats (missing, correlation)
        # and the cached blocks — Pass-2 must see exactly these rows.
        n_stat = n_total if unsup is None else int(unsup.sum())
        if n_rows is None:
            n_rows = n_stat
            if cache_dir is not None:
                est_gb = n_stat * len(features) * 8.0 / 1e9
                logger.info("scan cache enabled: ~%.2f GB of .npy blocks in %s",
                            est_gb, cache_dir)
        if n_total != n_expected_rows:
            raise ValueError(
                "chunk rows ({0}) != y rows ({1}); the CSV must be a single "
                "consistent file for chunked analysis.".format(
                    n_total, n_expected_rows))
        if n_total != m_e.shape[0]:
            raise ValueError("chunk rows != mask length")

        M = np.empty((n_stat, len(block)), dtype=np.float64)
        for j, feat in enumerate(block):
            spec = get_spec_fn(spec_map, feat)
            raw = df_chunk[feat]
            arr, mask = to_nan_array(raw, spec, analysis=True)
            M[:, j] = arr if unsup is None else arr[unsup]

            if feat in seen:
                continue
            seen.add(feat)
            arr_sup = arr if sup is None else arr[sup]
            iv_row, bs = iv_row_from_array(arr_sup, y_iv, feat, n_bins_cfg, strategy)
            bin_specs[feat] = bs
            iv_rows.append(iv_row)
            if unsup is None:
                miss_rows.append(missing_row_from_array(feat, str(raw.dtype),
                                                        arr, mask, n_total))
            else:
                miss_rows.append(missing_row_from_array(feat, str(raw.dtype),
                                                        arr[unsup], mask[unsup],
                                                        n_stat))
            lift_rows.append(lift_row_from_array(arr_sup, y_lift, feat, total_pos,
                                                 total_n, top_k_pct, n_bins_cfg))
            psi_rows.append(psi_row_from_array(arr, m_e, m_a, feat, n_bins_cfg,
                                               shift_t, broken_t))

        c, s, s2 = _column_stats(M)
        for j, feat in enumerate(block):
            i = feat_idx[feat]
            col_count[i] += c[j]
            col_sum[i] += s[j]
            col_sum_sq[i] += s2[j]

        if cache_dir is not None:
            np.save(str(block_cache_path(cache_dir, chunk_i)), M)

    if cache_dir is not None:
        manifest = {"n_rows": int(n_rows or 0), "blocks": blocks}
        with open(str(Path(cache_dir) / MANIFEST_NAME), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

    # Same per-signal sorts as the legacy table functions.
    iv_df = pd.DataFrame(iv_rows).sort_values(
        "iv", ascending=False).reset_index(drop=True)
    miss_df = pd.DataFrame(miss_rows).sort_values(
        "missing_rate", ascending=False).reset_index(drop=True)
    lift_df = pd.DataFrame(lift_rows).sort_values(
        "lift_at_k", ascending=False).reset_index(drop=True)
    psi_df = pd.DataFrame(psi_rows).sort_values(
        "psi", ascending=False).reset_index(drop=True)

    return ScanResult(iv_df=iv_df, bin_specs=bin_specs, miss_df=miss_df,
                      lift_df=lift_df, psi_df=psi_df,
                      col_count=col_count, col_sum=col_sum,
                      col_sum_sq=col_sum_sq,
                      n_rows=int(n_rows or 0), blocks=blocks,
                      cache_dir=cache_dir)
