"""Population Stability Index (PSI) — NaN-aware.

Standard formula: PSI = Σ (A_i - E_i) * ln(A_i / E_i) with ε clipping to avoid log(0).

Usage:
  1. Compute quantile edges from train (full pass through the column).
  2. For each chunk, count train-bin vs oot-bin members via np.digitize + np.bincount.
  3. Convert counts to percentages, apply the PSI formula.

Flags:
  stable  : PSI < 0.10
  shift   : 0.10 <= PSI < 0.25
  broken  : PSI >= 0.25
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from wdm.utils.binning import digitize_with_missing, equal_freq_edges

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _psi_from_counts(expected_pct, actual_pct):
    expected_pct = np.asarray(expected_pct, dtype=np.float64)
    actual_pct = np.asarray(actual_pct, dtype=np.float64)
    expected_pct = np.where(expected_pct <= 0, _EPS, expected_pct)
    actual_pct = np.where(actual_pct <= 0, _EPS, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_psi(expected_values_nan, actual_values_nan,
                edges=None, n_bins=10, missing_as_bin=True):
    """Compute PSI between two NaN-aware arrays.

    If edges is None, edges are computed from `expected_values_nan` (equal-freq quantiles).
    """
    e_arr = np.asarray(expected_values_nan, dtype=np.float64)
    a_arr = np.asarray(actual_values_nan, dtype=np.float64)
    if edges is None:
        edges = equal_freq_edges(e_arr, n_bins=n_bins)
    if edges.size < 2:
        return 0.0
    n_real = edges.size - 1
    e_bins = digitize_with_missing(e_arr, edges)
    a_bins = digitize_with_missing(a_arr, edges)

    def _pct(bins):
        total = bins.size
        miss = int((bins == -1).sum())
        non = bins[bins != -1]
        cnt = np.bincount(non, minlength=n_real)[:n_real]
        if missing_as_bin:
            full = np.concatenate([cnt, [miss]])
        else:
            full = cnt
            total = max(1, total - miss)
        if total <= 0:
            return np.zeros_like(full, dtype=np.float64)
        return full.astype(np.float64) / float(total)

    return _psi_from_counts(_pct(e_bins), _pct(a_bins))


def flag(psi_value):
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.25:
        return "shift"
    return "broken"


def compute_psi_table(chunk_iter_train, chunk_iter_oot, spec_map, feature_names,
                     cfg, get_spec_fn):
    """Compute a PSI table comparing train vs oot.

    chunk_iter_train and chunk_iter_oot yield chunks over the SAME feature
    blocks (same feature ordering), but with different rows (train vs oot).
    """
    n_bins_cfg = int(cfg["analysis"].get("n_bins", 10))

    rows = []
    # We iterate both iterators in lockstep.
    for (df_tr, block_tr), (df_oot, block_oot) in zip(chunk_iter_train, chunk_iter_oot):
        if block_tr != block_oot:
            raise ValueError("train/oot chunk iterators out of sync")
        for feat in block_tr:
            spec = get_spec_fn(spec_map, feat)
            from wdm.preprocess.missing import to_nan_array
            arr_tr, _ = to_nan_array(df_tr[feat], spec)
            arr_oot, _ = to_nan_array(df_oot[feat], spec)
            edges = equal_freq_edges(arr_tr, n_bins=n_bins_cfg)
            psi = compute_psi(arr_tr, arr_oot, edges=edges, n_bins=n_bins_cfg,
                              missing_as_bin=True)
            rows.append({"feature": feat, "psi": float(psi), "flag": flag(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def compute_psi_table_single_source(chunk_iter, mask_expected, mask_actual,
                                    spec_map, cfg, get_spec_fn):
    """When there is no OOT file: split the single CSV by boolean masks
    (e.g., yyyymmdd halves, or last N% chronologically) and compute PSI.

    mask_expected / mask_actual are boolean numpy arrays aligned to the CSV's
    total row count.
    """
    n_bins_cfg = int(cfg["analysis"].get("n_bins", 10))
    m_e = np.asarray(mask_expected, dtype=bool)
    m_a = np.asarray(mask_actual, dtype=bool)

    rows = []
    for df_chunk, block in chunk_iter:
        if len(df_chunk) != m_e.shape[0]:
            raise ValueError("chunk rows != mask length")
        for feat in block:
            spec = get_spec_fn(spec_map, feat)
            from wdm.preprocess.missing import to_nan_array
            full, _ = to_nan_array(df_chunk[feat], spec)
            arr_e = full[m_e]
            arr_a = full[m_a]
            edges = equal_freq_edges(arr_e, n_bins=n_bins_cfg)
            psi = compute_psi(arr_e, arr_a, edges=edges, n_bins=n_bins_cfg,
                              missing_as_bin=True)
            rows.append({"feature": feat, "psi": float(psi), "flag": flag(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
