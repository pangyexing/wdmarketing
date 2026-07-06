"""Per-feature Lift & Gini analysis — aligned with the "top-K ranking"
objective of the marketing project.

Semantics: for each feature, measure how well it alone ranks positives to the
top. Used as a feature-quality signal complementary to IV/PSI.

Metrics:
  * lift_at_k    — cum_pos_rate_in_top_K / base_rate
  * gini         — 2 · AUC(feature_as_score, y) − 1 with auto-flip for sign
  * concentration — fraction of total positives captured in the top bin

When cfg['data']['treatment_column'] is set, this module switches to T-learner
uplift (not implemented in the UCI demo; see plan).
"""
import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from wdm.utils.binning import (
    bin_counts, digitize_with_missing, equal_freq_edges,
)

logger = logging.getLogger(__name__)


def _feature_gini(values_nan, y):
    """2·AUC - 1 with sign flip: use whichever direction (ascending/descending)
    gives a positive score, so Gini represents "how well this feature ranks
    positives".
    """
    y = np.asarray(y).astype(np.int64)
    values = np.asarray(values_nan, dtype=np.float64)
    mask = ~np.isnan(values)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return 0.0
    # Fill NaN with median so AUC uses full data; tied at median
    median = float(np.nanmedian(values)) if np.any(mask) else 0.0
    filled = np.where(mask, values, median)
    try:
        auc = roc_auc_score(y, filled)
    except ValueError:
        return 0.0
    gini = 2.0 * auc - 1.0
    return float(abs(gini))


def _lift_at_k_from_bins(bin_specs, top_k_pct, total_pos, total_n):
    """Sum the top bins (by positive rate) until cumulative sample fraction
    crosses top_k_pct. Returns (lift_at_k, concentration_top_bin).
    """
    records = []
    for i, (n, pos) in enumerate(bin_specs):
        if n <= 0:
            continue
        records.append((pos / n, n, pos))
    records.sort(key=lambda r: r[0], reverse=True)
    k_target = max(1, int(math.ceil(top_k_pct * total_n)))
    base_rate = (total_pos / total_n) if total_n > 0 else 0.0
    cum_n, cum_pos = 0, 0
    concentration = 0.0
    for pr, n, pos in records:
        if cum_n >= k_target:
            break
        cum_n += n
        cum_pos += pos
        if concentration == 0.0:
            concentration = pos / total_pos if total_pos > 0 else 0.0
    if cum_n == 0 or base_rate == 0.0:
        return 0.0, concentration
    lift = (cum_pos / cum_n) / base_rate
    return float(lift), float(concentration)


def lift_row_from_array(arr, y, feat, total_pos, total_n, top_k_pct, n_bins):
    """Per-feature kernel shared by compute_feature_lift_table and the
    single-pass scan. Returns {feature, lift_at_k, gini, concentration}."""
    edges = equal_freq_edges(arr, n_bins=n_bins)
    if edges.size < 2:
        return {"feature": feat, "lift_at_k": 1.0, "gini": 0.0,
                "concentration": 0.0}
    n_real = edges.size - 1
    bins = digitize_with_missing(arr, edges)
    bc = bin_counts(bins, n_bins=n_real, y=y)
    # missing bin participates as its own bucket
    bin_specs = list(zip(bc["n"], bc["pos"]))
    if bc["n_missing"] > 0:
        bin_specs.append((bc["n_missing"], bc["pos_missing"]))
    lift, conc = _lift_at_k_from_bins(bin_specs, top_k_pct, total_pos, total_n)
    gini = _feature_gini(arr, y)
    return {"feature": feat, "lift_at_k": lift, "gini": gini,
            "concentration": conc}


def compute_feature_lift_table(chunk_iter, spec_map, y_series, cfg, get_spec_fn):
    """Compute per-feature lift table.

    Returns DataFrame[feature, lift_at_k, gini, concentration].
    """
    from wdm.preprocess.missing import to_nan_array

    y = y_series.values.astype(np.int64)
    total_n = y.size
    total_pos = int(y.sum())
    top_k_pct = float(cfg["training"].get("top_k_pct", 0.10))
    n_bins_cfg = int(cfg["analysis"].get("n_bins", 10))

    rows = []
    for df_chunk, block in chunk_iter:
        if len(df_chunk) != total_n:
            raise ValueError("chunk rows != label rows")
        for feat in block:
            spec = get_spec_fn(spec_map, feat)
            arr, _ = to_nan_array(df_chunk[feat], spec, analysis=True)
            rows.append(lift_row_from_array(arr, y, feat, total_pos, total_n,
                                            top_k_pct, n_bins_cfg))
    df = pd.DataFrame(rows).sort_values("lift_at_k", ascending=False).reset_index(drop=True)
    return df
