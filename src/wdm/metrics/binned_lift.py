"""Decile-binned lift / gain DataFrame for reporting and plotting."""
import math
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_binned_lift(y_true, y_score, n_bins=10):
    """Return a DataFrame with per-decile counts and cumulative metrics.

    Columns:
      bin, n_samples, n_positives, pos_rate, cum_samples, cum_positives,
      cum_recall, cum_precision, cum_lift, cum_pop_share
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = y_true.size
    total_pos = int(y_true.sum())
    base_rate = total_pos / n if n > 0 else 0.0

    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]

    # split into n_bins groups as evenly as possible
    edges = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]
    rows = []
    cum_n, cum_pos = 0, 0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        group = y_sorted[lo:hi]
        n_g = len(group)
        pos_g = int(group.sum())
        cum_n += n_g
        cum_pos += pos_g
        rows.append({
            "bin": b + 1,
            "n_samples": n_g,
            "n_positives": pos_g,
            "pos_rate": pos_g / n_g if n_g else 0.0,
            "cum_samples": cum_n,
            "cum_positives": cum_pos,
            "cum_pop_share": cum_n / n if n else 0.0,
            "cum_recall": cum_pos / total_pos if total_pos else 0.0,
            "cum_precision": cum_pos / cum_n if cum_n else 0.0,
            "cum_lift": (cum_pos / cum_n) / base_rate if cum_n and base_rate else 0.0,
        })
    return pd.DataFrame(rows)
