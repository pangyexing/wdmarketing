"""Kolmogorov-Smirnov statistic on (y_true, y_score)."""
import numpy as np


def ks_stat(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-y_score, kind="stable")
    ys = y_true[order]
    cum_pos = np.cumsum(ys)
    cum_neg = np.cumsum(1 - ys)
    tpr = cum_pos / n_pos
    fpr = cum_neg / n_neg
    return float(np.max(np.abs(tpr - fpr)))
