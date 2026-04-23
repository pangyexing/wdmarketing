"""Top-K ranking metrics for imbalanced marketing scoring.

All functions accept:
  y_true: 0/1 label array
  y_score: predicted probability / score (higher = more likely positive)

Contract: `k` is either an integer count or a float in (0, 1] meaning fraction.
Results: Precision@K, Recall@K, Lift@K, Top-K CVR.
"""
import math
from typing import Union

import numpy as np


def _top_k_mask(y_score, k):
    y_score = np.asarray(y_score, dtype=np.float64)
    n = y_score.size
    if isinstance(k, float) and 0 < k <= 1:
        k_int = max(1, int(math.ceil(k * n)))
    else:
        k_int = int(k)
    if k_int <= 0:
        raise ValueError("k must be positive")
    k_int = min(k_int, n)
    # argpartition finds the top-k indices in O(n); break ties deterministically
    order = np.argsort(-y_score, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k_int]] = True
    return mask, k_int


def precision_at_k(y_true, y_score, k):
    mask, k_int = _top_k_mask(y_score, k)
    y_true = np.asarray(y_true).astype(np.int64)
    return float(y_true[mask].sum()) / k_int


def recall_at_k(y_true, y_score, k):
    mask, _ = _top_k_mask(y_score, k)
    y_true = np.asarray(y_true).astype(np.int64)
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0
    return float(y_true[mask].sum()) / total_pos


def lift_at_k(y_true, y_score, k):
    mask, k_int = _top_k_mask(y_score, k)
    y_true = np.asarray(y_true).astype(np.int64)
    n = y_true.size
    base_rate = y_true.mean() if n else 0.0
    if base_rate == 0.0:
        return 0.0
    top_rate = y_true[mask].mean()
    return float(top_rate / base_rate)


def top_k_cvr(y_true, y_score, k):
    """Top-K CVR = Precision@K, exposed as its own function for report clarity."""
    return precision_at_k(y_true, y_score, k)
