"""yyyymmdd integer-based time splitting.

Why integer compare instead of datetime parsing:
- 45k-row dataset × chunked passes → datetime parse overhead adds up
- yyyymmdd preserves ordering under integer comparison
- Avoids pandas datetime inference surprises
"""
import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_yyyymmdd_int(series):
    """Accept int or str yyyymmdd; return int64 array. Raises on invalid."""
    s = pd.Series(series).dropna()
    if pd.api.types.is_integer_dtype(s):
        arr = s.astype(np.int64).values
    else:
        arr = s.astype(str).str.replace("-", "", regex=False).astype(np.int64).values
    if arr.size:
        lo, hi = arr.min(), arr.max()
        if lo < 19000101 or hi > 30001231:
            raise ValueError("yyyymmdd out of plausible range: [{0}, {1}]".format(lo, hi))
    return arr


def split_by_yyyymmdd(series, ratios):
    """Return three boolean masks (train, valid, oot) by sorting and slicing.

    ratios: e.g., [0.7, 0.15, 0.15].
    Ties at boundary go to the earlier split.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must have 3 elements")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    ser = pd.Series(series).reset_index(drop=True)
    ints = pd.Series(to_yyyymmdd_int(ser), index=ser.dropna().index)
    ints = ints.reindex(ser.index)  # preserve original length; NaN rows become NaN
    order = ints.rank(method="first", na_option="bottom").values
    n_total = ser.size
    n_tr = int(round(ratios[0] * n_total))
    n_va = int(round(ratios[1] * n_total))
    n_tr = max(1, n_tr)
    n_va = max(1, n_va)
    n_oot = max(1, n_total - n_tr - n_va)
    # order is 1-indexed rank
    mask_tr = order <= n_tr
    mask_va = (order > n_tr) & (order <= n_tr + n_va)
    mask_oot = order > (n_tr + n_va)
    return mask_tr, mask_va, mask_oot


def split_stratified(y, ratios, seed=42):
    """Stratified split of boolean/int label y into train/valid/oot masks."""
    if len(ratios) != 3:
        raise ValueError("ratios must have 3 elements")
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n = y.size
    mask_tr = np.zeros(n, dtype=bool)
    mask_va = np.zeros(n, dtype=bool)
    mask_oot = np.zeros(n, dtype=bool)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_lab = idx.size
        n_tr = int(round(ratios[0] * n_lab))
        n_va = int(round(ratios[1] * n_lab))
        n_tr = max(0, min(n_lab, n_tr))
        n_va = max(0, min(n_lab - n_tr, n_va))
        mask_tr[idx[:n_tr]] = True
        mask_va[idx[n_tr:n_tr + n_va]] = True
        mask_oot[idx[n_tr + n_va:]] = True
    return mask_tr, mask_va, mask_oot


def split_psi_halves(yyyymmdd_series):
    """Split into earlier/later halves by yyyymmdd — used when oot_path is absent."""
    ints = to_yyyymmdd_int(yyyymmdd_series)
    if ints.size < 2:
        raise ValueError("Not enough non-null yyyymmdd values for split")
    mid = int(np.median(ints))
    ser = pd.Series(yyyymmdd_series)
    early_mask = pd.to_numeric(ser.astype(str).str.replace("-", "", regex=False),
                               errors="coerce") <= mid
    later_mask = ~early_mask
    return early_mask.fillna(False).values, later_mask.fillna(False).values
