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


def build_forward_chaining_folds(dt_values, n_folds, min_test_rows=50):
    """Expanding-window CV folds for time-ordered data (hyperopt tuning).

    dt_values: 1-D array of yyyymmdd ints/floats aligned to the train rows;
    NaN allowed. Rows are partitioned into n_folds+1 contiguous time blocks of
    roughly equal row count, but cut points are snapped to calendar-day
    boundaries so a single day never straddles two blocks — this guarantees a
    strict max(train_dt) < min(test_dt) for every fold. Fold i trains on
    blocks 0..i and validates on block i+1. NaN-dt rows go to block 0 (always
    train, never test). Test blocks smaller than min_test_rows are merged into
    the previous block (fewer folds, with a warning).

    Returns a list of (train_idx, test_idx) int64 array tuples, len <= n_folds.
    """
    dt = pd.to_numeric(pd.Series(dt_values), errors="coerce").values.astype(np.float64)
    n = dt.size
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    valid = ~np.isnan(dt)
    if valid.sum() < 2:
        raise ValueError("Not enough non-null dt values for time folds")

    days, counts = np.unique(dt[valid], return_counts=True)
    n_blocks = n_folds + 1
    if days.size < n_blocks:
        n_blocks = int(days.size)
        logger.warning("Only %d unique days < requested %d blocks; reducing to %d folds",
                       days.size, n_folds + 1, n_blocks - 1)
        if n_blocks < 2:
            raise ValueError("Need at least 2 distinct days for forward-chaining folds")

    # Day-boundary cut points at ~equal cumulative row counts.
    cum = np.cumsum(counts)
    total = int(cum[-1])
    targets = [int(round(total * (i + 1) / float(n_blocks))) for i in range(n_blocks - 1)]
    cut_days = []
    for t in targets:
        pos = int(np.searchsorted(cum, max(1, t)))
        pos = min(pos, days.size - 2)  # keep at least one day for the last block
        day = days[pos]
        if cut_days and day <= cut_days[-1]:
            # Snap collisions forward to the next unused day.
            nxt = int(np.searchsorted(days, cut_days[-1], side="right"))
            if nxt >= days.size - 1:
                continue
            day = days[nxt]
        cut_days.append(day)
    # block_id per row: number of cut days strictly below the row's dt.
    # Rows with dt <= cut_days[0] -> block 0, etc. NaN -> block 0.
    block_id = np.zeros(n, dtype=np.int64)
    block_id[valid] = np.searchsorted(np.asarray(cut_days), dt[valid], side="left")

    folds = []
    last_block = int(block_id.max())
    merged_into_prev = 0
    for b in range(1, last_block + 1):
        test_idx = np.where(block_id == b)[0]
        if test_idx.size < min_test_rows and folds:
            # Merge a tiny test block into the previous fold's test set.
            prev_train, prev_test = folds[-1]
            folds[-1] = (prev_train, np.concatenate([prev_test, test_idx]))
            merged_into_prev += 1
            continue
        train_idx = np.where((block_id < b) & valid)[0]
        train_idx = np.concatenate([train_idx, np.where(~valid)[0]]).astype(np.int64)
        train_idx.sort()
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx.astype(np.int64)))
    if merged_into_prev:
        logger.warning("Merged %d small test blocks (< %d rows) into their predecessors; "
                       "%d folds remain", merged_into_prev, min_test_rows, len(folds))
    if not folds:
        raise ValueError("Could not build any forward-chaining fold")
    return folds


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
