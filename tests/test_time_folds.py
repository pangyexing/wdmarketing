"""Forward-chaining time folds: strict no-leakage, expanding windows, NaN/dup handling."""
import numpy as np
import pytest

from wdm.utils.time_utils import build_forward_chaining_folds


def _dts(n_days=60, rows_per_day=None, seed=0, start=20250101):
    rng = np.random.RandomState(seed)
    if rows_per_day is None:
        rows_per_day = rng.randint(5, 60, size=n_days)
    days = []
    d = start
    for c in rows_per_day:
        days.extend([d] * int(c))
        d += 1  # fake-but-ordered yyyymmdd increments are fine for the fold logic
    arr = np.array(days, dtype=np.float64)
    rng.shuffle(arr)  # row order must not matter
    return arr


def test_strict_no_time_leakage_and_expanding():
    dt = _dts()
    folds = build_forward_chaining_folds(dt, n_folds=5)
    assert 1 <= len(folds) <= 5
    prev_train = None
    seen_test = set()
    for train_idx, test_idx in folds:
        assert train_idx.size > 0 and test_idx.size > 0
        # STRICT: every train dt earlier than every test dt
        assert np.nanmax(dt[train_idx]) < np.min(dt[test_idx])
        # expanding window: each train set contains the previous one
        if prev_train is not None:
            assert set(prev_train).issubset(set(train_idx.tolist()))
        prev_train = train_idx.tolist()
        # test blocks disjoint
        assert not seen_test.intersection(test_idx.tolist())
        seen_test.update(test_idx.tolist())


def test_heavy_day_never_straddles_blocks():
    # One day holds ~40% of all rows; it must land wholly in one block.
    rows_per_day = [10] * 30
    rows_per_day[10] = 200
    dt = _dts(n_days=30, rows_per_day=rows_per_day)
    folds = build_forward_chaining_folds(dt, n_folds=4, min_test_rows=1)
    heavy_day = np.float64(20250101 + 10)
    for train_idx, test_idx in folds:
        in_train = np.any(dt[train_idx] == heavy_day)
        in_test = np.any(dt[test_idx] == heavy_day)
        assert not (in_train and in_test), "a single day straddles train/test"


def test_nan_rows_train_only():
    dt = _dts(n_days=20)
    dt[::7] = np.nan
    folds = build_forward_chaining_folds(dt, n_folds=3, min_test_rows=1)
    nan_idx = set(np.where(np.isnan(dt))[0].tolist())
    for train_idx, test_idx in folds:
        assert nan_idx.issubset(set(train_idx.tolist())), "NaN rows must always train"
        assert not nan_idx.intersection(test_idx.tolist()), "NaN rows must never test"


def test_fewer_days_than_folds_degrades_gracefully():
    dt = _dts(n_days=3, rows_per_day=[40, 40, 40])
    folds = build_forward_chaining_folds(dt, n_folds=10, min_test_rows=1)
    assert 1 <= len(folds) <= 2
    for train_idx, test_idx in folds:
        assert train_idx.size > 0 and test_idx.size > 0


def test_small_test_blocks_merged():
    # 6 days: the last blocks would be tiny; min_test_rows forces merging.
    rows_per_day = [100, 100, 100, 3, 2, 2]
    dt = _dts(n_days=6, rows_per_day=rows_per_day)
    folds = build_forward_chaining_folds(dt, n_folds=5, min_test_rows=50)
    for _train_idx, test_idx in folds:
        assert test_idx.size >= 50 or len(folds) == 1


def test_errors():
    with pytest.raises(ValueError):
        build_forward_chaining_folds(np.array([np.nan, np.nan]), n_folds=3)
    with pytest.raises(ValueError):
        build_forward_chaining_folds(np.array([20250101.0] * 100), n_folds=3)
