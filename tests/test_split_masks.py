"""wdm.utils.split_masks — the single source of truth for train/valid/oot
masks shared by Stage-1 analysis and Stage-2 build_dataset. Guards the
alignment fix: exclude_rows and embargo_days must shape BOTH stages' "train"
identically."""
import numpy as np
import pandas as pd

from wdm.utils.split_masks import compute_split_masks, exclude_mask, scatter_masks
from wdm.utils.time_utils import split_by_yyyymmdd


def _cfg(strategy="stratified", ratios=None, embargo_days=0, exclude_rows=None,
         time_column=None):
    return {
        "data": {"label_column": "y", "time_column": time_column,
                 "exclude_rows": exclude_rows or []},
        "training": {
            "random_seed": 42,
            "split": {"strategy": strategy,
                      "ratios": ratios or [0.6, 0.2, 0.2],
                      "embargo_days": embargo_days},
        },
    }


def _time_frame(n_days=20, rows_per_day=30, seed=0):
    rng = np.random.RandomState(seed)
    days = np.repeat([20240101 + d for d in range(n_days)], rows_per_day)
    y = rng.randint(0, 2, size=days.size)
    flag = rng.randint(0, 3, size=days.size)  # exclude-rule column
    return pd.DataFrame({"dt": days, "y": y, "flag": flag})


def test_masks_partition_and_bool_dtype():
    df = _time_frame()
    m_tr, m_va, m_oot, included = compute_split_masks(df, _cfg())
    for m in (m_tr, m_va, m_oot, included):
        assert m.dtype == bool and m.shape == (len(df),)
    assert included.all()
    # stratified: every row lands in exactly one split
    assert int(m_tr.sum() + m_va.sum() + m_oot.sum()) == len(df)
    assert not (m_tr & m_va).any() and not (m_va & m_oot).any()


def test_excluded_rows_absent_from_every_mask():
    df = _time_frame()
    rules = [{"column": "flag", "values": [2]}]
    m_tr, m_va, m_oot, included = compute_split_masks(
        df, _cfg(exclude_rows=rules))
    dropped = (df["flag"] == 2).values
    assert (~included == dropped).all()
    for m in (m_tr, m_va, m_oot):
        assert not (m & dropped).any()
    # surviving rows still fully partitioned
    assert int(m_tr.sum() + m_va.sum() + m_oot.sum()) == int(included.sum())


def test_time_split_applies_embargo():
    df = _time_frame()
    m_tr0, m_va0, m_oot0, _ = compute_split_masks(
        df, _cfg(strategy="time", time_column="dt", embargo_days=0))
    m_tr3, m_va3, m_oot3, _ = compute_split_masks(
        df, _cfg(strategy="time", time_column="dt", embargo_days=3))
    assert (m_tr3 == m_tr0).all()  # embargo purges only valid/oot
    purged = int((m_va0.sum() + m_oot0.sum()) - (m_va3.sum() + m_oot3.sum()))
    assert purged > 0
    # purged rows belong to NO split
    in_any = m_tr3 | m_va3 | m_oot3
    assert int((~in_any).sum()) == purged


def test_time_split_with_exclusions_matches_manual_scatter():
    """compute_split_masks == split on surviving rows + scatter — the exact
    semantics Stage-2 build_dataset always had, now shared with Stage-1."""
    df = _time_frame()
    rules = [{"column": "flag", "values": [0]}]
    cfg = _cfg(strategy="time", time_column="dt", embargo_days=2,
               exclude_rows=rules)
    m_tr, m_va, m_oot, included = compute_split_masks(df, cfg)

    keep = exclude_mask(df, rules)
    sub = df[keep].reset_index(drop=True)
    exp = split_by_yyyymmdd(sub["dt"], [0.6, 0.2, 0.2], embargo_days=2)
    e_tr, e_va, e_oot = scatter_masks(keep, exp)
    assert (m_tr == e_tr).all()
    assert (m_va == e_va).all()
    assert (m_oot == e_oot).all()


def test_scatter_masks_identity_when_all_included():
    included = np.ones(10, dtype=bool)
    sub = (np.arange(10) < 5, np.arange(10) >= 5)
    a, b = scatter_masks(included, sub)
    assert (a == sub[0]).all() and (b == sub[1]).all()
