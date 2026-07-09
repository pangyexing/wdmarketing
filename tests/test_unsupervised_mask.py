"""analysis.unsupervised_stats_split — masked scan over the full CSV must be
numerically identical to an unmasked scan over the masked-subset CSV, for
every statistic that feeds selection (missing rate, correlation Pass-1 stats,
cached blocks / Pass-2 edges, and PSI when its partition lies inside the
mask). Guards the train-only leak fix."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wdm.analysis.correlation import (
    compute_correlation_edges, compute_edges_from_cache)
from wdm.analysis.feature_scan import run_feature_scan
from wdm.preprocess.missing import MissingSpec, get_spec

CHUNK = 3


def _df(n=2000, seed=11):
    rng = np.random.RandomState(seed)
    data = {
        "a": rng.lognormal(1.0, 0.6, n),
        "c": rng.poisson(2.0, n).astype(float),
        "d": np.where(rng.rand(n) < 0.25, np.nan, rng.lognormal(0.5, 0.4, n)),
    }
    data["b"] = data["a"] * 2.0 + rng.lognormal(0.0, 0.1, n)
    for i in range(4):
        data["noise_{0}".format(i)] = rng.lognormal(0.0, 0.5, n)
    data["y"] = ((0.8 * np.log1p(data["a"]) + rng.randn(n)) > 1.0).astype(int)
    return pd.DataFrame(data)


def _spec_map():
    return {"__default__": MissingSpec(
        sentinels=[0], treat_negative_as_missing=True,
        treat_empty_as_missing=True,
        fill_strategy="constant", fill_constant=-999.0)}


def _cfg():
    return {
        "analysis": {"n_bins": 10, "binning": "equal_freq", "corr_cutoff": 0.5,
                     "psi_flag_thresholds": {"shift": 0.10, "broken": 0.25}},
        "training": {"top_k_pct": 0.1, "random_seed": 0},
        "io": {"column_chunk_size": CHUNK},
    }


@pytest.fixture(scope="module")
def data():
    df = _df()
    rng = np.random.RandomState(5)
    m = rng.rand(len(df)) < 0.6              # "train" rows
    # PSI partition entirely inside the mask so subset/masked runs agree.
    r = rng.rand(len(df))
    m_e = m & (r < 0.5)
    m_a = m & (r >= 0.5)

    td = Path(tempfile.mkdtemp())
    full_csv = td / "full.csv"
    sub_csv = td / "sub.csv"
    df.to_csv(full_csv, index=False)
    df[m].reset_index(drop=True).to_csv(sub_csv, index=False)
    feats = [c for c in df.columns if c != "y"]
    return df, m, m_e, m_a, full_csv, sub_csv, feats


@pytest.fixture(scope="module")
def masked_vs_subset(data, tmp_path_factory):
    df, m, m_e, m_a, full_csv, sub_csv, feats = data
    cache_masked = tmp_path_factory.mktemp("cache_masked")
    cache_sub = tmp_path_factory.mktemp("cache_sub")
    masked = run_feature_scan(full_csv, feats, df["y"], m_e, m_a,
                              _spec_map(), get_spec, _cfg(),
                              cache_dir=cache_masked,
                              supervised_mask=m, unsupervised_mask=m)
    sub_df = df[m].reset_index(drop=True)
    subset = run_feature_scan(sub_csv, feats, sub_df["y"], m_e[m], m_a[m],
                              _spec_map(), get_spec, _cfg(),
                              cache_dir=cache_sub)
    return masked, subset


def test_missing_stats_match_subset(masked_vs_subset):
    masked, subset = masked_vs_subset
    assert_frame_equal(masked.miss_df, subset.miss_df, check_exact=True)


def test_supervised_stats_match_subset(masked_vs_subset):
    masked, subset = masked_vs_subset
    assert_frame_equal(masked.iv_df, subset.iv_df, check_exact=True)
    assert_frame_equal(masked.lift_df, subset.lift_df, check_exact=True)


def test_psi_matches_subset(masked_vs_subset):
    masked, subset = masked_vs_subset
    assert_frame_equal(masked.psi_df, subset.psi_df, check_exact=True)


def test_correlation_stats_and_edges_match_subset(masked_vs_subset, data):
    masked, subset = masked_vs_subset
    _, _, _, _, _, _, feats = data
    assert masked.n_rows == subset.n_rows
    assert (masked.col_count == subset.col_count).all()
    np.testing.assert_array_equal(masked.col_sum, subset.col_sum)
    np.testing.assert_array_equal(masked.col_sum_sq, subset.col_sum_sq)
    e_masked = compute_edges_from_cache(
        feats, masked.blocks, masked.cache_dir, masked.col_count,
        masked.col_sum, masked.col_sum_sq, masked.n_rows, threshold=0.5)
    e_subset = compute_edges_from_cache(
        feats, subset.blocks, subset.cache_dir, subset.col_count,
        subset.col_sum, subset.col_sum_sq, subset.n_rows, threshold=0.5)
    assert len(e_masked) > 0
    assert_frame_equal(e_masked, e_subset, check_exact=True)


def test_csv_fallback_row_mask_matches_subset(data):
    _, m, _, _, full_csv, sub_csv, feats = data
    masked = compute_correlation_edges(
        feats, str(full_csv), always=["y"], spec_map=_spec_map(),
        get_spec_fn=get_spec, chunk_size=CHUNK, threshold=0.5, row_mask=m)
    subset = compute_correlation_edges(
        feats, str(sub_csv), always=["y"], spec_map=_spec_map(),
        get_spec_fn=get_spec, chunk_size=CHUNK, threshold=0.5)
    assert len(masked) > 0
    assert_frame_equal(masked, subset, check_exact=True)


def test_full_mask_is_noop(data, tmp_path_factory):
    df, _, m_e, m_a, full_csv, _, feats = data
    all_rows = np.ones(len(df), dtype=bool)
    plain = run_feature_scan(full_csv, feats, df["y"], m_e, m_a,
                             _spec_map(), get_spec, _cfg())
    noop = run_feature_scan(full_csv, feats, df["y"], m_e, m_a,
                            _spec_map(), get_spec, _cfg(),
                            unsupervised_mask=all_rows)
    assert_frame_equal(plain.miss_df, noop.miss_df, check_exact=True)
    np.testing.assert_array_equal(plain.col_sum, noop.col_sum)
