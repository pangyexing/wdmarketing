"""The load-bearing invariant: column-chunked PSI/IV/correlation must match
whole-table values to floating-point precision. If this test fails, all
Stage-1 numbers are suspect.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from wdm.analysis.correlation import compute_correlation_edges, _pairwise_cov_block
from wdm.analysis.iv_woe import compute_iv_table
from wdm.analysis.psi import compute_psi, compute_psi_table_single_source
from wdm.io.chunked_reader import iter_column_chunks
from wdm.preprocess.missing import MissingSpec, get_spec, to_nan_array


def _make_temp_csv(df):
    td = tempfile.mkdtemp()
    path = Path(td) / "data.csv"
    df.to_csv(path, index=False)
    return path


def _synthetic_df(n=10000, n_feat=10, seed=0):
    rng = np.random.RandomState(seed)
    data = {}
    for i in range(n_feat):
        x = rng.randn(n) + rng.rand() * 2
        # Sprinkle some NaN-equivalents: negatives per column spec
        data["f{0}".format(i)] = x
    y = (0.3 * data["f0"] - 0.2 * data["f3"] + rng.randn(n)) > 0.5
    data["y"] = y.astype(int)
    return pd.DataFrame(data)


def _cfg():
    return {
        "analysis": {"n_bins": 10, "binning": "equal_freq", "corr_cutoff": 0.0,
                     "iv_min": 0.0, "psi_cutoff": 0.25, "missing_rate_max": 0.95},
        "training": {"top_k_pct": 0.1, "random_seed": 0},
        "missing": {"global": {"sentinels": [], "treat_negative_as_missing": False,
                               "treat_empty_as_missing": True,
                               "fill_strategy": "constant", "fill_constant": -999.0}},
        "feature_groups": {"window_pattern": "",
                           "window_order": [],
                           "family_policy": {}},
        "io": {"column_chunk_size": 3},
    }


def _specs_all_default(features):
    s = {
        "__default__": MissingSpec(sentinels=[], treat_negative_as_missing=False,
                                   treat_empty_as_missing=True,
                                   fill_strategy="constant", fill_constant=-999.0)
    }
    return s


def test_iv_chunked_equals_whole_table():
    df = _synthetic_df()
    path = _make_temp_csv(df)
    feats = [c for c in df.columns if c != "y"]
    spec_map = _specs_all_default(feats)
    cfg = _cfg()

    # Chunked
    it = iter_column_chunks(path, feats, always=["y"], chunk_size=3)
    iv_chunked, _ = compute_iv_table(it, spec_map, df["y"], feats, cfg, get_spec)

    # Whole-table (single chunk of all columns)
    it2 = iter_column_chunks(path, feats, always=["y"], chunk_size=len(feats))
    iv_whole, _ = compute_iv_table(it2, spec_map, df["y"], feats, cfg, get_spec)

    iv_chunked = iv_chunked.sort_values("feature").set_index("feature")
    iv_whole = iv_whole.sort_values("feature").set_index("feature")
    diff = (iv_chunked["iv"] - iv_whole["iv"]).abs().max()
    assert diff < 1e-10, "Chunked vs whole-table IV mismatch: {0}".format(diff)


def test_psi_chunked_equals_whole_table():
    df = _synthetic_df()
    path = _make_temp_csv(df)
    feats = [c for c in df.columns if c != "y"]
    spec_map = _specs_all_default(feats)
    cfg = _cfg()

    rng = np.random.RandomState(0)
    mask_a = rng.rand(len(df)) < 0.5
    mask_b = ~mask_a

    it = iter_column_chunks(path, feats, always=["y"], chunk_size=3)
    psi_chunked = compute_psi_table_single_source(
        it, mask_a, mask_b, spec_map, cfg, get_spec)

    it2 = iter_column_chunks(path, feats, always=["y"], chunk_size=len(feats))
    psi_whole = compute_psi_table_single_source(
        it2, mask_a, mask_b, spec_map, cfg, get_spec)

    a = psi_chunked.sort_values("feature").set_index("feature")["psi"]
    b = psi_whole.sort_values("feature").set_index("feature")["psi"]
    diff = (a - b).abs().max()
    assert diff < 1e-10, "Chunked vs whole-table PSI mismatch: {0}".format(diff)


def test_correlation_chunked_equals_pandas_corr():
    df = _synthetic_df()
    path = _make_temp_csv(df)
    feats = [c for c in df.columns if c != "y"]
    spec_map = _specs_all_default(feats)

    edges = compute_correlation_edges(
        feats, str(path), always=["y"],
        spec_map=spec_map, get_spec_fn=get_spec,
        chunk_size=3, threshold=0.0)

    # Ground truth via pandas (no NaN handling needed since default spec keeps negatives)
    truth = df[feats].corr()

    for _, row in edges.iterrows():
        expected = float(truth.loc[row["f1"], row["f2"]])
        assert abs(row["r"] - expected) < 1e-10, (
            "corr({0},{1}) mismatch: got {2} vs pandas {3}".format(
                row["f1"], row["f2"], row["r"], expected))
