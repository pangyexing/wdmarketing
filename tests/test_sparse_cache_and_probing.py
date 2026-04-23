"""Tests for the sparse cache builder and Stage-1 probing module.

These exercise the two new pieces end-to-end on a small synthetic CSV:
  1. scripts/build_sparse_cache.py: 0/NaN/real-value encoding in CSR,
     manifest round-trip, staleness detection.
  2. wdm/analysis/probing.py: missing-value resolution from sentinels,
     importance frame shape, deterministic ranking under fixed seed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Make scripts/ importable (build_sparse_cache lives there, not under src/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_sparse_cache import build_sparse_cache, load_cache  # noqa: E402
from wdm.analysis.probing import (  # noqa: E402
    _feature_coverage, _resolve_missing_value, _rank_pct, run_probing,
)


def _make_cfg(tmp_path, csv_path, product="toy", sentinels=None,
              split_strategy="stratified"):
    """Minimal cfg dict with just the keys the cache builder + probing use."""
    return {
        "_repo_root": str(tmp_path),
        "name": product,
        "data": {
            "train_path": str(csv_path.relative_to(tmp_path)),
            "label_column": "y",
            "time_column": "yyyymmdd" if split_strategy == "time" else None,
            "id_columns": ["sid"],
            "treatment_column": None,
        },
        "missing": {"global": {"sentinels": list(sentinels or [])}},
        "training": {
            "random_seed": 42,
            "split": {"strategy": split_strategy,
                      "ratios": [0.6, 0.2, 0.2]},
        },
        "analysis": {"probing": {}},
    }


def _write_toy_csv(tmp_path, n_rows=300, n_feats=8, seed=0):
    """Synthetic table: some features predictive, sparsity mixed with legit 0s."""
    rng = np.random.RandomState(seed)
    cols = {"sid": np.arange(n_rows)}
    # Time column spans 20230101..20241231 so time-split has enough variety.
    start = 20230101
    cols["yyyymmdd"] = np.array(
        [start + (i // 2) for i in range(n_rows)], dtype=np.int64)
    X = rng.randn(n_rows, n_feats).astype(np.float32)
    # Deliberately inject 0s (legitimate) and NaNs (missing).
    X[rng.rand(n_rows, n_feats) < 0.30] = 0.0
    nan_mask = rng.rand(n_rows, n_feats) < 0.20
    X[nan_mask] = np.nan
    for j in range(n_feats):
        cols["f{0}".format(j)] = X[:, j]
    # y depends on f0 and f1 (interaction); feature Gini will catch it too,
    # but we just want the probing model to train non-trivially.
    lin = np.nan_to_num(X[:, 0], nan=0.0) + 0.7 * np.nan_to_num(X[:, 1], nan=0.0)
    prob = 1.0 / (1.0 + np.exp(-lin))
    cols["y"] = (rng.rand(n_rows) < prob).astype(np.int32)
    df = pd.DataFrame(cols)
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# cache builder
# -----------------------------------------------------------------------------

def test_cache_preserves_zero_as_implicit_and_nan_as_explicit(tmp_path):
    """0 → structural zero (absent from CSR.data); NaN → explicit in CSR.data."""
    # Deterministic hand-crafted block so we can count entries exactly.
    df = pd.DataFrame({
        "sid": [0, 1, 2, 3],
        "yyyymmdd": [20230101, 20230201, 20230301, 20230401],
        "f0": [1.5, 0.0,    np.nan, 2.5],
        "f1": [0.0, np.nan, 3.0,    0.0],
        "y":  [0, 1, 0, 1],
    })
    csv = tmp_path / "hand.csv"
    df.to_csv(csv, index=False)
    cfg = _make_cfg(tmp_path, csv, split_strategy="time")

    out_dir = tmp_path / "cache"
    meta = build_sparse_cache(csv, out_dir, cfg, chunk_rows=2)

    X = sp.load_npz(out_dir / "X.csr.npz")
    assert X.shape == (4, 2)
    # f0=[1.5,0,NaN,2.5] keeps 3 entries; f1=[0,NaN,3,0] keeps 2 entries → 5 nnz.
    # The four 0s stay implicit (structural zero) and save memory.
    assert X.nnz == 5
    data = X.data
    n_nan = int(np.isnan(data).sum())
    assert n_nan == 2, "expected 2 explicit NaN entries in CSR data"
    assert np.sum(~np.isnan(data)) == 3

    # Manifest captures correct shape + density.
    assert meta["n_rows"] == 4 and meta["n_features"] == 2
    assert meta["nnz"] == 5
    assert 0.0 < meta["density"] <= 1.0


def test_load_cache_detects_stale_csv(tmp_path):
    csv = _write_toy_csv(tmp_path)
    cfg = _make_cfg(tmp_path, csv)
    out_dir = tmp_path / "cache"
    build_sparse_cache(csv, out_dir, cfg, chunk_rows=100)

    # Clean load works.
    cache = load_cache(out_dir, csv_path=csv)
    assert cache["X"].shape[0] == 300
    assert "sid" in cache

    # Touch CSV to change mtime → load_cache must raise.
    import os
    import time
    time.sleep(0.02)
    os.utime(csv, None)
    with pytest.raises(RuntimeError, match="Cache stale"):
        load_cache(out_dir, csv_path=csv)


def test_manifest_roundtrip(tmp_path):
    csv = _write_toy_csv(tmp_path)
    cfg = _make_cfg(tmp_path, csv)
    out_dir = tmp_path / "cache"
    meta = build_sparse_cache(csv, out_dir, cfg, chunk_rows=75)

    disk_meta = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert disk_meta["n_rows"] == meta["n_rows"]
    assert disk_meta["n_features"] == meta["n_features"]
    assert disk_meta["nnz"] == meta["nnz"]
    assert disk_meta["label_column"] == "y"
    assert disk_meta["id_columns"] == ["sid"]
    assert "sparse_convention" in disk_meta


# -----------------------------------------------------------------------------
# probing
# -----------------------------------------------------------------------------

def test_resolve_missing_value_default_treats_zero_as_missing():
    """New default (2026-04-24): probing collapses 0 and NaN into missing
    regardless of missing.global.sentinels. Sparse aggregated tables (HC)
    want this — 0 count and NaN record carry the same "no contribution"
    meaning to a tree-split ranking tool.
    """
    # HC-style config: no sentinels, but probing still uses missing=0.0
    hc_cfg = {"missing": {"global": {"sentinels": []}}}
    v, _ = _resolve_missing_value(hc_cfg)
    assert v == 0.0

    # bank_marketing-style config: 0 in sentinels, same result.
    bm_cfg = {"missing": {"global": {"sentinels": [0]}}}
    v2, _ = _resolve_missing_value(bm_cfg)
    assert v2 == 0.0


def test_resolve_missing_value_opt_out_falls_back_to_sentinels():
    """When analysis.probing.treat_zero_as_missing=false, the old behavior
    returns: follow sentinels exactly.
    """
    hc_cfg = {
        "missing": {"global": {"sentinels": []}},
        "analysis": {"probing": {"treat_zero_as_missing": False}},
    }
    v, _ = _resolve_missing_value(hc_cfg)
    assert np.isnan(v)

    bm_cfg = {
        "missing": {"global": {"sentinels": [0]}},
        "analysis": {"probing": {"treat_zero_as_missing": False}},
    }
    v2, _ = _resolve_missing_value(bm_cfg)
    assert v2 == 0.0


def test_feature_coverage_matches_missing_semantic():
    """Coverage must follow the DMatrix `missing` convention exactly:
    with missing=0.0 structural zeros are missing, with missing=NaN they aren't.
    """
    # 4 rows × 3 features:
    #   f0: [1.5, 0,    NaN, 2.5]  → 2 explicit real, 1 NaN, 1 implicit 0
    #   f1: [0,   NaN,  3.0, 0  ]  → 1 explicit real, 1 NaN, 2 implicit 0
    #   f2: [0,   0,    0,   0  ]  → all implicit zeros
    df = pd.DataFrame({
        "f0": [1.5, 0.0,    np.nan, 2.5],
        "f1": [0.0, np.nan, 3.0,    0.0],
        "f2": [0.0, 0.0,    0.0,    0.0],
    })
    # Build CSR directly mirroring build_sparse_cache's convention:
    # zeros stay implicit, NaN is explicit.
    rows, cols, vals = [], [], []
    for j, c in enumerate(df.columns):
        for i, v in enumerate(df[c].values):
            if pd.isna(v):
                rows.append(i); cols.append(j); vals.append(np.nan)
            elif v != 0:
                rows.append(i); cols.append(j); vals.append(float(v))
    X = sp.csr_matrix((vals, (rows, cols)), shape=(4, 3))

    # missing=0.0 → only explicit non-NaN entries count as observed.
    cov_zero = _feature_coverage(X, 0.0, n_features=3)
    # f0: 2 real / 4 = 0.5; f1: 1/4 = 0.25; f2: 0/4 = 0.
    np.testing.assert_allclose(cov_zero, [0.5, 0.25, 0.0])

    # missing=NaN → implicit zeros are observed; only explicit NaN is missing.
    cov_nan = _feature_coverage(X, np.nan, n_features=3)
    # f0: 4 - 1 NaN = 3/4 = 0.75; f1: 4 - 1 = 0.75; f2: 4 - 0 = 1.0.
    np.testing.assert_allclose(cov_nan, [0.75, 0.75, 1.0])


def test_feature_coverage_handles_empty_matrix():
    X = sp.csr_matrix((0, 5))
    cov = _feature_coverage(X, 0.0, n_features=5)
    assert cov.shape == (5,)
    assert (cov == 0).all()


def test_rank_pct_is_bounded_and_monotonic():
    s = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    r = _rank_pct(s)
    assert (r >= 0).all() and (r <= 1).all()
    # Non-NaN inputs strictly increase in rank.
    assert r.iloc[0] < r.iloc[1] < r.iloc[2] < r.iloc[3]
    # NaN → 0 by contract.
    assert r.iloc[4] == 0.0


def test_probing_runs_end_to_end_and_is_deterministic(tmp_path):
    """Smoke: build cache, run probing twice, check importance is reproducible."""
    pytest.importorskip("xgboost")
    csv = _write_toy_csv(tmp_path, n_rows=400, seed=7)
    cfg = _make_cfg(tmp_path, csv)
    # Keep run tiny so the test stays fast.
    cfg["analysis"]["probing"] = {
        "num_boost_round": 30,
        "early_stopping_rounds": 10,
        "xgb_params": {"max_depth": 3, "eta": 0.2, "seed": 123,
                       "tree_method": "hist", "objective": "binary:logistic",
                       "eval_metric": "aucpr", "verbosity": 0},
    }
    out_dir = tmp_path / "cache"
    build_sparse_cache(csv, out_dir, cfg, chunk_rows=200)

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    r1 = run_probing(cfg, cache_dir=out_dir, out_dir=report_dir)
    imp1 = pd.read_csv(r1["importance_path"])
    assert set(["feature", "gain", "weight", "cover",
                "gain_rank_pct", "coverage"]).issubset(imp1.columns)
    assert (imp1["gain"] >= 0).all()
    # Coverage is a ratio in [0,1] on the rows the model fit.
    assert (imp1["coverage"] >= 0).all() and (imp1["coverage"] <= 1).all()

    # Fresh output dir: run again, compare rankings.
    report_dir2 = tmp_path / "report2"
    report_dir2.mkdir()
    r2 = run_probing(cfg, cache_dir=out_dir, out_dir=report_dir2)
    imp2 = pd.read_csv(r2["importance_path"])
    # Deterministic: feature order (by gain desc) should match.
    assert list(imp1["feature"]) == list(imp2["feature"])
    # gain values match to fp tolerance.
    np.testing.assert_allclose(
        imp1["gain"].values, imp2["gain"].values, rtol=1e-6, atol=1e-8)


def test_probing_opt_out_produces_different_missing_semantics(tmp_path):
    """End-to-end: when a product opts out of the default (treat_zero_as_missing=false),
    probing_meta.json's missing_value differs — and with sentinels=[] it becomes NaN
    instead of the default 0.0.
    """
    pytest.importorskip("xgboost")
    csv = _write_toy_csv(tmp_path, n_rows=300, seed=3)

    # Config A: new default — treat_zero_as_missing=true (implicit)
    cfg_default = _make_cfg(tmp_path, csv, sentinels=[])
    cfg_default["analysis"]["probing"] = {
        "num_boost_round": 20, "early_stopping_rounds": 10,
        "xgb_params": {"max_depth": 3, "eta": 0.3, "seed": 1,
                       "tree_method": "hist", "objective": "binary:logistic",
                       "eval_metric": "aucpr", "verbosity": 0},
    }

    # Config B: explicit opt-out — follow sentinels (here empty → NaN)
    cfg_optout = _make_cfg(tmp_path, csv, sentinels=[])
    cfg_optout["analysis"]["probing"] = dict(cfg_default["analysis"]["probing"],
                                             treat_zero_as_missing=False)

    out_dir = tmp_path / "cache"
    build_sparse_cache(csv, out_dir, cfg_default, chunk_rows=200)

    rdir_default = tmp_path / "rep_default"; rdir_default.mkdir()
    rdir_optout = tmp_path / "rep_optout"; rdir_optout.mkdir()
    run_probing(cfg_default, out_dir, rdir_default)
    run_probing(cfg_optout, out_dir, rdir_optout)

    meta_default = json.loads((rdir_default / "probing_meta.json").read_text(encoding="utf-8"))
    meta_optout = json.loads((rdir_optout / "probing_meta.json").read_text(encoding="utf-8"))
    assert meta_default["missing_value"] == 0.0
    assert meta_optout["missing_value"] == "NaN"
