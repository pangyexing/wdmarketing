"""Single-pass scan must be numerically identical to the legacy per-signal
chunk iterations, and the .npy-cached correlation Pass-2 must match the
CSV-rereading path. Also covers the scan-cache lifecycle in run_stage1.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wdm.analysis.correlation import (
    compute_correlation_edges, compute_edges_from_cache)
from wdm.analysis.feature_scan import run_feature_scan
from wdm.analysis.iv_woe import compute_iv_table
from wdm.analysis.lift import compute_feature_lift_table
from wdm.analysis.missing_stats import compute_missing_stats
from wdm.analysis.psi import compute_psi_table_single_source
from wdm.io.chunked_reader import iter_column_chunks
from wdm.preprocess.missing import MissingSpec, get_spec

sys.path.insert(0, str(Path(__file__).resolve().parent / "stage1_golden"))
import dataset_gen  # noqa: E402

CHUNK = 3


def _synthetic_df(n=4000, seed=7):
    rng = np.random.RandomState(seed)
    data = {}
    data["pos_a"] = rng.lognormal(1.0, 0.6, n)
    data["pos_b"] = data["pos_a"] * 2.0 + rng.lognormal(0.0, 0.1, n)  # high corr
    data["counts"] = rng.poisson(2.0, n).astype(float)                # zeros → sentinel
    data["signed"] = rng.randn(n)                                     # negatives kept by spec
    data["neg_as_miss"] = rng.randn(n) * 3.0                          # negatives → NaN
    data["with_nan"] = np.where(rng.rand(n) < 0.3, np.nan,
                                rng.lognormal(0.5, 0.4, n))
    data["const"] = np.ones(n)
    for i in range(5):
        data["noise_{0}".format(i)] = rng.lognormal(0.0, 0.5, n)
    y = (0.8 * np.log1p(data["pos_a"]) + 0.4 * data["signed"]
         + rng.randn(n)) > 1.2
    data["y"] = y.astype(int)
    return pd.DataFrame(data)


def _spec_map():
    default = MissingSpec(sentinels=[0], treat_negative_as_missing=True,
                          treat_empty_as_missing=True,
                          fill_strategy="constant", fill_constant=-999.0)
    signed = MissingSpec(sentinels=[], treat_negative_as_missing=False,
                         treat_empty_as_missing=True,
                         fill_strategy="constant", fill_constant=-999.0)
    return {"__default__": default, "signed": signed}


def _cfg():
    return {
        "analysis": {"n_bins": 10, "binning": "equal_freq", "corr_cutoff": 0.5,
                     "iv_min": 0.0, "psi_cutoff": 0.25, "missing_rate_max": 0.95,
                     "psi_flag_thresholds": {"shift": 0.10, "broken": 0.25}},
        "training": {"top_k_pct": 0.1, "random_seed": 0},
        "io": {"column_chunk_size": CHUNK},
    }


@pytest.fixture(scope="module")
def setup():
    df = _synthetic_df()
    td = tempfile.mkdtemp()
    path = Path(td) / "data.csv"
    df.to_csv(path, index=False)
    feats = [c for c in df.columns if c != "y"]
    rng = np.random.RandomState(3)
    m_e = rng.rand(len(df)) < 0.5
    m_a = ~m_e
    return df, path, feats, _spec_map(), _cfg(), m_e, m_a


@pytest.fixture(scope="module")
def scan_result(setup, tmp_path_factory):
    df, path, feats, spec_map, cfg, m_e, m_a = setup
    cache = tmp_path_factory.mktemp("scan_cache")
    return run_feature_scan(path, feats, df["y"], m_e, m_a,
                            spec_map, get_spec, cfg, cache_dir=cache)


def test_iv_equivalent(setup, scan_result):
    df, path, feats, spec_map, cfg, _, _ = setup
    it = iter_column_chunks(path, feats, always=["y"], chunk_size=CHUNK)
    legacy_df, legacy_specs = compute_iv_table(it, spec_map, df["y"], feats,
                                               cfg, get_spec)
    assert_frame_equal(scan_result.iv_df, legacy_df, check_exact=True)
    assert set(scan_result.bin_specs) == set(legacy_specs)
    for f in legacy_specs:
        assert scan_result.bin_specs[f].to_dict() == legacy_specs[f].to_dict()


def test_missing_equivalent(setup, scan_result):
    _, path, feats, spec_map, cfg, _, _ = setup
    it = iter_column_chunks(path, feats, always=[], chunk_size=CHUNK)
    legacy_df = compute_missing_stats(it, spec_map, get_spec)
    assert_frame_equal(scan_result.miss_df, legacy_df, check_exact=True)


def test_lift_equivalent(setup, scan_result):
    df, path, feats, spec_map, cfg, _, _ = setup
    it = iter_column_chunks(path, feats, always=["y"], chunk_size=CHUNK)
    legacy_df = compute_feature_lift_table(it, spec_map, df["y"], cfg, get_spec)
    assert_frame_equal(scan_result.lift_df, legacy_df, check_exact=True)


def test_psi_equivalent(setup, scan_result):
    _, path, feats, spec_map, cfg, m_e, m_a = setup
    it = iter_column_chunks(path, feats, always=["y"], chunk_size=CHUNK)
    legacy_df = compute_psi_table_single_source(it, m_e, m_a, spec_map, cfg,
                                                get_spec)
    assert_frame_equal(scan_result.psi_df, legacy_df, check_exact=True)


@pytest.mark.parametrize("mmap", [True, False])
def test_correlation_cache_equivalent(setup, scan_result, mmap):
    _, path, feats, spec_map, cfg, _, _ = setup
    legacy = compute_correlation_edges(
        feats, str(path), always=["y"], spec_map=spec_map, get_spec_fn=get_spec,
        chunk_size=CHUNK, threshold=0.5, min_overlap_frac=0.10)
    cached = compute_edges_from_cache(
        feats, scan_result.blocks, scan_result.cache_dir,
        scan_result.col_count, scan_result.col_sum, scan_result.col_sum_sq,
        scan_result.n_rows, threshold=0.5, min_overlap_frac=0.10, mmap=mmap)
    assert len(cached) > 0
    assert_frame_equal(cached, legacy, check_exact=True)


def test_stale_manifest_rejected(setup, scan_result):
    _, _, feats, _, _, _, _ = setup
    wrong_blocks = [list(b) for b in scan_result.blocks]
    wrong_blocks[0] = list(reversed(wrong_blocks[0]))
    with pytest.raises(ValueError):
        compute_edges_from_cache(
            feats, wrong_blocks, scan_result.cache_dir,
            scan_result.col_count, scan_result.col_sum, scan_result.col_sum_sq,
            scan_result.n_rows, threshold=0.5)


# ---------- scan-cache lifecycle inside run_stage1 ----------

def _run_stage1_small(tmp_path, scan_cache_overrides=None):
    from wdm.pipeline.stage1 import run_stage1
    dataset_gen.prepare_repo(tmp_path, n_rows=1200)
    cfg = dataset_gen.build_cfg(tmp_path)
    if scan_cache_overrides is not None:
        cfg["io"]["scan_cache"] = scan_cache_overrides
    run_stage1(cfg)
    return Path(cfg["_repo_root"]) / "artifacts" / cfg["name"] / "analysis" / "scan_cache"


def test_cache_removed_after_normal_run(tmp_path):
    base = _run_stage1_small(tmp_path,
                             {"enabled": True, "keep": False, "mmap": True})
    assert not list(base.glob("scan_*"))


def test_cache_kept_when_requested(tmp_path):
    base = _run_stage1_small(tmp_path,
                             {"enabled": True, "keep": True, "mmap": True})
    kept = list(base.glob("scan_*"))
    assert len(kept) == 1
    assert (kept[0] / "manifest.json").is_file()
    assert list(kept[0].glob("block_*.npy"))


def test_cache_removed_on_failure(tmp_path, monkeypatch):
    import wdm.analysis.correlation as corr_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("injected pass-2 failure")

    monkeypatch.setattr(corr_mod, "compute_edges_from_cache", _boom)
    with pytest.raises(RuntimeError):
        _run_stage1_small(tmp_path,
                          {"enabled": True, "keep": False, "mmap": True})
    base = tmp_path / "artifacts" / dataset_gen.PRODUCT_NAME / "analysis" / "scan_cache"
    assert not list(base.glob("scan_*"))


def test_disabled_cache_wide_table_hard_guard(tmp_path):
    """With scan_cache off and more features than
    analysis.slow_correlation_max_features, Stage-1 must refuse to run the
    quadratic CSV-reparsing fallback unless explicitly allowed."""
    from wdm.pipeline.stage1 import run_stage1
    dataset_gen.prepare_repo(tmp_path, n_rows=1200)
    cfg = dataset_gen.build_cfg(tmp_path)
    cfg["io"]["scan_cache"] = {"enabled": False}
    cfg["analysis"]["slow_correlation_max_features"] = 3
    with pytest.raises(RuntimeError):
        run_stage1(cfg)
    # explicit opt-in unblocks the fallback
    cfg["analysis"]["allow_slow_correlation"] = True
    run_stage1(cfg)


def test_disabled_cache_falls_back_to_csv_path(tmp_path):
    base = _run_stage1_small(tmp_path, {"enabled": False})
    assert not base.exists() or not list(base.glob("scan_*"))
    summary = (tmp_path / "artifacts" / dataset_gen.PRODUCT_NAME /
               "analysis" / "report" / "summary.csv")
    assert summary.is_file()