"""Deterministic synthetic dataset + config for Stage-1 golden snapshot tests.

The dataset is engineered to exercise every Stage-1 code path:
  * time-window families (amt_*/cnt_* shared-latent, txn_* near-duplicate)
  * a prefix-declared semantic group (bureau_*)
  * plain high-correlation pairs (dup_a/dup_b/dup_c) for union-find clustering
  * hard-filter triggers: constant column, >95% missing column, low-IV noise
  * a short-window feature (rare_7d) whose missing rate sits between the
    global 0.95 cap and the short-window 0.98 soft cap
  * PSI drift: drift_feat (broken, >=0.25) and mild_drift (shift, 0.10-0.25)
  * sentinel zeros (counts) and a per-column spec override (signed_ok keeps
    negative values)

Everything is seeded — same numpy/pandas versions produce byte-identical
Stage-1 artifacts, which is what tests/test_stage1_golden.py asserts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_SEED = 20260611
N_ROWS = 20000
PRODUCT_NAME = "golden_stage1"
DATA_REL_PATH = "data/golden_stage1.csv"


def _z(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()


def make_dataframe(n_rows=N_ROWS, seed=DATASET_SEED):
    rng = np.random.RandomState(seed)

    # Time column: 60 consecutive days as yyyymmdd ints, random row order.
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    day_ints = np.array([int(d.strftime("%Y%m%d")) for d in days], dtype=np.int64)
    day_idx = rng.randint(0, len(days), n_rows)
    dt = day_ints[day_idx]
    time_frac = day_idx / float(len(days) - 1)   # 0 .. 1 across the period

    cols = {}
    cols["cust_id"] = np.arange(1, n_rows + 1, dtype=np.int64)
    cols["dt"] = dt

    # --- amt_* family: shared lognormal latent, moderate within-family corr ---
    l_amt = rng.lognormal(3.0, 0.6, n_rows)
    cols["amt_7d"] = l_amt * 0.2 + rng.lognormal(1.0, 0.5, n_rows)
    cols["amt_30d"] = l_amt * 0.6 + rng.lognormal(1.0, 0.5, n_rows)
    cols["amt_90d"] = l_amt * 1.0 + rng.lognormal(1.0, 0.5, n_rows)
    cols["amt_all"] = l_amt * 1.4 + rng.lognormal(1.0, 0.5, n_rows)

    # --- cnt_* family: poisson counts with shared rate; zeros hit the 0-sentinel ---
    lam = rng.gamma(2.0, 1.5, n_rows)
    cols["cnt_7d"] = rng.poisson(lam * 0.3).astype(np.float64)
    cols["cnt_30d"] = rng.poisson(lam * 1.0).astype(np.float64)
    cols["cnt_90d"] = rng.poisson(lam * 2.5).astype(np.float64)
    cols["cnt_all"] = rng.poisson(lam * 4.0).astype(np.float64)

    # --- txn_* family: near-duplicates within family (|r| > 0.95) ---
    l_txn = rng.lognormal(2.0, 0.7, n_rows)
    cols["txn_7d"] = l_txn + rng.lognormal(0.0, 0.05, n_rows) * 0.1
    cols["txn_30d"] = l_txn * 3.0 + rng.lognormal(0.0, 0.05, n_rows) * 0.2
    cols["txn_90d"] = l_txn * 6.0 + rng.lognormal(1.0, 0.6, n_rows)
    cols["txn_180d"] = l_txn * 9.0 + rng.lognormal(1.5, 0.7, n_rows)
    cols["txn_360d"] = l_txn * 12.0 + rng.lognormal(2.0, 0.7, n_rows)
    cols["txn_all"] = l_txn * 15.0 + rng.lognormal(2.0, 0.8, n_rows)

    # --- bureau_* semantic group (declared by prefix in the config) ---
    l_bureau = rng.gamma(3.0, 1.2, n_rows)
    cols["bureau_org_cnt"] = rng.poisson(l_bureau).astype(np.float64) + 1.0
    cols["bureau_query_cnt"] = cols["bureau_org_cnt"] * 2.0 + rng.poisson(1.0, n_rows)
    cols["bureau_overdue_cnt"] = rng.poisson(l_bureau * 0.3).astype(np.float64)

    # --- plain near-duplicates for global union-find clustering ---
    dup_base = rng.lognormal(2.5, 0.8, n_rows)
    cols["dup_a"] = dup_base
    cols["dup_b"] = dup_base + rng.lognormal(0.0, 0.03, n_rows) * 0.05
    cols["dup_c"] = dup_base * 1.5 + rng.lognormal(0.0, 0.4, n_rows) * 0.5

    # --- hard-filter triggers ---
    cols["const_one"] = np.ones(n_rows, dtype=np.float64)
    mm = rng.lognormal(2.0, 0.5, n_rows)
    mm[rng.rand(n_rows) < 0.97] = np.nan
    cols["mostly_missing"] = mm
    rare = rng.lognormal(2.0, 0.5, n_rows)
    rare[rng.rand(n_rows) < 0.965] = np.nan   # between 0.95 cap and 0.98 short-window cap
    cols["rare_7d"] = rare

    # --- per-column spec override target: negatives are legitimate here ---
    cols["signed_ok"] = rng.randn(n_rows) * 2.0 + 0.5

    # --- PSI drift features ---
    cols["drift_feat"] = rng.lognormal(2.0, 0.4, n_rows) + time_frac * 25.0
    cols["mild_drift"] = rng.lognormal(2.0, 0.4, n_rows) * (1.0 + 0.35 * time_frac)

    # --- pure noise (low IV) ---
    for i in range(20):
        cols["noise_{0:02d}".format(i)] = rng.lognormal(1.0, 0.6, n_rows)

    # --- label: driven by a few latents, base rate ~0.15 ---
    logit = (-2.2
             + 0.9 * _z(l_amt)
             + 0.7 * _z(l_bureau)
             + 0.5 * _z(np.log1p(dup_base))
             + 0.4 * _z(lam))
    p = 1.0 / (1.0 + np.exp(-logit))
    cols["y"] = (rng.rand(n_rows) < p).astype(np.int64)

    return pd.DataFrame(cols)


def prepare_repo(repo_root, n_rows=N_ROWS):
    """Write the synthetic CSV under <repo_root>/data/. Returns the CSV path."""
    repo_root = Path(repo_root)
    out_path = repo_root / DATA_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataframe(n_rows=n_rows)
    df.to_csv(out_path, index=False)
    return out_path


def build_cfg(repo_root):
    """A fully-resolved config dict mimicking load_config() output."""
    repo_root = Path(repo_root)
    return {
        "name": PRODUCT_NAME,
        "_repo_root": str(repo_root),
        "_configs_dir": str(repo_root / "configs"),
        "data": {
            "train_path": DATA_REL_PATH,
            "label_column": "y",
            "time_column": "dt",
            "id_columns": ["cust_id"],
            "column_mapping": None,
        },
        "missing": {
            "global": {
                "sentinels": [0],
                "treat_negative_as_missing": True,
                "treat_empty_as_missing": True,
                "fill_strategy": "constant",
                "fill_constant": -999,
                "analysis_use_mask": True,
            },
            "per_column": {
                "signed_ok": {"sentinels": [], "treat_negative_as_missing": False},
            },
        },
        "analysis": {
            "missing_rate_max": 0.95,
            "psi_cutoff": 0.25,
            "iv_min": 0.02,
            "corr_cutoff": 0.95,
            "n_bins": 10,
            "binning": "equal_freq",
            "per_feature_plot_top_n": 50,
            # required keys (defaults live only in configs/global.yaml);
            # values match the Batch-1 defaults the snapshot was built under.
            "psi_partition": "train_halves",
            "supervised_stats_split": "train_only",
            "unsupervised_stats_split": "train_only",
            "lift_keep_min": 1.2,
            "rank_weights": {
                "iv": 1.0,
                "lift": 1.0,
                "gini": 1.0,
                "concentration": 0.0,
                "psi": 1.0,
                "missing_penalty": 0.5,
                "missing_penalty_threshold": 0.5,
            },
        },
        "training": {
            "top_k_pct": 0.10,
            "final_feature_count": 25,
            "random_seed": 42,
            # explicit split (was previously the silent in-code default)
            "split": {"strategy": "stratified",
                      "ratios": [0.70, 0.15, 0.15], "embargo_days": 0},
        },
        "io": {
            "column_chunk_size": 7,
        },
        "feature_groups": {
            "window_pattern": r"^(?P<base>.+?)_(?P<window>7d|30d|90d|180d|360d|all|life|hist)$",
            "window_order": ["7d", "30d", "90d", "180d", "360d", "all", "life", "hist"],
            "family_policy": {
                "max_per_family": 2,
                "prefer": "best_iv",
                "corr_cutoff_in_family": 0.90,
            },
            "enable_window_family": True,
            "semantic_groups": [
                {
                    "name": "bureau",
                    "description": "credit bureau counters",
                    "feature_prefix": "bureau_",
                    "prefer": "best_iv",
                    "max_keep": 2,
                    "corr_cutoff_in_group": 0.85,
                },
            ],
        },
        "selected_features": {
            "active_version": "v1_auto",
        },
    }
