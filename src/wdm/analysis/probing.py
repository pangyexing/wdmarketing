"""Stage-1 probing model: a cheap, deterministic XGBoost trained on the full
raw feature pool to surface interaction-level signal that univariate IV /
PSI / feature-Gini cannot see.

Design choices:
- **Orthogonal to statistical Stage 1**: output is additive to summary.csv's
  rank_score, never replaces PSI/IV/Lift/correlation.
- **Missing semantic comes from config**: cfg.missing.global.sentinels.
  If 0 ∈ sentinels → DMatrix missing=0.0 (0 and NaN both routed to missing
  branch). Else → missing=np.nan (0 is a legitimate value, only NaN is missing).
- **Split respects Stage 2 discipline**: train only on train fold of
  cfg.training.split (time or stratified). Validation fold is used for
  early stopping. OOT is never touched — otherwise probing importance
  would leak OOT signal into feature selection.
- **Fixed, cheap hyperparameters**: probing is a signal-generator, not a
  model to deploy. Default: max_depth=6, eta=0.1, subsample=0.8,
  colsample_bytree=0.8, seed=42, num_boost_round=300, early_stopping=30.
- **Deterministic**: fixed seed + single-threaded tie-break → reproducible
  rank_score contribution across runs.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Defaults applied when cfg.analysis.probing.xgb_params is sparse/missing.
# colsample_bynode (not bytree): at each split node XGBoost re-samples 80% of
# columns. For high-dim sparse inputs this is preferable to per-tree sampling —
# a sparse feature with real signal has many more chances to be considered,
# rather than being absent from an entire tree because that tree's bytree
# sample missed it.
_DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bynode": 0.8,
    "min_child_weight": 1.0,
    "lambda": 1.0,
    "verbosity": 0,
}

_DEFAULT_NUM_BOOST_ROUND = 300
_DEFAULT_EARLY_STOPPING = 30
_DEFAULT_SEED = 42


def _resolve_missing_value(cfg):
    """Choose DMatrix `missing` for the probing model.

    Resolution order:
    1. cfg.analysis.probing.treat_zero_as_missing (default True):
       True  → missing=0.0  (both implicit 0 and NaN go to missing branch;
                             collapses "0 count" and "NaN no record" — sane
                             default for sparse aggregated tables).
       False → fall back to cfg.missing.global.sentinels:
         - 0 in sentinels → missing=0.0
         - else            → missing=np.nan (0 is a real value; only NaN is missing).

    Note: this only affects the probing ranking tool. Stage 2 training /
    deployment still respect cfg.missing.global.* untouched.
    """
    probing_cfg = (cfg.get("analysis") or {}).get("probing") or {}
    treat_zero_as_missing = bool(probing_cfg.get("treat_zero_as_missing", True))
    if treat_zero_as_missing:
        return 0.0, ("analysis.probing.treat_zero_as_missing=true → 0 and NaN "
                     "both treated as missing by probing model")

    sentinels = list((cfg.get("missing", {}).get("global", {}) or {})
                     .get("sentinels", []) or [])
    if 0 in sentinels or 0.0 in sentinels:
        return 0.0, ("treat_zero_as_missing=false; sentinels contain 0 → "
                     "0 treated as missing")
    return np.nan, ("treat_zero_as_missing=false; sentinels do not contain 0 → "
                    "only NaN treated as missing")


def _build_split_masks(cfg, n_rows, time_values=None, y=None):
    """Return (train_mask, valid_mask, oot_mask) following cfg.training.split."""
    from wdm.utils.time_utils import split_by_yyyymmdd, split_stratified
    split_cfg = cfg["training"]["split"]
    strategy = split_cfg.get("strategy", "stratified")
    ratios = list(split_cfg.get("ratios", [0.70, 0.15, 0.15]))
    seed = int(cfg["training"].get("random_seed", _DEFAULT_SEED))
    if strategy == "time":
        if time_values is None:
            raise ValueError("split.strategy='time' but no time_column in cache.")
        return split_by_yyyymmdd(time_values, ratios)
    if y is None:
        raise ValueError("Stratified split requires y.")
    return split_stratified(y, ratios, seed=seed)


def _resolve_probing_params(cfg):
    """Merge defaults with cfg.analysis.probing.xgb_params."""
    probing = (cfg.get("analysis") or {}).get("probing") or {}
    user_params = dict(probing.get("xgb_params") or {})
    params = dict(_DEFAULT_XGB_PARAMS)
    params.update(user_params)
    # seed lives on xgb_params for xgb 1.5.2
    params.setdefault("seed", int(cfg["training"].get("random_seed", _DEFAULT_SEED)))
    num_boost_round = int(probing.get("num_boost_round", _DEFAULT_NUM_BOOST_ROUND))
    early_stopping = int(probing.get("early_stopping_rounds", _DEFAULT_EARLY_STOPPING))
    return params, num_boost_round, early_stopping


def _importance_to_df(booster, feature_names):
    """Extract gain/weight/cover per feature, aligned to full feature list.

    XGBoost 1.5.2: get_score returns {fN: value} where fN indexes into
    feature_names passed to DMatrix. When DMatrix had feature_names set
    the keys are the original names directly. Features not used by any
    split are absent from get_score → filled with 0.
    """
    rows = {name: {"gain": 0.0, "weight": 0.0, "cover": 0.0}
            for name in feature_names}
    for imp_type in ("gain", "weight", "cover"):
        d = booster.get_score(importance_type=imp_type)
        for k, v in d.items():
            if k in rows:
                rows[k][imp_type] = float(v)
            else:
                # Defensive: fN-style keys if feature_names weren't propagated.
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(feature_names):
                        rows[feature_names[idx]][imp_type] = float(v)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "feature"
    df = df.reset_index()
    return df


def _rank_pct(series):
    """Percentile rank in [0, 1]; ties averaged. NaN → 0."""
    s = pd.Series(series).astype(float)
    r = s.rank(pct=True, method="average")
    return r.fillna(0.0)


def _feature_coverage(X_csr, missing_value, n_features):
    """Per-feature non-missing ratio on the rows used for training.

    Mirrors the DMatrix `missing` semantic so the reported coverage matches
    what the tree actually sees:
      - missing=0.0 → coverage = fraction of rows with an explicit non-NaN
        entry (cache stores no explicit zeros, so any CSR entry that is not
        NaN is a real observation).
      - missing=NaN → coverage = fraction of rows where the cell is not NaN
        (implicit zeros count as observed).

    Exposing coverage lets downstream ranking distinguish a sparse feature
    with weak signal from a sparse feature whose low gain is just low support.
    """
    n_rows = int(X_csr.shape[0])
    out = np.zeros(int(n_features), dtype=np.float64)
    if n_rows == 0:
        return out
    data = X_csr.data
    cols = X_csr.indices
    nan_mask = np.isnan(data)
    is_nan_missing = isinstance(missing_value, float) and np.isnan(missing_value)
    if is_nan_missing:
        nan_per_col = np.bincount(cols[nan_mask], minlength=int(n_features))
        nonmissing = n_rows - nan_per_col
    else:
        nonmissing = np.bincount(cols[~nan_mask], minlength=int(n_features))
    out[:] = nonmissing.astype(np.float64) / float(n_rows)
    return out


def run_probing(cfg, cache_dir, out_dir):
    """Train probing model → write probing_importance.csv + probing_meta.json.

    Returns a dict with paths and summary stats.
    """
    import xgboost as xgb
    from scripts.build_sparse_cache import load_cache  # reuse loader

    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load cache (and verify freshness against the configured CSV)
    csv_path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    cache = load_cache(cache_dir, csv_path=csv_path)
    X = cache["X"]
    y = cache["y"].astype(np.int32)
    feature_names = cache["feature_names"]
    time_col = cache["meta"].get("time_column")
    time_values = cache.get(time_col) if time_col else None

    logger.info("Probing: cache loaded from %s; shape=%s density=%.4f",
                cache_dir, X.shape, cache["meta"]["density"])

    # 2) Resolve missing semantic from config
    missing_value, missing_why = _resolve_missing_value(cfg)
    logger.info("Probing missing=%s (%s)", missing_value, missing_why)

    # 3) Split — use train+valid only; never peek at OOT for probing
    tr_mask, va_mask, oot_mask = _build_split_masks(
        cfg, X.shape[0], time_values=time_values, y=y)
    n_tr = int(np.sum(tr_mask))
    n_va = int(np.sum(va_mask))
    n_oot = int(np.sum(oot_mask))
    logger.info("Probing split: train=%d valid=%d oot=%d (oot untouched)",
                n_tr, n_va, n_oot)

    # 4) Build DMatrix on CSR slices — csr row-slicing is cheap and keeps sparsity.
    X_tr = X[tr_mask]
    X_va = X[va_mask]
    y_tr = y[tr_mask]
    y_va = y[va_mask]
    dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=missing_value,
                         feature_names=list(feature_names))
    dvalid = xgb.DMatrix(X_va, label=y_va, missing=missing_value,
                         feature_names=list(feature_names))

    # 5) Train with early stopping on valid
    params, num_boost_round, early_stopping = _resolve_probing_params(cfg)
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping,
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", num_boost_round - 1))
    best_score = float(getattr(booster, "best_score", float("nan")))
    logger.info("Probing trained: best_iter=%d best_valid_aucpr=%.4f",
                best_iter, best_score)

    # 6) Importance → DataFrame with rank_pct columns
    imp_df = _importance_to_df(booster, list(feature_names))
    imp_df["gain_rank_pct"] = _rank_pct(imp_df["gain"])
    imp_df["weight_rank_pct"] = _rank_pct(imp_df["weight"])
    imp_df["cover_rank_pct"] = _rank_pct(imp_df["cover"])

    # 6b) Per-feature coverage on the rows the model actually fit (train+valid).
    # Aligned to feature_names order before the sort.
    fit_mask = tr_mask | va_mask
    coverage = _feature_coverage(X[fit_mask], missing_value, X.shape[1])
    imp_df["coverage"] = coverage

    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)

    # 7) Write artifacts — probing_importance.csv lives next to summary.csv
    imp_path = out_dir / "probing_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    meta = {
        "n_rows_total": int(X.shape[0]),
        "n_train_rows": n_tr,
        "n_valid_rows": n_va,
        "n_oot_rows": n_oot,
        "n_features": int(len(feature_names)),
        "split_strategy": cfg["training"]["split"].get("strategy"),
        "split_ratios": list(cfg["training"]["split"].get("ratios", [])),
        "missing_value": ("NaN" if isinstance(missing_value, float)
                           and np.isnan(missing_value) else float(missing_value)),
        "missing_why": missing_why,
        "xgb_params": params,
        "num_boost_round_cap": num_boost_round,
        "early_stopping_rounds": early_stopping,
        "best_iteration": best_iter,
        "best_valid_aucpr": best_score,
        "coverage_basis": "train_plus_valid_rows",
        "cache_dir": str(cache_dir),
    }
    (out_dir / "probing_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info("Probing importance written: %s", imp_path)
    return {
        "importance_path": str(imp_path),
        "meta_path": str(out_dir / "probing_meta.json"),
        "best_valid_aucpr": best_score,
        "n_used_features": int((imp_df["gain"] > 0).sum()),
    }
