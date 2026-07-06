"""Stage-1.5 model-based feature screen: target-permutation null importance.

Idea: a feature's XGBoost gain is only meaningful relative to the gain it
collects when the label carries no signal. Train a few small fixed-parameter
models on the REAL label (averaged → gain_actual), then many on SHUFFLED
labels (→ per-feature null gain distribution). Keep features whose actual
gain exceeds the configured percentile of their own null distribution.

This is an OPTIONAL refinement on top of the Stage-1 statistical screen — it
consumes a selected-features list (default: the active version, normally
v1_auto) and writes a smaller <out_version>.txt (default v2_model) that
Stage-2 can consume via `run_training.py --features-version v2_model`.
It never touches v1_auto.txt or the Stage-1 report.

Training uses the TRAIN split only (for xc products that is a time split, so
no future rows leak into the screen). Sample weights are intentionally not
applied — the screen measures ranking signal per person, mirroring how
Stage-2 evaluates.

Requires xgboost; run with the ML conda environment (see
scripts/run_model_screen.py).
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small, mildly regularized trees: enough capacity to surface real signal,
# cheap enough to train n_actual_runs + n_null_runs times.
_FIXED_PARAMS = {
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}


def _resolve_params(cfg, ni_cfg, y_train):
    params = dict(cfg["training"].get("xgb_base_params") or {})
    params.update(_FIXED_PARAMS)
    pos = float((np.asarray(y_train) == 1).sum())
    neg = float(np.asarray(y_train).size - pos)
    if pos > 0:
        params["scale_pos_weight"] = neg / pos
    params.update(ni_cfg.get("xgb_params") or {})
    params.pop("eval_metric", None)
    return params


def _gain_vector(booster, feature_list, importance_type):
    score = booster.get_score(importance_type=importance_type)
    return np.array([float(score.get(f, 0.0)) for f in feature_list],
                    dtype=np.float64)


def run_null_importance(cfg, base_version=None, out_version=None,
                        n_actual_runs=None, n_null_runs=None, seed=None,
                        data=None):
    """Run the screen and materialize:
      * report/null_importance.csv            — per-feature gains + verdict
      * selected_features/<out_version>.txt   — kept features, parent=base
      * analysis/null_importance_meta.json    — parameters for reproducibility
    Returns a summary dict.

    data: optionally pass an already-built StageTwoData for base_version to
    skip the full CSV reload (callers that just built the same dataset).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise RuntimeError(
            "xgboost is not available in this Python environment — run the "
            "model screen with the ML conda env (see scripts/run_model_screen.py).")

    from wdm.pipeline.stage1 import report_hash, write_auto_features_txt
    from wdm.model.dataset import build_dataset
    from wdm.utils.paths import (
        analysis_dir, ensure_dirs, inject_cn_column, load_column_mapping,
        report_dir, selected_features_dir,
    )
    from wdm.utils.progress import track

    ni_cfg = (cfg.get("analysis") or {}).get("null_importance") or {}
    base_version = (base_version
                    or cfg.get("selected_features", {}).get("active_version")
                    or "v1_auto")
    out_version = out_version or str(ni_cfg.get("out_version", "v2_model"))
    n_actual = int(n_actual_runs if n_actual_runs is not None
                   else ni_cfg.get("n_actual_runs", 3))
    n_null = int(n_null_runs if n_null_runs is not None
                 else ni_cfg.get("n_null_runs", 30))
    n_rounds = int(ni_cfg.get("n_boost_rounds", 100))
    keep_pct = float(ni_cfg.get("keep_percentile", 95))
    imp_type = str(ni_cfg.get("importance_type", "gain"))
    seed = int(seed if seed is not None
               else cfg["training"].get("random_seed", 42))
    max_features = ni_cfg.get("max_features")
    top_n = int(max_features) if max_features else int(
        cfg["training"]["final_feature_count"])

    t0 = time.time()
    if data is None:
        data = build_dataset(cfg, version=base_version)
    X = data.X_train
    y = data.y_train
    feature_list = list(data.feature_list)
    base_features = set(data.base_feature_list)
    logger.info("null importance: %d train rows × %d features (base=%s, "
                "%d actual + %d null runs × %d rounds)",
                X.shape[0], X.shape[1], base_version, n_actual, n_null, n_rounds)

    params = _resolve_params(cfg, ni_cfg, y)

    def _train_gain(labels, run_seed):
        p = dict(params)
        p["seed"] = int(run_seed)
        dtrain = xgb.DMatrix(X, label=labels, feature_names=feature_list)
        booster = xgb.train(params=p, dtrain=dtrain, num_boost_round=n_rounds)
        return _gain_vector(booster, feature_list, imp_type)

    actual_gains = np.vstack([
        _train_gain(y, seed + i)
        for i in track(range(n_actual), total=n_actual,
                       label="null importance: actual runs")])
    gain_actual = actual_gains.mean(axis=0)

    rng = np.random.RandomState(seed)
    null_gains = np.vstack([
        _train_gain(rng.permutation(y), seed + 1000 + i)
        for i in track(range(n_null), total=n_null,
                       label="null importance: null runs")])

    null_p50 = np.percentile(null_gains, 50, axis=0)
    null_p75 = np.percentile(null_gains, 75, axis=0)
    null_p95 = np.percentile(null_gains, 95, axis=0)
    null_keep_ref = np.percentile(null_gains, keep_pct, axis=0)
    keep = gain_actual > null_keep_ref
    # Ranking score: how far the actual gain sits above the null reference,
    # log-compressed so a handful of huge-gain features don't dwarf the rest.
    score = np.log1p(gain_actual) - np.log1p(null_p75)

    report = pd.DataFrame({
        "feature": feature_list,
        "gain_actual": gain_actual,
        "null_p50": null_p50,
        "null_p75": null_p75,
        "null_p95": null_p95,
        "null_keep_ref": null_keep_ref,
        "score": score,
        "keep": keep,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    rdir = report_dir(cfg)
    ensure_dirs(rdir)
    mapping = load_column_mapping(cfg)
    report_out = inject_cn_column(report, mapping)
    csv_path = rdir / "null_importance.csv"
    report_out.to_csv(csv_path, index=False)

    # Feature list: only base features are eligible (indicator columns like
    # foo__isnan are derived inside Stage-2 and must not appear in the list).
    eligible = report[report["feature"].isin(base_features)]
    out_df = pd.DataFrame({
        "feature": eligible["feature"],
        "rank_score": eligible["score"],
        "auto_keep": eligible["keep"],
    })
    sf_dir = selected_features_dir(cfg)
    ensure_dirs(sf_dir)
    out_path = sf_dir / "{0}.txt".format(out_version)
    write_auto_features_txt(out_df, out_path, top_n=top_n,
                            report_hash=report_hash(csv_path),
                            parent=base_version,
                            source="analysis/null_importance.py")

    n_kept_written = int(min(int(eligible["keep"].sum()), top_n))
    meta = {
        "base_version": base_version,
        "out_version": out_version,
        "n_features_in": len(feature_list),
        "n_kept": int(report["keep"].sum()),
        "n_written": n_kept_written,
        "n_actual_runs": n_actual,
        "n_null_runs": n_null,
        "n_boost_rounds": n_rounds,
        "keep_percentile": keep_pct,
        "importance_type": imp_type,
        "seed": seed,
        "xgb_params": {k: str(v) for k, v in params.items()},
        "train_rows": int(X.shape[0]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    meta_path = analysis_dir(cfg) / "null_importance_meta.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("null importance done: kept %d / %d features → %s (%.1fs)",
                meta["n_kept"], meta["n_features_in"], out_path,
                meta["elapsed_seconds"])
    return {
        "report_csv": str(csv_path),
        "features_txt": str(out_path),
        "meta_json": str(meta_path),
        "n_features_in": meta["n_features_in"],
        "n_kept": meta["n_kept"],
        "n_written": meta["n_written"],
    }
