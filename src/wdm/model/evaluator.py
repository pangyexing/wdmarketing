"""Compute the required suite of metrics for train/valid/oot and assemble reports.

Metrics per split:
  * roc_auc, pr_auc, ks
  * precision_at_k, recall_at_k, lift_at_k, top_k_cvr (k = cfg.training.top_k_pct)
  * binned_lift (10 deciles)

Also audits which families/semantic groups dominate feature importance — a
>40% single-family gain share triggers a WARNING so the user reviews whether
one business dimension is bleeding importance into the model.
"""
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from wdm.metrics.binned_lift import compute_binned_lift
from wdm.metrics.ks import ks_stat
from wdm.metrics.pr_auc import pr_auc, roc_auc
from wdm.metrics.ranking import lift_at_k, precision_at_k, recall_at_k, top_k_cvr

logger = logging.getLogger(__name__)


def _predict(booster, X):
    dmat = xgb.DMatrix(X)
    try:
        return booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
    except (TypeError, xgb.core.XGBoostError):
        # Older xgb variants
        return booster.predict(dmat, ntree_limit=booster.best_ntree_limit)


def _metrics_for_split(name, y, score, top_k_pct):
    return {
        "split": name,
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "roc_auc": roc_auc(y, score),
        "pr_auc": pr_auc(y, score),
        "ks": ks_stat(y, score),
        "precision_at_k": precision_at_k(y, score, top_k_pct),
        "recall_at_k": recall_at_k(y, score, top_k_pct),
        "lift_at_k": lift_at_k(y, score, top_k_pct),
        "top_k_cvr": top_k_cvr(y, score, top_k_pct),
        "top_k_pct": float(top_k_pct),
    }


def evaluate_all(booster, data, cfg):
    """Return (metrics_df, binned_lifts_dict, scores_dict, importance_df).

    metrics_df is a wide table with one row per split.
    """
    top_k_pct = float(cfg["training"].get("top_k_pct", 0.10))

    s_tr = _predict(booster, data.X_train)
    s_va = _predict(booster, data.X_valid)
    s_oot = _predict(booster, data.X_oot)

    rows = [
        _metrics_for_split("train", data.y_train, s_tr, top_k_pct),
        _metrics_for_split("valid", data.y_valid, s_va, top_k_pct),
        _metrics_for_split("oot",   data.y_oot,   s_oot, top_k_pct),
    ]
    metrics_df = pd.DataFrame(rows)

    binned = {
        "train": compute_binned_lift(data.y_train, s_tr, n_bins=10),
        "valid": compute_binned_lift(data.y_valid, s_va, n_bins=10),
        "oot":   compute_binned_lift(data.y_oot,   s_oot, n_bins=10),
    }

    # Feature importance
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    # XGBoost keys features as f0, f1, ... by default when trained from DMatrix(numpy).
    # We own the column order via data.feature_list → map fN → feature_list[N].
    n_feat = len(data.feature_list)
    rename = {"f{0}".format(i): data.feature_list[i] for i in range(n_feat)}
    gain = {rename.get(k, k): v for k, v in gain.items()}
    weight = {rename.get(k, k): v for k, v in weight.items()}
    cover = {rename.get(k, k): v for k, v in cover.items()}
    imp = pd.DataFrame({
        "feature": data.feature_list,
        "gain": [gain.get(f, 0.0) for f in data.feature_list],
        "weight": [weight.get(f, 0.0) for f in data.feature_list],
        "cover": [cover.get(f, 0.0) for f in data.feature_list],
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    return metrics_df, binned, {"train": s_tr, "valid": s_va, "oot": s_oot}, imp


def family_importance_audit(importance_df, cfg):
    """Group gain by family_base; warn if any family commands >40% of total gain."""
    from wdm.analysis.family import parse_families
    fam = parse_families(importance_df["feature"].tolist(), cfg)
    merged = importance_df.merge(fam, on="feature", how="left")
    total_gain = float(merged["gain"].sum())
    if total_gain <= 0:
        return pd.DataFrame()
    audit = merged.groupby("family_base", dropna=False).agg(
        total_gain=("gain", "sum"),
        kept_count=("feature", "count")).reset_index()
    audit["gain_share"] = audit["total_gain"] / total_gain
    audit = audit.sort_values("gain_share", ascending=False).reset_index(drop=True)
    for _, r in audit.iterrows():
        if r["gain_share"] > 0.40 and r["kept_count"] > 1:
            logger.warning("Family '%s' owns %.1f%% of total gain across %d features — "
                           "consider trimming the feature list.", r["family_base"],
                           r["gain_share"] * 100, r["kept_count"])
    return audit


def write_metrics_artifacts(out_dir, metrics_df, binned, imp_df, audit_df,
                            run_manifest, best_params):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        metrics_df.to_json(orient="records", indent=2), encoding="utf-8")
    imp_df.to_csv(out_dir / "importance.csv", index=False)
    for name, df in binned.items():
        df.to_csv(out_dir / "binned_lift_{0}.csv".format(name), index=False)
    if audit_df is not None and len(audit_df):
        audit_df.to_csv(out_dir / "family_gain_audit.csv", index=False)
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # Markdown summary
    md_lines = ["# Training metrics", "", "| split | n | pos_rate | PR-AUC | ROC-AUC | KS | P@K | R@K | Lift@K | Top-K CVR |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in metrics_df.iterrows():
        md_lines.append("| {0} | {1} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} | {6:.4f} | {7:.4f} | {8:.4f} | {9:.4f} |"
                        .format(r["split"], r["n"], r["base_rate"],
                                r["pr_auc"], r["roc_auc"], r["ks"],
                                r["precision_at_k"], r["recall_at_k"],
                                r["lift_at_k"], r["top_k_cvr"]))
    (out_dir / "metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
