"""Model-level plots: ROC / PR / KS / gain / calibration / SHAP summary."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from wdm.utils.paths import cn, load_column_mapping

logger = logging.getLogger(__name__)


def _ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def plot_roc_pr(scores_dict, y_dict, out_dir):
    _ensure(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("train", "valid", "oot"):
        y = y_dict[name]
        s = scores_dict[name]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label="{0} (AUC={1:.3f})".format(
            name, roc_auc_score(y, s)))
        prec, rec, _ = precision_recall_curve(y, s)
        axes[1].plot(rec, prec, label="{0} (AP={1:.3f})".format(
            name, average_precision_score(y, s)))
    axes[0].plot([0, 1], [0, 1], "--", color="#888", linewidth=0.5)
    axes[0].set_title("ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].set_title("PR"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "roc_pr.png", dpi=120)
    plt.close(fig)


def plot_ks(scores_dict, y_dict, out_dir):
    _ensure(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, name in enumerate(("train", "valid", "oot")):
        y = np.asarray(y_dict[name]).astype(int)
        s = np.asarray(scores_dict[name])
        if len(np.unique(y)) < 2:
            continue
        order = np.argsort(-s, kind="stable")
        ys = y[order]
        n_pos = ys.sum(); n_neg = ys.size - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        tpr = np.cumsum(ys) / n_pos
        fpr = np.cumsum(1 - ys) / n_neg
        ks = np.max(np.abs(tpr - fpr))
        axes[i].plot(np.arange(len(tpr)) / len(tpr), tpr, label="TPR")
        axes[i].plot(np.arange(len(fpr)) / len(fpr), fpr, label="FPR")
        axes[i].plot(np.arange(len(fpr)) / len(fpr), np.abs(tpr - fpr), label="|TPR-FPR|")
        axes[i].set_title("{0} KS={1:.3f}".format(name, ks))
        axes[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "ks.png", dpi=120)
    plt.close(fig)


def plot_gain_decile(binned_dict, out_dir):
    _ensure(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in binned_dict.items():
        ax.plot(df["cum_pop_share"], df["cum_recall"], marker="o", label="{0} gain".format(name))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=0.5, label="baseline")
    ax.set_xlabel("cumulative population"); ax.set_ylabel("cumulative recall")
    ax.set_title("Gain chart")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(out_dir) / "gain.png", dpi=120)
    plt.close(fig)

    # Lift per decile (valid as the display split)
    if "valid" in binned_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        df = binned_dict["valid"]
        ax.bar(df["bin"], df["cum_lift"], color="#4C78A8")
        ax.axhline(1.0, linestyle="--", color="#888", linewidth=0.5)
        ax.set_xlabel("decile"); ax.set_ylabel("cum lift")
        ax.set_title("Lift-per-decile (valid)")
        fig.tight_layout(); fig.savefig(Path(out_dir) / "lift_decile.png", dpi=120)
        plt.close(fig)


def plot_calibration(scores_dict, y_dict, out_dir, n_bins=10):
    _ensure(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("train", "valid", "oot"):
        y = np.asarray(y_dict[name]).astype(int); s = np.asarray(scores_dict[name])
        if len(s) == 0: continue
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(s, bins) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0: continue
            xs.append(s[mask].mean()); ys.append(y[mask].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=0.5)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed positive rate")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(out_dir) / "calibration.png", dpi=120)
    plt.close(fig)


def plot_importance(imp_df, out_dir, mapping=None, top_n=20):
    _ensure(out_dir)
    df = imp_df.head(top_n).copy()
    if mapping:
        df["label"] = df["feature"].map(lambda f: "{0} ({1})".format(
            cn(mapping, f), f) if cn(mapping, f) != f else f)
    else:
        df["label"] = df["feature"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    ax.barh(df["label"][::-1], df["gain"][::-1], color="#4C78A8")
    ax.set_xlabel("gain")
    ax.set_title("Top-{0} feature importance (gain)".format(top_n))
    fig.tight_layout(); fig.savefig(Path(out_dir) / "importance_gain.png", dpi=120)
    plt.close(fig)


def plot_shap_summary(booster, X_sample, feature_names, out_dir, mapping=None, max_display=20):
    """SHAP summary plots. Wraps the occasional TreeExplainer failure gracefully."""
    _ensure(out_dir)
    try:
        import shap
    except Exception as e:
        logger.warning("shap import failed (%s) — skipping SHAP summary", e)
        return
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_sample)
        display_names = feature_names
        if mapping:
            display_names = ["{0} ({1})".format(cn(mapping, f), f) if cn(mapping, f) != f else f
                             for f in feature_names]
        plt.figure(figsize=(8, max(4, 0.3 * min(len(feature_names), max_display))))
        shap.summary_plot(shap_values, X_sample, feature_names=display_names,
                          plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "shap_bar.png", dpi=120, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(8, max(4, 0.3 * min(len(feature_names), max_display))))
        shap.summary_plot(shap_values, X_sample, feature_names=display_names,
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("SHAP failed: %s", e)


def make_all_model_plots(cfg, booster, data, scores, binned, imp_df, out_dir):
    mapping = load_column_mapping(cfg)
    y_dict = {"train": data.y_train, "valid": data.y_valid, "oot": data.y_oot}
    plot_roc_pr(scores, y_dict, out_dir)
    plot_ks(scores, y_dict, out_dir)
    plot_gain_decile(binned, out_dir)
    plot_calibration(scores, y_dict, out_dir)
    plot_importance(imp_df, out_dir, mapping=mapping)
    # SHAP on a sampled subset of validation
    n_sample = min(2000, len(data.X_valid))
    if n_sample > 0:
        rng = np.random.RandomState(cfg["training"]["random_seed"])
        idx = rng.choice(len(data.X_valid), size=n_sample, replace=False)
        plot_shap_summary(booster, data.X_valid[idx], data.feature_list,
                          out_dir, mapping=mapping)
