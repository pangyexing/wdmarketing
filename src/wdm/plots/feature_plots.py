"""Per-feature and per-family plots for Stage 1.

Chinese font handling: we try PingFang SC / Heiti TC / SimHei / Arial Unicode MS
in that order via matplotlib rcParams. If none is found on the machine, the
titles still render — they just show the English name (feature) while the
Chinese label falls back via the cn() helper (which returns the English name
as well). The plots remain useful.

To keep plot count bounded (real data can have 3000+ features), we only plot
the top-N features by rank_score (default 50, configurable).
"""
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wdm.utils.paths import analysis_dir, cn, ensure_dirs, load_column_mapping, per_family_dir, per_feature_dir, report_dir
from wdm.utils.progress import track

logger = logging.getLogger(__name__)


def _configure_cn_font():
    """Register a Chinese-capable font if available; silent no-op otherwise."""
    candidates = ["PingFang SC", "Heiti TC", "SimHei", "Arial Unicode MS",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get(
                    "font.sans-serif", [])
                plt.rcParams["axes.unicode_minus"] = False
                return name
    except Exception as e:
        logger.warning("Could not configure CN font: %s", e)
    return None


def _bilingual_title(feature, mapping):
    c = cn(mapping, feature)
    if c == feature:
        return feature
    return "{0} ({1})".format(c, feature)


def _plot_distribution(ax, values_nan, y, feature, mapping):
    arr = np.asarray(values_nan, dtype=np.float64)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "all missing", ha="center", va="center")
        ax.set_title(_bilingual_title(feature, mapping))
        return
    ax.hist(arr[mask], bins=30, color="#4C78A8", alpha=0.7, edgecolor="white")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax2 = ax.twinx()
    # Positive rate per histogram bin
    counts, edges = np.histogram(arr[mask], bins=30)
    pos = np.zeros_like(counts, dtype=float)
    non_missing_y = y[mask]
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        if i == len(counts) - 1:
            bin_mask = (arr[mask] >= lo) & (arr[mask] <= hi)
        else:
            bin_mask = (arr[mask] >= lo) & (arr[mask] < hi)
        if bin_mask.sum() > 0:
            pos[i] = non_missing_y[bin_mask].mean()
    centers = (edges[:-1] + edges[1:]) / 2
    ax2.plot(centers, pos, color="#E45756", marker="o", linewidth=1.5, markersize=3)
    ax2.set_ylabel("positive rate", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    ax.set_title(_bilingual_title(feature, mapping))


def _plot_woe(ax, bin_spec, feature, mapping):
    edges = np.asarray(bin_spec.edges)
    woes = np.asarray(bin_spec.woe_values)
    cnts = np.asarray(bin_spec.bin_counts)
    if bin_spec.missing_woe is not None:
        woes = np.concatenate([woes, [bin_spec.missing_woe]])
        cnts = np.concatenate([cnts, [bin_spec.missing_n]])
        labels = [str(i) for i in range(len(woes) - 1)] + ["missing"]
    else:
        labels = [str(i) for i in range(len(woes))]
    xs = np.arange(len(woes))
    ax.bar(xs, cnts, color="#B7B7B7", alpha=0.5, width=0.6, label="count")
    ax.set_ylabel("bin count")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(xs, woes, marker="o", color="#4C78A8", linewidth=1.5, label="WOE")
    ax2.axhline(0, color="#888", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("WOE")
    ax.set_title("{0} — IV={1:.3f}".format(_bilingual_title(feature, mapping), bin_spec.iv))


def _plot_psi_bars(ax, values_nan_expected, values_nan_actual, feature, mapping,
                   n_bins=10):
    from wdm.utils.binning import digitize_with_missing, equal_freq_edges
    edges = equal_freq_edges(values_nan_expected, n_bins=n_bins)
    if edges.size < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title(_bilingual_title(feature, mapping))
        return
    n_real = edges.size - 1
    e_bins = digitize_with_missing(values_nan_expected, edges)
    a_bins = digitize_with_missing(values_nan_actual, edges)

    def _pct(bins):
        total = bins.size
        miss = (bins == -1).sum()
        non = bins[bins != -1]
        cnt = np.bincount(non, minlength=n_real)[:n_real]
        full = np.concatenate([cnt, [miss]]).astype(float)
        return full / total if total else full

    e_pct = _pct(e_bins)
    a_pct = _pct(a_bins)
    xs = np.arange(len(e_pct))
    w = 0.35
    ax.bar(xs - w / 2, e_pct, width=w, color="#4C78A8", label="expected")
    ax.bar(xs + w / 2, a_pct, width=w, color="#E45756", label="actual")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in range(n_real)] + ["missing"], rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel("share")
    ax.legend(fontsize=8)
    ax.set_title(_bilingual_title(feature, mapping))


def _plot_lift(ax, values_nan, y, feature, mapping, top_k_pct=0.10):
    from wdm.utils.binning import digitize_with_missing, equal_freq_edges
    arr = np.asarray(values_nan, dtype=np.float64)
    edges = equal_freq_edges(arr, n_bins=10)
    if edges.size < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title(_bilingual_title(feature, mapping))
        return
    bins = digitize_with_missing(arr, edges)
    # Include missing as its own bin at the end
    ordered_bins = []
    for b in range(edges.size - 1):
        m = bins == b
        if m.sum() > 0:
            ordered_bins.append((y[m].mean(), m.sum(), int(y[m].sum())))
    miss = bins == -1
    if miss.sum() > 0:
        ordered_bins.append((y[miss].mean(), miss.sum(), int(y[miss].sum())))
    # Sort bins by positive rate descending
    ordered_bins.sort(key=lambda t: t[0], reverse=True)
    total_n = y.size
    total_pos = y.sum()
    base_rate = total_pos / total_n
    cum_n, cum_pos = 0, 0
    xs, ys_lift, ys_recall = [], [], []
    for pr, n, pos in ordered_bins:
        cum_n += n
        cum_pos += pos
        xs.append(cum_n / total_n)
        ys_lift.append((cum_pos / cum_n) / base_rate if cum_n else 0)
        ys_recall.append(cum_pos / total_pos if total_pos else 0)
    ax.plot(xs, ys_lift, marker="o", color="#4C78A8", label="cum lift")
    ax.axhline(1.0, color="#888", linewidth=0.5, linestyle="--")
    ax.axvline(top_k_pct, color="#E45756", linewidth=0.5, linestyle=":")
    ax.set_xlabel("cumulative population")
    ax.set_ylabel("cumulative lift")
    ax.set_title("{0} — Lift@{1:.0%}".format(_bilingual_title(feature, mapping), top_k_pct))


def _plot_missing(ax, mask, feature, mapping):
    rate = float(np.mean(mask)) if mask.size else 0.0
    ax.bar(["missing", "present"], [rate, 1 - rate], color=["#E45756", "#4C78A8"])
    ax.set_ylim(0, 1)
    ax.set_title("{0} — missing={1:.2%}".format(_bilingual_title(feature, mapping), rate))


def run_per_feature_plots(cfg, bin_specs):
    """Generate per-feature plots for the top-N by rank_score.

    bin_specs: Dict[feature -> BinSpec] produced by compute_iv_table.
    """
    _configure_cn_font()

    mapping = load_column_mapping(cfg)
    summary_csv = report_dir(cfg) / "summary.csv"
    if not summary_csv.is_file():
        logger.warning("summary.csv missing; skipping plots")
        return
    summary = pd.read_csv(summary_csv)
    top_n = int(cfg["analysis"].get("per_feature_plot_top_n", 50))
    top = summary.sort_values("rank_score", ascending=False).head(top_n)

    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    raw = pd.read_csv(path)
    y = raw[cfg["data"]["label_column"]].values

    from wdm.preprocess.missing import build_missing_spec, get_spec, to_nan_array
    spec_map = build_missing_spec(cfg)

    rng = np.random.RandomState(cfg["training"]["random_seed"])
    r = rng.rand(len(raw))
    m_e = r < 0.5
    m_a = ~m_e

    generated = []
    for _, row in track(top.iterrows(), total=len(top),
                        label="per-feature plots", every=5):
        feat = row["feature"]
        spec = get_spec(spec_map, feat)
        arr, mask = to_nan_array(raw[feat], spec, analysis=True)
        outdir = per_feature_dir(cfg, feat)
        ensure_dirs(outdir)
        # dist
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _plot_distribution(ax, arr, y, feat, mapping)
        fig.tight_layout()
        fig.savefig(outdir / "dist.png", dpi=cfg["plots"].get("dpi", 120))
        plt.close(fig)
        # woe
        bs = bin_specs.get(feat)
        if bs is not None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            _plot_woe(ax, bs, feat, mapping)
            fig.tight_layout()
            fig.savefig(outdir / "woe.png", dpi=cfg["plots"].get("dpi", 120))
            plt.close(fig)
        # psi
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _plot_psi_bars(ax, arr[m_e], arr[m_a], feat, mapping,
                       n_bins=int(cfg["analysis"].get("n_bins", 10)))
        fig.tight_layout()
        fig.savefig(outdir / "psi.png", dpi=cfg["plots"].get("dpi", 120))
        plt.close(fig)
        # missing
        fig, ax = plt.subplots(figsize=(4, 3.5))
        _plot_missing(ax, mask, feat, mapping)
        fig.tight_layout()
        fig.savefig(outdir / "missing.png", dpi=cfg["plots"].get("dpi", 120))
        plt.close(fig)
        # lift
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _plot_lift(ax, arr, y, feat, mapping,
                   top_k_pct=float(cfg["training"].get("top_k_pct", 0.10)))
        fig.tight_layout()
        fig.savefig(outdir / "lift.png", dpi=cfg["plots"].get("dpi", 120))
        plt.close(fig)
        generated.append(feat)
    logger.info("Generated %d per-feature plot sets", len(generated))

    # Family comparison plots — only for families with >1 member
    families = summary.dropna(subset=["family_base"])
    from wdm.analysis.family import parse_families
    fam_df = parse_families(summary["feature"].tolist(), cfg)
    fam_counts = fam_df["family_base"].value_counts()
    multi_fams = [b for b, c in fam_counts.items() if c > 1 and
                  fam_df[fam_df["family_base"] == b]["window"].notna().all()]
    for base in multi_fams:
        members_df = summary[summary["feature"].isin(
            fam_df[fam_df["family_base"] == base]["feature"].tolist())]
        if members_df.empty:
            continue
        outdir = per_family_dir(cfg, base)
        ensure_dirs(outdir)
        # IV by window
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ivs = members_df.sort_values("window")
        ax.bar(ivs["window"].astype(str), ivs["iv"], color="#4C78A8")
        ax.set_ylabel("IV")
        ax.set_title("family: {0} — IV by window".format(base))
        fig.tight_layout()
        fig.savefig(outdir / "iv_by_window.png", dpi=cfg["plots"].get("dpi", 120))
        plt.close(fig)
    return generated
