"""Stage-1 scoring and filtering: combines PSI / IV / Lift / correlation /
family signals into a single ranked feature report with auto-keep flags.
(Pipeline orchestration and artifact writing live in wdm.pipeline.stage1.)

Rank score:
    rank_score = z(iv) + z(lift_at_k) + z(gini)
               − psi_penalty_weight · z(psi)              # soft, configurable
               − 0.5 · 1[missing_rate > 0.5]
               − window_penalty(window, group)
               + probing_weight · z(gain_rank_pct)        # when probing enabled

Auto-keep rule (the feature passes into v1_auto.txt):
    family_kept AND group_kept
    AND (cluster_id is singleton OR is cluster's max-rank member)
    AND (psi_mode != 'hard' OR psi < psi_cutoff)
    AND missing_rate < missing_rate_max_for_window
    AND iv >= iv_min

PSI role is deliberately **soft by default** (psi_mode='soft'):
  - 'hard': psi >= psi_cutoff drops the feature outright. Legacy behavior.
  - 'soft' (default): high-PSI only penalizes rank_score; features stay in
    the pool. Tree models often extract conditional signal from drifted
    features (rank relations can survive mean shifts).
  - 'off': PSI is informational only, no effect on selection.

missing_rate_max_for_window: short-window features (analysis.short_windows,
default 7d/30d) get the softer analysis.short_window_missing_rate_max cap
(default 0.98) instead of the global missing_rate_max, since "business didn't
happen in the window" ≠ "data quality".
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from wdm.analysis.family import effective_family_policy

logger = logging.getLogger(__name__)


def _zscore(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s.fillna(mu) - mu) / sigma


def _build_ranked_report(iv_df, psi_df, lift_df, missing_df, family_df, semantic_df,
                         cluster_map, cfg, probing_df=None):
    """Merge all per-feature tables on 'feature' into one.

    probing_df: optional DataFrame with columns (feature, gain, weight, cover,
    gain_rank_pct, ...) from Stage-1 probing model. Left-joined when present.
    """
    df = iv_df.merge(psi_df, on="feature", how="outer")
    df = df.merge(lift_df, on="feature", how="outer")
    df = df.merge(missing_df, on="feature", how="outer")
    df = df.merge(family_df, on="feature", how="left")
    df = df.merge(semantic_df.drop(columns=["group_description"], errors="ignore"),
                  on="feature", how="left")
    df["corr_cluster"] = df["feature"].map(cluster_map).fillna(-1).astype(int)

    fillna_map = {
        "iv": 0.0, "psi": 0.0, "lift_at_k": 1.0, "gini": 0.0,
        "concentration": 0.0, "missing_rate": 0.0, "n_unique": 0,
    }

    if probing_df is not None and len(probing_df):
        keep = [c for c in ("feature", "gain", "weight", "cover",
                            "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
                            "coverage")
                if c in probing_df.columns]
        pr = probing_df[keep].rename(columns={
            "gain": "probe_gain", "weight": "probe_weight", "cover": "probe_cover"})
        df = df.merge(pr, on="feature", how="left")
        for c in ("probe_gain", "probe_weight", "probe_cover",
                  "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
                  "coverage"):
            if c in df.columns:
                fillna_map[c] = 0.0

    df = df.fillna(fillna_map)
    return df


def _resolve_psi_knobs(cfg):
    """Pull psi_mode / psi_penalty_weight with sane fallbacks.

    psi_mode ∈ {"soft", "hard", "off"}. Historically PSI was a hard filter
    (implicit 'hard'). We default to 'soft' now — high PSI only dampens
    rank_score, it does not drop the feature. Products that really want the
    old behavior can set analysis.psi_mode: hard.
    """
    ana = cfg.get("analysis", {}) or {}
    mode = str(ana.get("psi_mode", "soft")).lower()
    if mode not in ("soft", "hard", "off"):
        mode = "soft"
    weight = float(ana.get("psi_penalty_weight", 0.25))
    return mode, weight


def _apply_hard_filters(df, cfg):
    miss_max = float(cfg["analysis"]["missing_rate_max"])
    iv_min = float(cfg["analysis"]["iv_min"])
    psi_cutoff = float(cfg["analysis"]["psi_cutoff"])
    psi_mode, _ = _resolve_psi_knobs(cfg)
    lift_keep_min = cfg["analysis"].get("lift_keep_min")
    lift_keep_min = float(lift_keep_min) if lift_keep_min is not None else None
    short_windows = set(cfg["analysis"].get("short_windows") or ["7d", "30d"])
    short_cap = float(cfg["analysis"].get("short_window_missing_rate_max", 0.98))

    if "window" in df.columns:
        is_short = df["window"].isin(short_windows).values
    else:
        is_short = np.zeros(len(df), dtype=bool)
    mr_cap = np.where(is_short, max(miss_max, short_cap), miss_max)

    # NaN comparisons are False, matching the legacy per-row behavior.
    with np.errstate(invalid="ignore"):
        constant = df["n_unique"].values <= 1
        high_missing = df["missing_rate"].values > mr_cap
        low_iv = df["iv"].values < iv_min
        if lift_keep_min is not None:
            # Positive-oriented soft gate: keep a weak-IV feature if it still
            # ranks positives well (lift_at_k >= lift_keep_min).
            if "lift_at_k" in df.columns:
                lift_vals = df["lift_at_k"].astype(float).values
            else:
                lift_vals = np.zeros(len(df), dtype=np.float64)
            low_iv = low_iv & (lift_vals < lift_keep_min)
        # PSI only hard-drops in psi_mode='hard'; soft/off keep the feature
        # (rank_score penalty / informational flag handle it downstream).
        if psi_mode == "hard":
            high_psi = df["psi"].values >= psi_cutoff
        else:
            high_psi = np.zeros(len(df), dtype=bool)

    reasons = []
    for c, hm, li, hp in zip(constant, high_missing, low_iv, high_psi):
        drop = []
        if c:
            drop.append("constant")
        if hm:
            drop.append("high_missing")
        if li:
            drop.append("low_iv")
        if hp:
            drop.append("high_psi")
        reasons.append(";".join(drop))
    df["_hard_drop"] = [bool(r) for r in reasons]
    df["_hard_drop_reason"] = reasons
    # Informational flag: annotate high-PSI features even in soft/off mode
    # so the report still shows drift risk without dropping the feature.
    df["psi_over_cutoff"] = (df["psi"] >= psi_cutoff).astype(bool)
    return df


def _penalty_table_for(policy, cfg):
    """Resolve window_penalty_table for a given effective policy; fall back to linear."""
    table = policy.get("window_penalty_table") or {}
    table = {str(k): float(v) for k, v in table.items()}
    if not table:
        order = list((cfg.get("feature_groups") or {}).get("window_order") or [])
        n = max(len(order), 1)
        table = {w: (i / n) * 0.3 for i, w in enumerate(order)}
    return table


def _row_penalty_contribution(df, cfg):
    """Per-row `gamma * penalty(window)` resolved from each row's semantic_group.

    When a semantic_group declares its own family_policy, that row uses the
    group's gamma and penalty_table; otherwise falls back to the global policy.
    Rows without a canonical string window contribute 0.
    """
    default_policy = (cfg.get("feature_groups") or {}).get("family_policy") or {}
    # Pre-resolve per semantic_group to avoid repeated dict merges.
    group_policies: Dict[Optional[str], Dict[str, Any]] = {None: default_policy}
    seen_groups = df.get("semantic_group")
    if seen_groups is not None:
        for g in seen_groups.dropna().unique():
            group_policies[str(g)] = effective_family_policy(str(g), cfg)
    group_gamma = {g: float(p.get("window_penalty_gamma", 0.0))
                   for g, p in group_policies.items()}
    group_table = {g: _penalty_table_for(p, cfg) for g, p in group_policies.items()}

    def _score(row):
        w = row.get("window")
        if not isinstance(w, str):
            return 0.0
        g = row.get("semantic_group")
        key = str(g) if isinstance(g, str) else None
        tbl = group_table.get(key, group_table[None])
        gamma = group_gamma.get(key, group_gamma[None])
        return gamma * float(tbl.get(w, 0.0))

    return df.apply(_score, axis=1).astype(float)


def _rank_and_auto_keep(df, cfg):
    df = df.copy()
    psi_mode, psi_weight = _resolve_psi_knobs(cfg)
    w = (cfg.get("analysis") or {}).get("rank_weights") or {}
    w_iv = float(w.get("iv", 1.0))
    w_lift = float(w.get("lift", 1.0))
    w_gini = float(w.get("gini", 1.0))
    w_conc = float(w.get("concentration", 0.0))
    w_miss = float(w.get("missing_penalty", 0.5))
    miss_thr = float(w.get("missing_penalty_threshold", 0.5))
    # PSI contribution to rank_score:
    #   - 'off'  : zero weight — PSI is informational only
    #   - else   : rank_weights.psi when explicitly configured, otherwise
    #              analysis.psi_penalty_weight (default 0.25 — meaningfully
    #              smaller than z(iv) / z(lift_at_k) / z(gini) each at 1.0)
    w_psi = float(w["psi"]) if "psi" in w else psi_weight
    effective_psi_weight = 0.0 if psi_mode == "off" else w_psi
    conc_term = (w_conc * _zscore(df["concentration"])
                 if "concentration" in df.columns else 0.0)
    df["rank_score"] = (
        w_iv * _zscore(df["iv"])
        + w_lift * _zscore(df["lift_at_k"])
        + w_gini * _zscore(df["gini"])
        + conc_term
        - effective_psi_weight * _zscore(df["psi"])
        - w_miss * (df["missing_rate"] > miss_thr).astype(float)
        - _row_penalty_contribution(df, cfg)
    )

    # Probing model contribution: add when Stage 1 probing wrote gain_rank_pct.
    # Weight is config-driven (analysis.probing.weight_in_rank_score, default 0.25).
    probing_cfg = (cfg.get("analysis") or {}).get("probing") or {}
    w_probe = float(probing_cfg.get("weight_in_rank_score", 0.25))
    if "gain_rank_pct" in df.columns and w_probe > 0:
        # gain_rank_pct is already in [0,1]; z-score makes it commensurate with
        # the other z-scored terms.
        df["rank_score"] = df["rank_score"] + w_probe * _zscore(df["gain_rank_pct"])

        # Coverage-stratified gain rank: features compete for "high gain" only
        # against peers of similar support. Raw gain/split-count is biased
        # toward dense features (more non-missing rows → more splits), so a
        # sparse feature with real conditional signal can be unfairly parked
        # in "noise" by the global ranking. Binning by coverage quintile and
        # ranking gain within each bin neutralizes that bias.
        if "coverage" in df.columns:
            try:
                cov_bin = pd.qcut(df["coverage"], q=5, labels=False,
                                   duplicates="drop")
            except ValueError:
                # Degenerate coverage (all equal) — fall back to a single bin.
                cov_bin = pd.Series(0, index=df.index)
            df["gain_rank_pct_by_coverage"] = (
                df.groupby(cov_bin)["gain_rank_pct"]
                  .rank(pct=True, method="average")
                  .fillna(0.0)
            )
            gain_high = df["gain_rank_pct_by_coverage"] > 0.7
        else:
            gain_high = df["gain_rank_pct"] > 0.7

        # Quadrant labels — expose what probing adds vs what IV already said.
        iv_high = df["iv"].rank(pct=True) > 0.7
        df["discover"] = (~iv_high) & gain_high
        df["stable"]   = iv_high & gain_high
        df["interp"]   = iv_high & (~gain_high)
        df["noise"]    = (~iv_high) & (~gain_high)

    # Within each correlation cluster, only the top-score survivor passes.
    # Winner pick keeps the legacy stable sort so rank_score ties resolve to
    # the member appearing first in the frame.
    cluster_winners = set()
    winner_by_cid = {}
    for cid, block in df.groupby("corr_cluster"):
        if cid == -1 or len(block) <= 1:
            cluster_winners.update(block["feature"].tolist())
            continue
        best = block.sort_values("rank_score", ascending=False).iloc[0]["feature"]
        cluster_winners.add(best)
        winner_by_cid[cid] = best

    n = len(df)
    if "family_kept" in df.columns:
        # bool(NaN) is True in the legacy row loop → fillna(True).
        fam_dropped = ~df["family_kept"].fillna(True).astype(bool).values
    else:
        fam_dropped = np.zeros(n, dtype=bool)
    if "group_kept" in df.columns:
        grp_dropped = ~df["group_kept"].fillna(True).astype(bool).values
    else:
        grp_dropped = np.zeros(n, dtype=bool)
    not_winner = (~df["feature"].isin(cluster_winners)).values
    hard_reasons = (df["_hard_drop_reason"].tolist()
                    if "_hard_drop_reason" in df.columns else [None] * n)
    cids = df["corr_cluster"].tolist()

    auto_keep = []
    drop_reason = []
    for k, feat_reason in enumerate(hard_reasons):
        reasons = []
        if feat_reason:
            reasons.append(feat_reason)
        if fam_dropped[k]:
            reasons.append("family_dropped_by_policy")
        if grp_dropped[k]:
            reasons.append("group_dropped_by_policy")
        if not_winner[k]:
            # A non-winner always belongs to a multi-member cluster, which
            # always has a recorded winner.
            reasons.append("corr_dup_of:{0}".format(winner_by_cid.get(cids[k], "?")))
        auto_keep.append(not reasons)
        drop_reason.append(";".join(reasons))
    df["auto_keep"] = auto_keep
    df["drop_reason"] = drop_reason
    return df.drop(columns=["_hard_drop", "_hard_drop_reason"], errors="ignore")


# Backward-compatible entrypoint: the Stage-1 orchestration moved to
# wdm.pipeline.stage1; existing callers keep importing it from here.
from wdm.pipeline.stage1 import run_stage1  # noqa: E402,F401
