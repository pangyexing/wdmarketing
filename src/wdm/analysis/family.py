"""Time-window family + business-semantic group analysis.

Three kinds of correlation to handle differently:
  (A) Time-window families (auto): feat_{7d,30d,90d,all}. Name-pattern detectable.
  (B) Semantic groups (manual):     机构数 / 还款金额 / 申请数 — user declares.
  (C) Plain numerical:              falls back to correlation.py union-find.

This module enriches the feature report with:
  * family_base, window, window_rank, family_size, in_family_rank, family_kept
  * semantic_group, group_size, in_group_rank, group_kept

And provides `apply_group_correlation` which tightens the corr cutoff inside
a family (0.90) or semantic group (0.85) while keeping the global cutoff (0.95).
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_families(features, cfg):
    """Parse feature list for time-window membership.

    Returns DataFrame[feature, family_base, window, window_rank].
    """
    pattern = cfg["feature_groups"]["window_pattern"]
    order = cfg["feature_groups"]["window_order"]
    rank_map = {w: i for i, w in enumerate(order)}
    regex = re.compile(pattern)

    rows = []
    for f in features:
        m = regex.match(f)
        if m:
            base = m.group("base")
            w = m.group("window")
            rows.append({
                "feature": f,
                "family_base": base,
                "window": w,
                "window_rank": int(rank_map.get(w, len(order))),
            })
        else:
            rows.append({
                "feature": f,
                "family_base": f,
                "window": None,
                "window_rank": None,
            })
    return pd.DataFrame(rows)


def parse_semantic_groups(features, cfg):
    """Return DataFrame[feature, semantic_group, group_prefer, group_max_keep, group_corr_cutoff]

    A group can declare its members via:
      - `features: [list, of, names]`  — explicit list, or
      - `feature_prefix: "bureau_"`    — match all features whose name starts with it
    If both are set, `features` is treated as an additive explicit list on top of the prefix match.
    Unmatched features get semantic_group=None.
    """
    groups = (cfg.get("feature_groups") or {}).get("semantic_groups") or []
    feat_set = set(features)
    rows = []
    missing_by_group = {}
    assigned = {}

    for g in groups:
        name = g["name"]
        declared = list(g.get("features", []) or [])
        prefix = g.get("feature_prefix")
        matched = []
        if prefix:
            matched = [f for f in features if f.startswith(prefix)]
        explicit_matched = [f for f in declared if f in feat_set]
        group_members = list(dict.fromkeys(matched + explicit_matched))
        missing_by_group[name] = [f for f in declared if f not in feat_set]

        for f in group_members:
            if f in assigned:
                logger.warning("Feature %s declared in multiple semantic groups: %s and %s",
                               f, assigned[f], name)
                continue
            assigned[f] = name
            rows.append({
                "feature": f,
                "semantic_group": name,
                "group_prefer": g.get("prefer", "best_iv"),
                "group_max_keep": int(g.get("max_keep", 2)),
                "group_corr_cutoff": float(g.get("corr_cutoff_in_group", 0.85)),
                "group_description": g.get("description", ""),
            })

    for f in features:
        if f not in assigned:
            rows.append({
                "feature": f, "semantic_group": None,
                "group_prefer": None, "group_max_keep": None,
                "group_corr_cutoff": None, "group_description": None,
            })
    df = pd.DataFrame(rows).drop_duplicates("feature", keep="first")
    return df, missing_by_group


def rank_within_family(feature_report, cfg):
    """Annotate feature_report with family_size, in_family_rank, family_kept.

    Expects feature_report to already contain family_base, window, window_rank, iv.
    """
    policy = cfg["feature_groups"]["family_policy"]
    prefer = policy.get("prefer", "best_iv")
    max_per = int(policy.get("max_per_family", 2))

    df = feature_report.copy()
    df["family_size"] = df.groupby("family_base")["feature"].transform("size").astype(int)

    def _rank_block(block):
        if prefer == "best_iv":
            return block.sort_values("iv", ascending=False).reset_index()
        if prefer == "shortest":
            return block.sort_values(["window_rank", "iv"],
                                     ascending=[True, False],
                                     na_position="last").reset_index()
        if prefer == "longest":
            return block.sort_values(["window_rank", "iv"],
                                     ascending=[False, False],
                                     na_position="last").reset_index()
        raise ValueError("Unknown family prefer: {0}".format(prefer))

    df["in_family_rank"] = 0
    df["family_kept"] = True
    for base, block in df.groupby("family_base"):
        if len(block) <= 1:
            df.loc[block.index, "in_family_rank"] = 1
            df.loc[block.index, "family_kept"] = True
            continue
        ranked = _rank_block(block)
        for rank, orig_idx in enumerate(ranked["index"], start=1):
            df.loc[orig_idx, "in_family_rank"] = rank
            df.loc[orig_idx, "family_kept"] = (rank <= max_per)
    return df


def rank_within_semantic_group(feature_report, cfg):
    """Annotate feature_report with group_size, in_group_rank, group_kept.

    Expects feature_report to already contain semantic_group, group_prefer,
    group_max_keep, iv. Features without a semantic_group get group_kept=True.
    """
    df = feature_report.copy()
    df["group_size"] = 0
    df["in_group_rank"] = 0
    df["group_kept"] = True
    grouped = df.dropna(subset=["semantic_group"]).groupby("semantic_group")
    for name, block in grouped:
        df.loc[block.index, "group_size"] = len(block)
        prefer = block["group_prefer"].iloc[0]
        max_keep = int(block["group_max_keep"].iloc[0])
        if prefer == "best_iv":
            ranked = block.sort_values("iv", ascending=False).reset_index()
        elif prefer == "first":
            ranked = block.reset_index()
        else:
            ranked = block.sort_values("iv", ascending=False).reset_index()
        for rank, orig_idx in enumerate(ranked["index"], start=1):
            df.loc[orig_idx, "in_group_rank"] = rank
            df.loc[orig_idx, "group_kept"] = (rank <= max_keep)
    return df


def apply_group_correlation(edges_df, family_df, semantic_df, cfg):
    """Prune correlation edges using three-tier thresholds:
      - same family:    |r| >= corr_cutoff_in_family (default 0.90)
      - same semantic:  |r| >= corr_cutoff_in_group  (default 0.85)
      - elsewhere:      |r| >= corr_cutoff           (default 0.95)

    Input `edges_df` must already be filtered at the LOOSEST threshold
    (global corr_cutoff) — this function only drops edges that survive
    within their family/group threshold.
    """
    if edges_df is None or edges_df.empty:
        return edges_df.copy() if edges_df is not None else pd.DataFrame()

    policy = cfg["feature_groups"]["family_policy"]
    thr_family = float(policy.get("corr_cutoff_in_family", 0.90))
    thr_global = float(cfg["analysis"].get("corr_cutoff", 0.95))

    fam_map = dict(zip(family_df["feature"], family_df["family_base"]))
    sem_map = dict(zip(semantic_df["feature"], semantic_df["semantic_group"]))
    sem_cutoff_map = dict(zip(semantic_df["feature"], semantic_df["group_corr_cutoff"]))

    def _keep(row):
        f1, f2 = row["f1"], row["f2"]
        r = abs(row["r"])
        same_family = (fam_map.get(f1) == fam_map.get(f2)) and (fam_map.get(f1) is not None)
        sg1, sg2 = sem_map.get(f1), sem_map.get(f2)
        same_semantic = (sg1 is not None and sg1 == sg2)
        if same_family:
            return r >= thr_family
        if same_semantic:
            cutoff = sem_cutoff_map.get(f1)
            if cutoff is None or pd.isna(cutoff):
                cutoff = 0.85
            return r >= float(cutoff)
        return r >= thr_global

    keep_mask = edges_df.apply(_keep, axis=1)
    out = edges_df[keep_mask].copy()
    return out


def build_families_summary(feature_report):
    """Aggregate view grouped by family_base, for the Families sheet/CSV."""
    df = feature_report.copy()
    rows = []
    for base, block in df.groupby("family_base"):
        if len(block) <= 1:
            continue  # skip singletons
        windows = [w for w in block["window"].tolist() if w is not None]
        rows.append({
            "family_base": base,
            "window_list": ",".join(windows),
            "iv_best": float(block["iv"].max()) if "iv" in block else np.nan,
            "iv_median": float(block["iv"].median()) if "iv" in block else np.nan,
            "psi_max": float(block["psi"].max()) if "psi" in block else np.nan,
            "kept_count": int(block.get("family_kept", pd.Series([True] * len(block))).sum()),
            "kept_features": ",".join(
                block.loc[block.get("family_kept", True) == True, "feature"].tolist()),
        })
    out = pd.DataFrame(rows, columns=["family_base", "window_list", "iv_best",
                                      "iv_median", "psi_max", "kept_count",
                                      "kept_features"])
    if out.empty:
        return out
    return out.sort_values("iv_best", ascending=False).reset_index(drop=True)


def build_semantic_groups_summary(feature_report, missing_by_group, cfg):
    """Aggregate view for the SemanticGroups sheet/CSV.

    Always returns a DataFrame with the full column schema, even when no
    semantic groups are configured — so the emitted CSV has consistent
    headers across products.
    """
    columns = ["group_name", "description", "member_features", "iv_best",
               "kept_count", "kept_features", "missing_members"]
    groups = (cfg.get("feature_groups") or {}).get("semantic_groups") or []
    rows = []
    for g in groups:
        name = g["name"]
        block = feature_report[feature_report["semantic_group"] == name]
        rows.append({
            "group_name": name,
            "description": g.get("description", ""),
            "member_features": ",".join(block["feature"].tolist()),
            "iv_best": float(block["iv"].max()) if "iv" in block and len(block) else np.nan,
            "kept_count": int(block.get("group_kept",
                                        pd.Series([True] * len(block))).sum()) if len(block) else 0,
            "kept_features": ",".join(
                block.loc[block.get("group_kept", True) == True, "feature"].tolist()),
            "missing_members": ",".join(missing_by_group.get(name, [])),
        })
    return pd.DataFrame(rows, columns=columns)
