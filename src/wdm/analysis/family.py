"""Time-window family + business-semantic group analysis.

Three kinds of correlation to handle differently:
  (A) Time-window families (auto): feat_{7d,30d,90d,all}. Name-pattern detectable.
  (B) Semantic groups (manual):     机构数 / 还款金额 / 申请数 — user declares.
  (C) Plain numerical:              falls back to correlation.py union-find.

This module enriches the feature report with:
  * family_base, window, window_rank, pattern_id, family_size,
    in_family_rank, family_kept
  * semantic_group, group_size, in_group_rank, group_kept

And provides `apply_group_correlation` which tightens the corr cutoff inside
a family (0.90) or semantic group (0.85) while keeping the global cutoff (0.95).

Pattern configuration (feature_groups):
  Two schemas are accepted (pick one):
    (1) window_pattern: "<single regex with (?P<base>...) (?P<window>...)>"
    (2) window_patterns:
          - preset: suffix_day          # reference a named preset, or
          - pattern: "<regex>"          # define inline
            alias:        {raw: canonical}  # optional explicit map
            alias_rule:   "{window}d"       # optional template alternative

  Multi-pattern mode tries each pattern in order; the first one to match
  a feature name wins. Matched `window` tokens are canonicalized via
  `alias` > `alias_rule` > raw, so downstream code only sees canonical
  keys from `window_order` (e.g. "7d", "30d", "1mon", "1y").
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from wdm.utils.window_patterns import WINDOW_PATTERN_PRESETS, resolve_patterns

logger = logging.getLogger(__name__)


def _canonicalize_window(raw_window: str,
                         alias_map: Dict[str, str],
                         alias_rule: Optional[str]) -> str:
    """Normalize a matched raw window token to its canonical key.

    Priority: explicit alias dict > alias_rule template > raw value.
    """
    if raw_window in alias_map:
        return alias_map[raw_window]
    if alias_rule:
        return alias_rule.format(window=raw_window)
    return raw_window


def parse_families(features, cfg):
    """Parse feature list for time-window membership.

    Returns DataFrame[feature, family_base, window, window_rank, pattern_id].

    Supports both single-regex (`feature_groups.window_pattern`) and
    multi-pattern (`feature_groups.window_patterns`) config; see module
    docstring for schema details.

    When feature_groups.enable_window_family is False (e.g. datasets whose
    features are plain numeric IDs with no time-window suffix), every feature
    is treated as its own singleton family — no name-based clustering.
    """
    if not cfg["feature_groups"].get("enable_window_family", True):
        return pd.DataFrame([
            {"feature": f, "family_base": f, "window": None,
             "window_rank": None, "pattern_id": None}
            for f in features
        ])

    patterns = resolve_patterns(cfg)
    order = list(cfg["feature_groups"].get("window_order") or [])
    rank_map = {w: i for i, w in enumerate(order)}

    rows = []
    unknown_windows = set()
    for f in features:
        matched = False
        for regex, alias_map, alias_rule, pid in patterns:
            m = regex.match(f)
            if not m:
                continue
            base = m.group("base")
            raw_w = m.group("window")
            w = _canonicalize_window(raw_w, alias_map, alias_rule)
            if rank_map and w not in rank_map:
                unknown_windows.add(w)
            rows.append({
                "feature": f,
                "family_base": base,
                "window": w,
                "window_rank": int(rank_map.get(w, len(order))),
                "pattern_id": pid,
            })
            matched = True
            break
        if not matched:
            rows.append({
                "feature": f,
                "family_base": f,
                "window": None,
                "window_rank": None,
                "pattern_id": None,
            })

    if unknown_windows:
        logger.warning("Unknown canonical windows (not in window_order): %s "
                       "— they will rank last. Update window_order or the "
                       "alias_rule/alias for the relevant pattern.",
                       sorted(unknown_windows))
    return pd.DataFrame(rows)


def parse_semantic_groups(features, cfg):
    """Return DataFrame[feature, semantic_group, group_prefer, group_max_keep, group_corr_cutoff]

    A group can declare its members via:
      - `features: [list, of, names]`     — explicit list, or
      - `feature_prefix: "bureau_"`       — single prefix match, or
      - `feature_prefix: ["3m_", "6m_"]`  — any-of-list prefix match
    If both `features` and `feature_prefix` are set, `features` is treated as an
    additive explicit list on top of the prefix match.
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
        if isinstance(prefix, str):
            prefixes = [prefix]
        elif isinstance(prefix, (list, tuple)):
            prefixes = [p for p in prefix if isinstance(p, str) and p]
        else:
            prefixes = []
        matched = [f for f in features if any(f.startswith(p) for p in prefixes)] if prefixes else []
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


def effective_family_policy(semantic_group_name: Optional[str],
                            cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the family_policy for a given semantic group name.

    Start from `feature_groups.family_policy` (the global default) and layer
    the group's own `family_policy` on top. Missing keys fall through. When
    `semantic_group_name` is None or the group has no override, the global
    default is returned verbatim.

    Example config:
        feature_groups:
          family_policy:
            prefer: best_iv_short_bias
            window_penalty_gamma: 0.15
          semantic_groups:
            - name: bureau
              feature_prefix: "bureau_"
              family_policy:
                prefer: best_iv          # bureau keeps long-window features
                window_penalty_gamma: 0.0
    """
    default = dict((cfg.get("feature_groups") or {}).get("family_policy") or {})
    if not semantic_group_name:
        return default
    groups = (cfg.get("feature_groups") or {}).get("semantic_groups") or []
    for g in groups:
        if g.get("name") != semantic_group_name:
            continue
        override = g.get("family_policy")
        if not override:
            return default
        merged = dict(default)
        merged.update(override)
        return merged
    return default


def _family_anchor_group(block: pd.DataFrame) -> Optional[str]:
    """Return the semantic_group a family is anchored to, or None if mixed / unset."""
    if "semantic_group" not in block.columns:
        return None
    vals = block["semantic_group"].dropna().unique().tolist()
    if len(vals) == 1:
        return str(vals[0])
    return None


def rank_within_family(feature_report, cfg):
    """Annotate feature_report with family_size, in_family_rank, family_kept.

    Expects feature_report to already contain family_base, window, window_rank, iv.
    When the report has a `semantic_group` column and a group declares its own
    `family_policy`, that override takes effect for families anchored to it.
    """
    df = feature_report.copy()
    df["family_size"] = df.groupby("family_base")["feature"].transform("size").astype(int)

    def _rank_block(block, prefer, tol):
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
        if prefer == "best_iv_short_bias":
            # Within IV tolerance of the family's best, prefer the shortest window;
            # outside tolerance, fall back to pure IV ordering. This counteracts
            # the coverage advantage long windows have on families whose short
            # siblings carry comparable predictive signal.
            iv_best = float(block["iv"].max())
            mask = block["iv"] >= (iv_best - tol)
            head = block[mask].sort_values(["window_rank", "iv"],
                                           ascending=[True, False],
                                           na_position="last").reset_index()
            tail = block[~mask].sort_values("iv", ascending=False).reset_index()
            return pd.concat([head, tail], ignore_index=True)
        raise ValueError("Unknown family prefer: {0}".format(prefer))

    df["in_family_rank"] = 0
    df["family_kept"] = True
    for base, block in df.groupby("family_base"):
        if len(block) <= 1:
            df.loc[block.index, "in_family_rank"] = 1
            df.loc[block.index, "family_kept"] = True
            continue
        anchor = _family_anchor_group(block)
        policy = effective_family_policy(anchor, cfg)
        prefer = policy.get("prefer", "best_iv")
        max_per = int(policy.get("max_per_family", 2))
        tol = float(policy.get("coverage_bias_iv_tolerance", 0.02))
        ranked = _rank_block(block, prefer, tol)
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


def _family_iv_stats(block: pd.DataFrame) -> Dict[str, Any]:
    """Compute shortest/longest window IV stats for a family block.

    Returns dict with keys: iv_best, iv_best_window, iv_shortest, iv_shortest_window,
    iv_longest, iv_longest_window, iv_ratio_long_over_short, iv_delta_best_vs_short,
    missing_rate_shortest. NaN-safe.
    """
    if "iv" not in block.columns or block.empty:
        return {}
    ranked = block.dropna(subset=["window_rank"]).copy()
    iv_best_idx = block["iv"].idxmax() if block["iv"].notna().any() else None
    out: Dict[str, Any] = {
        "iv_best": float(block["iv"].max()) if iv_best_idx is not None else np.nan,
        "iv_best_window": block.loc[iv_best_idx, "window"] if iv_best_idx is not None else None,
    }
    if ranked.empty:
        out.update({
            "iv_shortest": np.nan, "iv_shortest_window": None,
            "iv_longest": np.nan, "iv_longest_window": None,
            "iv_ratio_long_over_short": np.nan,
            "iv_delta_best_vs_short": np.nan,
            "missing_rate_shortest": np.nan,
        })
        return out
    short_row = ranked.loc[ranked["window_rank"].idxmin()]
    long_row = ranked.loc[ranked["window_rank"].idxmax()]
    iv_short = float(short_row["iv"]) if pd.notna(short_row["iv"]) else np.nan
    iv_long = float(long_row["iv"]) if pd.notna(long_row["iv"]) else np.nan
    ratio = (iv_long / iv_short) if (iv_short and iv_short > 0) else np.nan
    delta = (out["iv_best"] - iv_short) if pd.notna(iv_short) else np.nan
    out.update({
        "iv_shortest": iv_short,
        "iv_shortest_window": short_row["window"],
        "iv_longest": iv_long,
        "iv_longest_window": long_row["window"],
        "iv_ratio_long_over_short": ratio,
        "iv_delta_best_vs_short": delta,
        "missing_rate_shortest": (float(short_row["missing_rate"])
                                  if "missing_rate" in short_row.index
                                  and pd.notna(short_row["missing_rate"]) else np.nan),
    })
    return out


def build_families_summary(feature_report, cfg=None):
    """Aggregate view grouped by family_base, for the Families sheet/CSV.

    When `cfg` is provided, also includes two window-bias diagnostics
    consumed by notebooks/06_window_family_analysis:
    `iv_ratio_long_over_short`, `iv_delta_best_vs_short`.
    """
    df = feature_report.copy()
    rows = []
    for base, block in df.groupby("family_base"):
        if len(block) <= 1:
            continue  # skip singletons
        windows = [w for w in block["window"].tolist() if w is not None]
        stats = _family_iv_stats(block) if cfg is not None else {}
        row = {
            "family_base": base,
            "window_list": ",".join(windows),
            "iv_best": float(block["iv"].max()) if "iv" in block else np.nan,
            "iv_median": float(block["iv"].median()) if "iv" in block else np.nan,
            "psi_max": float(block["psi"].max()) if "psi" in block else np.nan,
            "kept_count": int(block.get("family_kept", pd.Series([True] * len(block))).sum()),
            "kept_features": ",".join(
                block.loc[block.get("family_kept", True) == True, "feature"].tolist()),
        }
        if cfg is not None:
            row.update({
                "iv_ratio_long_over_short": stats.get("iv_ratio_long_over_short", np.nan),
                "iv_delta_best_vs_short": stats.get("iv_delta_best_vs_short", np.nan),
            })
        rows.append(row)
    columns = ["family_base", "window_list", "iv_best", "iv_median", "psi_max",
               "kept_count", "kept_features"]
    if cfg is not None:
        columns += ["iv_ratio_long_over_short", "iv_delta_best_vs_short"]
    out = pd.DataFrame(rows, columns=columns)
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
