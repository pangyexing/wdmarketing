"""Tests for discover_derivation_candidates (Part B).

Given a synthesized Stage-1 feature report, assert that the advice CSV:
  * skips singleton families,
  * flags delta/ratio when long window's IV is much bigger than short's,
  * adds incremental when 3+ windows present,
  * suggests keep_short_only when short IV is within 20% of best.
"""
import numpy as np
import pandas as pd

from wdm.analysis.family import discover_derivation_candidates


CFG = {
    "feature_groups": {
        "window_order": ["7d", "30d", "90d", "all"],
        "family_policy": {},
    },
    "feature_derivations": {"enabled": False},
}


def _row(feature, base, window, rank, iv, miss=0.1):
    return {"feature": feature, "family_base": base, "window": window,
            "window_rank": rank, "iv": iv, "missing_rate": miss}


def test_singleton_family_is_excluded():
    rep = pd.DataFrame([_row("loner_7d", "loner", "7d", 0, 0.15)])
    out = discover_derivation_candidates(rep, CFG)
    assert out.empty


def test_delta_ratio_suggested_when_long_dominates():
    rep = pd.DataFrame([
        _row("amt_7d",  "amt", "7d",  0, 0.05),
        _row("amt_all", "amt", "all", 3, 0.20),  # ratio 0.20/0.05 = 4x > 1.5
    ])
    out = discover_derivation_candidates(rep, CFG)
    row = out.set_index("family_base").loc["amt"]
    ops = row["suggested_ops"].split(",") if row["suggested_ops"] else []
    assert "delta" in ops and "ratio" in ops
    assert row["iv_shortest_window"] == "7d"
    assert row["iv_best_window"] == "all"


def test_incremental_suggested_when_three_plus_windows():
    rep = pd.DataFrame([
        _row("cnt_7d",  "cnt", "7d",  0, 0.04),
        _row("cnt_30d", "cnt", "30d", 1, 0.10),
        _row("cnt_all", "cnt", "all", 3, 0.30),  # ratio 7.5x > 1.5 AND 3+ windows
    ])
    out = discover_derivation_candidates(rep, CFG)
    ops = out.set_index("family_base").loc["cnt", "suggested_ops"].split(",")
    assert "incremental" in ops


def test_keep_short_only_when_short_iv_within_20pct_of_best():
    rep = pd.DataFrame([
        _row("stab_7d",  "stab", "7d",  0, 0.18),   # 0.18 >= 0.8 * 0.19
        _row("stab_all", "stab", "all", 3, 0.19),   # ratio ~1.06 (< 1.5)
    ])
    out = discover_derivation_candidates(rep, CFG)
    ops = out.set_index("family_base").loc["stab", "suggested_ops"].split(",")
    assert "keep_short_only" in ops
    # ratio is tiny so delta/ratio should NOT be suggested
    assert "delta" not in ops
    assert "ratio" not in ops


def test_already_configured_flag_reflects_feature_derivations_config():
    cfg = {
        "feature_groups": CFG["feature_groups"],
        "feature_derivations": {
            "enabled": True,
            "families": [{"family_base": "amt", "ops": []}],
        },
    }
    rep = pd.DataFrame([
        _row("amt_7d",  "amt", "7d",  0, 0.05),
        _row("amt_all", "amt", "all", 3, 0.20),
        _row("other_7d",  "other", "7d",  0, 0.05),
        _row("other_all", "other", "all", 3, 0.25),
    ])
    out = discover_derivation_candidates(rep, cfg).set_index("family_base")
    assert bool(out.loc["amt", "already_configured"]) is True
    assert bool(out.loc["other", "already_configured"]) is False
