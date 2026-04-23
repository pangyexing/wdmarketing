"""Tests for window-pattern resolution (Part 0) and ranking bias fixes (Part A).

Part 0 — window_pattern can be a single regex or a list of patterns with
alias normalization; supports day/month/year canonical keys.
Part A — `best_iv_short_bias` prefer mode and `window_penalty_gamma` bias
away from long-window coverage advantage in `rank_score`.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from wdm.analysis.family import (
    _WINDOW_PATTERN_PRESETS,
    _resolve_patterns,
    effective_family_policy,
    parse_families,
    rank_within_family,
)
from wdm.analysis.selector import _rank_and_auto_keep


# ---------- Part 0 fixtures ----------

def _cfg_with_patterns(patterns, window_order):
    return {
        "feature_groups": {
            "window_patterns": patterns,
            "window_order": list(window_order),
            "family_policy": {"max_per_family": 2, "prefer": "best_iv"},
        }
    }


def _cfg_single_pattern(pattern, window_order):
    return {
        "feature_groups": {
            "window_pattern": pattern,
            "window_order": list(window_order),
            "family_policy": {"max_per_family": 2, "prefer": "best_iv"},
        }
    }


# ---------- Part 0 — pattern resolution ----------

def test_multi_pattern_first_match_wins():
    cfg = _cfg_with_patterns(
        patterns=[
            {"preset": "suffix_day"},
            {"preset": "prefix_d"},
            {"preset": "chinese_jin_days"},
        ],
        window_order=["7d", "30d", "90d"],
    )
    features = ["foo_7d", "bar_d7", "baz_近7天"]
    out = parse_families(features, cfg)
    assert set(out["window"].tolist()) == {"7d"}
    assert out.set_index("feature").loc["foo_7d", "family_base"] == "foo"
    assert out.set_index("feature").loc["bar_d7", "family_base"] == "bar"
    assert out.set_index("feature").loc["baz_近7天", "family_base"] == "baz"
    # window_rank is consistent across patterns (all → 0, shortest in order)
    assert out["window_rank"].unique().tolist() == [0]


def test_preset_suffix_day_backward_compatible():
    legacy = _cfg_single_pattern(
        _WINDOW_PATTERN_PRESETS["suffix_day"]["pattern"],
        window_order=["7d", "30d", "all"],
    )
    new = _cfg_with_patterns([{"preset": "suffix_day"}], window_order=["7d", "30d", "all"])
    features = ["foo_7d", "foo_30d", "foo_all", "nomatch"]
    legacy_df = parse_families(features, legacy).drop(columns=["pattern_id"])
    new_df = parse_families(features, new).drop(columns=["pattern_id"])
    # Legacy carries pattern_id "legacy_window_pattern"; new carries "suffix_day";
    # but family_base/window/window_rank must match.
    pd.testing.assert_frame_equal(
        legacy_df.sort_values("feature").reset_index(drop=True),
        new_df.sort_values("feature").reset_index(drop=True),
    )


def test_unknown_canonical_window_warns_and_tails(caplog):
    cfg = _cfg_with_patterns(
        patterns=[{"preset": "suffix_day"}],
        window_order=["7d", "30d"],  # 'all' is valid by regex but absent from order
    )
    with caplog.at_level(logging.WARNING, logger="wdm.analysis.family"):
        out = parse_families(["foo_all"], cfg)
    assert out.loc[0, "window"] == "all"
    # Missing from window_order → ranked last (index == len(order))
    assert int(out.loc[0, "window_rank"]) == 2
    assert any("Unknown canonical windows" in rec.message for rec in caplog.records)


def test_config_validation_rejects_missing_named_group():
    cfg = _cfg_with_patterns(
        patterns=[{"pattern": r"^(.+?)_(7d|30d)$"}],  # no (?P<base>) / (?P<window>)
        window_order=["7d", "30d"],
    )
    with pytest.raises(ValueError, match=r"named groups"):
        _resolve_patterns(cfg)


def test_suffix_mon_preset_recognizes_month_windows():
    cfg = _cfg_with_patterns(
        patterns=[{"preset": "suffix_mon"}],
        window_order=["1mon", "3mon", "6mon", "12mon"],
    )
    out = parse_families(["foo_1mon", "foo_6mon", "foo_12mon"], cfg)
    assert out["family_base"].unique().tolist() == ["foo"]
    assert set(out["window"].tolist()) == {"1mon", "6mon", "12mon"}
    # window_rank matches position in window_order
    ranks = dict(zip(out["window"], out["window_rank"]))
    assert ranks == {"1mon": 0, "6mon": 2, "12mon": 3}


def test_suffix_mon_to_days_alias_maps_to_day_keys():
    cfg = _cfg_with_patterns(
        patterns=[{"preset": "suffix_day"}, {"preset": "suffix_mon_to_days"}],
        window_order=["7d", "30d", "90d", "180d", "360d"],
    )
    out = parse_families(["foo_6mon", "foo_180d"], cfg).set_index("feature")
    # 6mon → 180d via alias; identical to foo_180d canonicalization
    assert out.loc["foo_6mon", "window"] == "180d"
    assert out.loc["foo_180d", "window"] == "180d"
    assert out.loc["foo_6mon", "family_base"] == "foo"
    assert out.loc["foo_180d", "family_base"] == "foo"
    assert out.loc["foo_6mon", "window_rank"] == out.loc["foo_180d", "window_rank"]


def test_suffix_year_preset_recognizes_year_windows():
    cfg = _cfg_with_patterns(
        patterns=[{"preset": "suffix_year"}],
        window_order=["1y", "2y", "5y"],
    )
    out = parse_families(["bar_1y", "bar_2y", "bar_5y"], cfg)
    assert out["family_base"].unique().tolist() == ["bar"]
    assert sorted(out["window"].tolist()) == ["1y", "2y", "5y"]


def test_mixed_day_mon_year_to_days():
    cfg = _cfg_with_patterns(
        patterns=[
            {"preset": "suffix_day"},
            {"preset": "suffix_mon_to_days"},
            {"preset": "suffix_year_to_days"},
        ],
        window_order=["7d", "30d", "90d", "180d", "360d", "720d"],
    )
    out = parse_families(["baz_7d", "baz_6mon", "baz_2y"], cfg).set_index("feature")
    assert out.loc["baz_7d", "window"] == "7d"
    assert out.loc["baz_6mon", "window"] == "180d"
    assert out.loc["baz_2y", "window"] == "720d"
    assert out["family_base"].unique().tolist() == ["baz"]
    # Strictly increasing window_rank (7d=0, 180d=3, 720d=5)
    ranks = [out.loc[f, "window_rank"] for f in ["baz_7d", "baz_6mon", "baz_2y"]]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)


# ---------- Part A — ranking bias fixes ----------

def _policy(**kwargs):
    base = {"max_per_family": 2, "prefer": "best_iv",
            "coverage_bias_iv_tolerance": 0.02, "window_penalty_gamma": 0.0}
    base.update(kwargs)
    return {
        "feature_groups": {
            "window_pattern": _WINDOW_PATTERN_PRESETS["suffix_day"]["pattern"],
            "window_order": ["7d", "30d", "90d", "all"],
            "family_policy": base,
        },
        "analysis": {},
    }


def test_best_iv_short_bias_picks_shorter_within_tolerance():
    cfg = _policy(prefer="best_iv_short_bias", coverage_bias_iv_tolerance=0.02)
    rep = pd.DataFrame([
        {"feature": "foo_7d", "family_base": "foo", "window": "7d",
         "window_rank": 0, "iv": 0.18},
        {"feature": "foo_all", "family_base": "foo", "window": "all",
         "window_rank": 3, "iv": 0.19},
    ])
    out = rank_within_family(rep, cfg).set_index("feature")
    assert out.loc["foo_7d", "in_family_rank"] == 1
    assert out.loc["foo_all", "in_family_rank"] == 2


def test_best_iv_short_bias_falls_back_when_gap_large():
    cfg = _policy(prefer="best_iv_short_bias", coverage_bias_iv_tolerance=0.02)
    rep = pd.DataFrame([
        {"feature": "foo_7d", "family_base": "foo", "window": "7d",
         "window_rank": 0, "iv": 0.10},
        {"feature": "foo_all", "family_base": "foo", "window": "all",
         "window_rank": 3, "iv": 0.25},
    ])
    out = rank_within_family(rep, cfg).set_index("feature")
    assert out.loc["foo_all", "in_family_rank"] == 1
    assert out.loc["foo_7d", "in_family_rank"] == 2


def _rank_df(missing_rate=None):
    df = pd.DataFrame([
        {"feature": "foo_7d", "window": "7d", "iv": 0.20, "lift_at_k": 1.5,
         "gini": 0.20, "psi": 0.05, "missing_rate": 0.1,
         "family_kept": True, "group_kept": True, "corr_cluster": -1,
         "_hard_drop_reason": ""},
        {"feature": "foo_all", "window": "all", "iv": 0.20, "lift_at_k": 1.5,
         "gini": 0.20, "psi": 0.05, "missing_rate": 0.1,
         "family_kept": True, "group_kept": True, "corr_cluster": -1,
         "_hard_drop_reason": ""},
    ])
    if missing_rate is not None:
        df["missing_rate"] = missing_rate
    return df


def test_window_penalty_zero_preserves_legacy():
    cfg = _policy(window_penalty_gamma=0.0)
    df = _rank_df()
    out = _rank_and_auto_keep(df.copy(), cfg)
    # With equal signals, rank_score should be equal (no window bias applied)
    rs = out.set_index("feature")["rank_score"]
    assert abs(rs["foo_7d"] - rs["foo_all"]) < 1e-9


def test_window_penalty_demotes_long_when_equal():
    cfg = _policy(
        window_penalty_gamma=0.5,
        window_penalty_table={"7d": 0.0, "30d": 0.1, "90d": 0.2, "all": 0.4},
    )
    df = _rank_df()
    out = _rank_and_auto_keep(df.copy(), cfg)
    rs = out.set_index("feature")["rank_score"]
    # 7d penalty 0.0, all penalty 0.4 → 7d strictly higher rank_score
    assert rs["foo_7d"] > rs["foo_all"]
    assert abs((rs["foo_7d"] - rs["foo_all"]) - 0.5 * 0.4) < 1e-9


# ---------- Stage 3 — semantic-group-level family_policy override ----------

def test_effective_family_policy_merges_group_override():
    cfg = {
        "feature_groups": {
            "family_policy": {
                "prefer": "best_iv_short_bias",
                "window_penalty_gamma": 0.15,
                "max_per_family": 2,
            },
            "semantic_groups": [
                {"name": "bureau", "family_policy": {"prefer": "best_iv",
                                                     "window_penalty_gamma": 0.0}},
                {"name": "cc"},  # no override — inherits global
            ],
        }
    }
    bureau = effective_family_policy("bureau", cfg)
    cc = effective_family_policy("cc", cfg)
    none_group = effective_family_policy(None, cfg)
    assert bureau["prefer"] == "best_iv"
    assert bureau["window_penalty_gamma"] == 0.0
    assert bureau["max_per_family"] == 2  # fell through from global
    assert cc["prefer"] == "best_iv_short_bias"
    assert cc["window_penalty_gamma"] == 0.15
    assert none_group["prefer"] == "best_iv_short_bias"


def test_rank_within_family_uses_group_override_for_anchored_family():
    cfg = {
        "feature_groups": {
            "family_policy": {"prefer": "best_iv_short_bias",
                              "coverage_bias_iv_tolerance": 0.02,
                              "max_per_family": 2},
            "semantic_groups": [
                {"name": "bureau", "family_policy": {"prefer": "best_iv"}},
            ],
        }
    }
    # bureau-anchored family: global says short_bias (would pick 7d),
    # but bureau override says best_iv → picks `all` (higher IV).
    rep = pd.DataFrame([
        {"feature": "bureau_amt_7d",  "family_base": "bureau_amt",
         "window": "7d",  "window_rank": 0, "iv": 0.18,
         "semantic_group": "bureau"},
        {"feature": "bureau_amt_all", "family_base": "bureau_amt",
         "window": "all", "window_rank": 3, "iv": 0.19,
         "semantic_group": "bureau"},
        # cc-anchored family: no override → uses global short_bias (picks 7d)
        {"feature": "cc_bal_7d",  "family_base": "cc_bal",
         "window": "7d",  "window_rank": 0, "iv": 0.18,
         "semantic_group": "cc"},
        {"feature": "cc_bal_all", "family_base": "cc_bal",
         "window": "all", "window_rank": 3, "iv": 0.19,
         "semantic_group": "cc"},
    ])
    out = rank_within_family(rep, cfg).set_index("feature")
    # bureau → best_iv picks highest IV → all is rank 1
    assert out.loc["bureau_amt_all", "in_family_rank"] == 1
    assert out.loc["bureau_amt_7d", "in_family_rank"] == 2
    # cc → short_bias → 7d within tol wins
    assert out.loc["cc_bal_7d", "in_family_rank"] == 1
    assert out.loc["cc_bal_all", "in_family_rank"] == 2


def test_row_penalty_respects_per_group_gamma():
    """Two identical rows, different semantic groups, different effective gamma
    → different rank_score."""
    cfg = {
        "feature_groups": {
            "window_pattern": _WINDOW_PATTERN_PRESETS["suffix_day"]["pattern"],
            "window_order": ["7d", "30d", "90d", "all"],
            "family_policy": {
                "window_penalty_gamma": 0.5,
                "window_penalty_table": {"7d": 0.0, "30d": 0.1, "90d": 0.2, "all": 0.4},
            },
            "semantic_groups": [
                {"name": "bureau", "family_policy": {"window_penalty_gamma": 0.0}},
            ],
        },
        "analysis": {},
    }
    df = pd.DataFrame([
        {"feature": "bureau_amt_all", "window": "all", "iv": 0.2, "lift_at_k": 1.5,
         "gini": 0.2, "psi": 0.05, "missing_rate": 0.1,
         "family_kept": True, "group_kept": True, "corr_cluster": -1,
         "semantic_group": "bureau", "_hard_drop_reason": ""},
        {"feature": "cc_bal_all", "window": "all", "iv": 0.2, "lift_at_k": 1.5,
         "gini": 0.2, "psi": 0.05, "missing_rate": 0.1,
         "family_kept": True, "group_kept": True, "corr_cluster": -1,
         "semantic_group": "cc", "_hard_drop_reason": ""},
    ])
    out = _rank_and_auto_keep(df.copy(), cfg)
    rs = out.set_index("feature")["rank_score"]
    # bureau gamma=0 → zero penalty contribution; cc uses global 0.5 → −0.5*0.4 = −0.2
    # Both have identical z-scored signals (zero since equal), so diff is exactly 0.2
    assert abs((rs["bureau_amt_all"] - rs["cc_bal_all"]) - 0.5 * 0.4) < 1e-9
