"""Tests for the PSI selection knobs (psi_mode / psi_penalty_weight).

Covers the three modes — soft (default), hard (legacy drop), off — and
verifies that psi_over_cutoff is always populated for reporting regardless
of mode.
"""
import pandas as pd
import pytest

from wdm.analysis.selector import (
    apply_hard_filters, rank_and_auto_keep, _resolve_psi_knobs,
)


def _toy_df():
    """Minimal per-feature frame with the columns apply_hard_filters /
    rank_and_auto_keep actually touch. PSI spans below and above the
    default cutoff (0.25) so we can distinguish modes.
    """
    return pd.DataFrame([
        # feature, iv,   psi,   missing_rate, n_unique, lift_at_k, gini, window
        ("low_psi_good",   0.10, 0.05, 0.10,  50, 3.0, 0.4, "all"),
        ("high_psi_good",  0.10, 0.40, 0.10,  50, 3.0, 0.4, "all"),
        ("low_psi_weak",   0.001, 0.05, 0.10, 50, 1.1, 0.02, "all"),
    ], columns=["feature", "iv", "psi", "missing_rate", "n_unique",
                "lift_at_k", "gini", "window"])


def _cfg(**overrides):
    base = {
        "analysis": {
            "missing_rate_max": 0.95,
            "iv_min": 0.005,
            "psi_cutoff": 0.25,
            "psi_mode": "soft",
            "psi_penalty_weight": 0.25,
        },
        "feature_groups": {"family_policy": {}, "semantic_groups": []},
    }
    for k, v in overrides.items():
        base["analysis"][k] = v
    return base


# --- _resolve_psi_knobs -----------------------------------------------------

def test_resolve_defaults_are_soft_and_025():
    mode, w = _resolve_psi_knobs({"analysis": {}})
    assert mode == "soft"
    assert w == pytest.approx(0.25)


def test_resolve_invalid_mode_falls_back_to_soft():
    mode, _ = _resolve_psi_knobs({"analysis": {"psi_mode": "garbage"}})
    assert mode == "soft"


def test_resolve_honors_explicit_hard_and_off():
    assert _resolve_psi_knobs({"analysis": {"psi_mode": "hard"}})[0] == "hard"
    assert _resolve_psi_knobs({"analysis": {"psi_mode": "off"}})[0] == "off"


# --- apply_hard_filters ----------------------------------------------------

def test_soft_mode_does_not_drop_high_psi():
    df = apply_hard_filters(_toy_df(), _cfg(psi_mode="soft"))
    high = df[df["feature"] == "high_psi_good"].iloc[0]
    assert bool(high["_hard_drop"]) is False
    assert "high_psi" not in high["_hard_drop_reason"]
    # reporting column always reflects the raw threshold crossing
    assert bool(high["psi_over_cutoff"]) is True


def test_hard_mode_drops_high_psi():
    df = apply_hard_filters(_toy_df(), _cfg(psi_mode="hard"))
    high = df[df["feature"] == "high_psi_good"].iloc[0]
    assert "high_psi" in high["_hard_drop_reason"]
    # low-psi feature is untouched
    low = df[df["feature"] == "low_psi_good"].iloc[0]
    assert "high_psi" not in low["_hard_drop_reason"]


def test_off_mode_matches_soft_for_hard_filters():
    """'off' differs from 'soft' only in the rank_score term; hard-filter
    behavior is identical."""
    df_soft = apply_hard_filters(_toy_df(), _cfg(psi_mode="soft"))
    df_off = apply_hard_filters(_toy_df(), _cfg(psi_mode="off"))
    assert list(df_soft["_hard_drop_reason"]) == list(df_off["_hard_drop_reason"])


# --- rank_and_auto_keep (rank_score term) ----------------------------------

def test_off_mode_zeros_psi_contribution_to_rank_score():
    base = apply_hard_filters(_toy_df(), _cfg(psi_mode="off"))
    base["corr_cluster"] = -1
    base["family_kept"] = True
    base["group_kept"] = True
    ranked = rank_and_auto_keep(base, _cfg(psi_mode="off"))
    # With PSI zeroed out, high-PSI and low-PSI features with identical
    # iv/lift/gini/missing should have the same rank_score.
    r_low = ranked[ranked["feature"] == "low_psi_good"].iloc[0]["rank_score"]
    r_high = ranked[ranked["feature"] == "high_psi_good"].iloc[0]["rank_score"]
    assert r_low == pytest.approx(r_high)


def test_soft_mode_penalizes_high_psi_in_rank_score():
    base = apply_hard_filters(_toy_df(), _cfg(psi_mode="soft"))
    base["corr_cluster"] = -1
    base["family_kept"] = True
    base["group_kept"] = True
    ranked = rank_and_auto_keep(base, _cfg(psi_mode="soft"))
    r_low = ranked[ranked["feature"] == "low_psi_good"].iloc[0]["rank_score"]
    r_high = ranked[ranked["feature"] == "high_psi_good"].iloc[0]["rank_score"]
    # Soft mode: high-PSI gets a smaller rank_score (penalty active)
    assert r_high < r_low


def test_penalty_weight_zero_collapses_soft_into_off():
    """psi_penalty_weight=0 with psi_mode=soft should behave like 'off'
    w.r.t. rank_score (no penalty applied)."""
    base = apply_hard_filters(_toy_df(), _cfg(psi_mode="soft"))
    base["corr_cluster"] = -1
    base["family_kept"] = True
    base["group_kept"] = True
    ranked = rank_and_auto_keep(
        base, _cfg(psi_mode="soft", psi_penalty_weight=0.0))
    r_low = ranked[ranked["feature"] == "low_psi_good"].iloc[0]["rank_score"]
    r_high = ranked[ranked["feature"] == "high_psi_good"].iloc[0]["rank_score"]
    assert r_low == pytest.approx(r_high)
