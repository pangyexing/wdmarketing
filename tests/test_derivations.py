"""Tests for src/wdm/feature_engineering/derivations.py (阶段 2 foundation).

Covers operator correctness (delta / ratio / incremental / velocity),
NaN-policy branches (ratio zero-denominator, both-sides-nan, one-side-nan),
keep_original semantics, recipe round-trip determinism, and plan-loading
validation errors.

NOT in scope: wiring into `build_dataset` or `predict_template.py`. Those
arrive in a follow-up PR once these semantics are locked down.
"""
import math

import numpy as np
import pandas as pd
import pytest

from wdm.feature_engineering.derivations import (
    DerivationPlan,
    apply_derivations,
    apply_recipe,
    dumps,
    load_derivation_plan,
    to_recipe_json,
)


def _cfg(**kwargs):
    fd = {"enabled": True, "families": []}
    fd.update(kwargs)
    return {"feature_derivations": fd}


# ---------- Plan loading ----------

def test_plan_disabled_returns_none():
    assert load_derivation_plan({"feature_derivations": {"enabled": False}}) is None
    assert load_derivation_plan({}) is None


def test_plan_loads_simple_delta():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "delta", "left": "7d", "right": "180d",
                 "output": "amt_delta_7d_vs_180d"}],
    }]))
    assert plan.enabled
    assert len(plan.families) == 1
    assert plan.families[0].ops[0].inputs == ("amt_7d", "amt_180d")


def test_plan_rejects_unknown_op():
    with pytest.raises(ValueError, match="op"):
        load_derivation_plan(_cfg(families=[{
            "family_base": "amt",
            "ops": [{"op": "frobnicate", "left": "7d", "right": "all", "output": "x"}],
        }]))


def test_plan_rejects_missing_key():
    with pytest.raises(ValueError, match="missing required key"):
        load_derivation_plan(_cfg(families=[{
            "family_base": "amt",
            "ops": [{"op": "delta", "left": "7d"}],  # missing right/output
        }]))


def test_plan_rejects_bad_keep_original():
    with pytest.raises(ValueError, match="keep_original"):
        load_derivation_plan(_cfg(families=[{
            "family_base": "amt",
            "keep_original": "maybe",
            "ops": [{"op": "delta", "left": "7d", "right": "all", "output": "x"}],
        }]))


def test_velocity_requires_known_day_windows():
    with pytest.raises(ValueError, match="velocity"):
        load_derivation_plan(_cfg(families=[{
            "family_base": "amt",
            "ops": [{"op": "velocity", "short": "7d", "long": "all",  # all → None days
                     "output": "amt_velocity"}],
        }]))


def test_velocity_long_must_be_greater_than_short():
    with pytest.raises(ValueError, match="long.*more days"):
        load_derivation_plan(_cfg(families=[{
            "family_base": "amt",
            "ops": [{"op": "velocity", "short": "180d", "long": "7d",
                     "output": "amt_velocity"}],
        }]))


# ---------- Operator correctness ----------

def _df():
    return pd.DataFrame({
        "amt_7d":   [10.0, 20.0, np.nan,  0.0,  5.0],
        "amt_30d":  [30.0, 50.0,  2.0,   0.0,  8.0],
        "amt_180d": [60.0, 80.0,  5.0,   0.0,  np.nan],
        "amt_all":  [90.0, 100.0, 15.0, 20.0, 12.0],
    })


def test_delta_basic():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "delta", "left": "all", "right": "7d",
                 "output": "amt_delta"}],
    }]))
    out, added, _ = apply_derivations(_df(), plan)
    assert added == ["amt_delta"]
    # [90-10, 100-20, 15-NaN=NaN, 20-0, 12-5]
    got = out["amt_delta"].tolist()
    assert got[0] == 80.0 and got[1] == 80.0 and math.isnan(got[2]) and got[3] == 20.0 and got[4] == 7.0


def test_ratio_zero_denominator_policies():
    base = _df()
    # Policy: nan
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "ratio", "numerator": "all", "denominator": "7d",
                 "output": "r", "nan_policy": {"ratio_zero_denominator": "nan"}}],
    }]))
    out, _, _ = apply_derivations(base, plan)
    # row 3: 20/0 → nan under 'nan' policy
    assert math.isnan(out["r"].iloc[3])
    # Other non-NaN rows: 90/10=9, 100/20=5, 12/5=2.4
    assert out["r"].iloc[0] == 9.0
    assert out["r"].iloc[1] == 5.0
    assert abs(out["r"].iloc[4] - 2.4) < 1e-9

    # Policy: zero
    plan2 = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "ratio", "numerator": "all", "denominator": "7d",
                 "output": "r", "nan_policy": {"ratio_zero_denominator": "zero"}}],
    }]))
    out2, _, _ = apply_derivations(base, plan2)
    assert out2["r"].iloc[3] == 0.0

    # Policy: inf_clipped
    plan3 = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "ratio", "numerator": "all", "denominator": "7d",
                 "output": "r",
                 "nan_policy": {"ratio_zero_denominator": "inf_clipped",
                                "ratio_clip": 1000.0}}],
    }]))
    out3, _, _ = apply_derivations(base, plan3)
    assert out3["r"].iloc[3] == 1000.0  # 20 / 0 → +1000


def test_one_side_nan_fill_with_zero():
    """amt_180d[4] = NaN; with fill_with_zero_then_op + delta vs amt_all: result = 12 - 0 = 12."""
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "delta", "left": "all", "right": "180d", "output": "d",
                 "nan_policy": {"one_side_nan": "fill_with_zero_then_op"}}],
    }]))
    out, _, _ = apply_derivations(_df(), plan)
    # row 4: all=12, 180d=NaN → filled to 0 → 12-0=12
    assert out["d"].iloc[4] == 12.0


def test_incremental_emits_chain_deltas():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "incremental", "chain": ["7d", "30d", "180d", "all"],
                 "output_prefix": "amt_incr"}],
    }]))
    out, added, _ = apply_derivations(_df(), plan)
    # Three deltas: 30d-7d, 180d-30d, all-180d
    assert set(added) == {"amt_incr_30d_minus_7d", "amt_incr_180d_minus_30d",
                          "amt_incr_all_minus_180d"}
    # Row 0 sanity
    assert out["amt_incr_30d_minus_7d"].iloc[0] == 20.0
    assert out["amt_incr_180d_minus_30d"].iloc[0] == 30.0
    assert out["amt_incr_all_minus_180d"].iloc[0] == 30.0


def test_velocity_per_day_rate():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "velocity", "short": "7d", "long": "180d",
                 "output": "amt_velocity"}],
    }]))
    out, _, _ = apply_derivations(_df(), plan)
    # row 0: (60 - 10) / (180 - 7) = 50/173 ≈ 0.289
    assert abs(out["amt_velocity"].iloc[0] - (50.0 / 173.0)) < 1e-9


# ---------- keep_original semantics ----------

def test_keep_original_both_keeps_inputs():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt", "keep_original": "both",
        "ops": [{"op": "delta", "left": "all", "right": "7d", "output": "d"}],
    }]))
    out, added, dropped = apply_derivations(_df(), plan)
    assert added == ["d"]
    assert dropped == []
    assert "amt_7d" in out.columns and "amt_all" in out.columns


def test_keep_original_replace_drops_used_inputs():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt", "keep_original": "replace",
        "ops": [{"op": "delta", "left": "all", "right": "7d", "output": "d"}],
    }]))
    out, added, dropped = apply_derivations(_df(), plan)
    assert sorted(dropped) == ["amt_7d", "amt_all"]
    assert "amt_30d" in out.columns  # untouched by this op, preserved
    assert "amt_180d" in out.columns


# ---------- Recipe round-trip ----------

def test_recipe_round_trip_equivalence():
    plan = load_derivation_plan(_cfg(families=[
        {"family_base": "amt",
         "ops": [{"op": "delta", "left": "all", "right": "7d", "output": "d"},
                 {"op": "ratio", "numerator": "all", "denominator": "30d", "output": "r"},
                 {"op": "velocity", "short": "7d", "long": "180d", "output": "v"},
                 {"op": "incremental", "chain": ["7d", "30d", "all"],
                  "output_prefix": "amt_incr"}]},
    ]))
    a, added_a, _ = apply_derivations(_df(), plan)
    recipe = to_recipe_json(plan)
    b = apply_recipe(_df(), recipe)

    # Both frames should carry the same derived columns with identical values.
    for c in added_a:
        # Using fillna sentinels because NaN != NaN in direct compare.
        assert a[c].fillna(-9999).equals(b[c].fillna(-9999)), "column {0} differs".format(c)

    # Recipe dumps to stable JSON (no fields sorted alphabetically, but deterministic).
    s1 = dumps(recipe)
    s2 = dumps(to_recipe_json(plan))
    assert s1 == s2


def test_apply_recipe_rejects_unknown_version():
    with pytest.raises(ValueError, match="version"):
        apply_recipe(_df(), {"version": 99, "enabled": True, "ops": []})


def test_apply_missing_input_raises():
    plan = load_derivation_plan(_cfg(families=[{
        "family_base": "amt",
        "ops": [{"op": "delta", "left": "99d", "right": "7d", "output": "d"}],
    }]))
    with pytest.raises(KeyError, match="amt_99d"):
        apply_derivations(_df(), plan)
