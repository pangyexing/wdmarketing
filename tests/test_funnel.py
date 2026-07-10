"""wdm.model.funnel — the extracted, unit-testable half of
scripts/run_funnel_eval.py: per-stage lift rows, bundle boundaries,
tier-value parsing, markdown rendering."""
import json

import numpy as np
import pandas as pd
import pytest

from wdm.model.funnel import (
    funnel_rows, lift, model_boundaries, parse_tier_values, rate,
    write_markdown)


def test_rate_and_lift_edge_cases():
    flag = np.array([1, 0, 1, 0], dtype=bool)
    assert rate(flag, np.ones(4, dtype=bool)) == 0.5
    assert np.isnan(rate(flag, np.zeros(4, dtype=bool)))
    assert lift(0.4, 0.2) == pytest.approx(2.0)
    assert np.isnan(lift(0.4, 0.0))
    assert np.isnan(lift(float("nan"), 0.2))


def _toy_funnel(n=100):
    """Scores rank rows 0..n-1 descending; stage flags concentrate at the top."""
    scores = np.linspace(1.0, 0.0, n)
    reg = np.zeros(n, dtype=bool)
    reg[:60] = True
    finish = np.zeros(n, dtype=bool)
    finish[:30] = True
    credit = np.zeros(n, dtype=bool)
    credit[:10] = True
    return scores, {"is_reg": reg, "is_finish_task": finish,
                    "is_credit_succ": credit}


def test_funnel_rows_absolute_math():
    scores, flags = _toy_funnel()
    stages = ["is_reg", "is_finish_task", "is_credit_succ"]
    rows = pd.DataFrame(funnel_rows("fused", scores, flags, 0.10, stages))
    ab = rows[rows["view"] == "absolute"].set_index("stage")
    # top-10% = rows 0..9, all of which are credit=1
    assert ab.loc["is_credit_succ", "n_topk"] == 10
    assert ab.loc["is_credit_succ", "pos_topk"] == 10
    assert ab.loc["is_credit_succ", "topk_rate"] == pytest.approx(1.0)
    assert ab.loc["is_credit_succ", "base_rate"] == pytest.approx(0.10)
    assert ab.loc["is_credit_succ", "lift"] == pytest.approx(10.0)


def test_funnel_rows_conditional_chains_stages():
    scores, flags = _toy_funnel()
    stages = ["is_reg", "is_finish_task", "is_credit_succ"]
    rows = pd.DataFrame(funnel_rows("fused", scores, flags, 0.30, stages))
    cond = rows[rows["view"] == "conditional"].set_index("stage")
    # conditional finish|reg over the population: 30/60
    assert cond.loc["is_finish_task", "base_rate"] == pytest.approx(0.5)
    # inside top-30 all rows are reg → n_topk for the finish step is 30
    assert cond.loc["is_finish_task", "n_topk"] == 30
    # credit|finish inside top-30: prev_top = first 30 ∩ finish = 30 rows...
    # finish flags rows 0..29 → credit step conditions on those 30 rows, 10 hit
    assert cond.loc["is_credit_succ", "n_topk"] == 30
    assert cond.loc["is_credit_succ", "pos_topk"] == 10


def test_funnel_rows_value_capture():
    scores, flags = _toy_funnel()
    value = np.zeros(100)
    value[:10] = 790.0
    rows = pd.DataFrame(funnel_rows(
        "fused", scores, flags, 0.10, ["is_reg"], value_vec=value))
    vc = rows[rows["stage"] == "value_capture"].iloc[0]
    assert vc["topk_rate"] == pytest.approx(790.0)
    assert vc["base_rate"] == pytest.approx(79.0)
    assert vc["lift"] == pytest.approx(10.0)


def test_parse_tier_values():
    assert parse_tier_values("1:120,2:290,3:790") == {1.0: 120.0, 2.0: 290.0,
                                                      3.0: 790.0}
    with pytest.raises(ValueError):
        parse_tier_values(",")


def test_model_boundaries_prefers_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "split_boundaries": {"valid_min_dt": 20250201, "oot_min_dt": 20250301},
    }), encoding="utf-8")
    b = model_boundaries("resp", {"training": {"split": {}}}, tmp_path)
    assert b == {"valid_min_dt": 20250201, "oot_min_dt": 20250301,
                 "source": "manifest"}


def test_model_boundaries_none_for_non_time_split(tmp_path):
    cfg = {"training": {"split": {"strategy": "stratified"}}}
    assert model_boundaries("resp", cfg, tmp_path) is None


def test_write_markdown_smoke(tmp_path):
    scores, flags = _toy_funnel()
    stages = ["is_reg", "is_finish_task", "is_credit_succ"]
    df = pd.DataFrame(funnel_rows("fused", scores, flags, 0.10, stages))
    out = tmp_path / "funnel_eval.md"
    write_markdown(out, df, [("eval window", "dt >= 20250301")],
                   "is_credit_succ")
    text = out.read_text(encoding="utf-8")
    assert "# Fused funnel evaluation" in text
    assert "dt >= 20250301" in text
    assert "## Top 10% — absolute" in text
    assert "| fused | is_credit_succ |" in text
