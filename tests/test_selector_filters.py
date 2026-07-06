"""Edge-case regression for selector hard filters + rank/auto-keep.

Expected values were captured from the pre-vectorization (iterrows)
implementation — the vectorized version must reproduce them exactly,
including the quirky-but-load-bearing corners:
  * family_kept=NaN counts as kept (bool(NaN) is True)
  * short-window (7d/30d) features get the 0.98 soft missing cap
  * low-IV features survive via the lift_at_k soft gate
  * rank_score ties inside a correlation cluster resolve by stable sort
    order (first occurrence in the frame wins)
"""
import numpy as np
import pandas as pd

from wdm.analysis.selector import _apply_hard_filters, _rank_and_auto_keep

CFG = {"analysis": {
    "missing_rate_max": 0.95, "iv_min": 0.02, "psi_cutoff": 0.25,
    "lift_keep_min": 1.2, "short_windows": ["7d", "30d"],
    "short_window_missing_rate_max": 0.98,
    "rank_weights": {"iv": 1.0, "lift": 1.0, "gini": 1.0, "concentration": 0.0,
                     "psi": 1.0, "missing_penalty": 0.5,
                     "missing_penalty_threshold": 0.5}}}


def _input_df():
    return pd.DataFrame([
        ("f_const",      None,  1, 0.00, 0.000, 1.00, 0.0, 0.0, 0.00, True,   True,  -1),
        ("f_hi_miss",    None, 50, 0.96, 0.100, 1.50, 0.1, 0.1, 0.01, True,   True,  -1),
        ("f_short_7d",   "7d", 50, 0.96, 0.100, 1.50, 0.1, 0.1, 0.01, True,   True,  -1),
        ("f_lowiv",      None, 50, 0.10, 0.010, 1.00, 0.0, 0.0, 0.01, True,   True,  -1),
        ("f_lowiv_save", None, 50, 0.10, 0.010, 1.90, 0.2, 0.2, 0.01, True,   True,  -1),
        ("f_hi_psi",     None, 50, 0.10, 0.300, 1.80, 0.2, 0.2, 0.30, True,   True,  -1),
        ("f_fam_nan",    None, 50, 0.10, 0.200, 1.60, 0.2, 0.2, 0.01, np.nan, True,  -1),
        ("f_fam_drop",   None, 50, 0.10, 0.200, 1.60, 0.2, 0.2, 0.01, False,  True,  -1),
        ("f_grp_drop",   None, 50, 0.10, 0.200, 1.60, 0.2, 0.2, 0.01, True,   False, -1),
        ("f_tie_a",      None, 50, 0.10, 0.200, 1.60, 0.2, 0.2, 0.01, True,   True,   3),
        ("f_tie_b",      None, 50, 0.10, 0.200, 1.60, 0.2, 0.2, 0.01, True,   True,   3),
        ("f_clu_lo",     None, 50, 0.10, 0.150, 1.40, 0.1, 0.1, 0.01, True,   True,   3),
        ("f_miss_pen",   None, 50, 0.60, 0.200, 1.60, 0.2, 0.2, 0.01, True,   True,  -1),
    ], columns=["feature", "window", "n_unique", "missing_rate", "iv", "lift_at_k",
                "gini", "concentration", "psi", "family_kept", "group_kept",
                "corr_cluster"])


# (feature, auto_keep, drop_reason, rank_score rounded to 1e-10)
EXPECTED = [
    ("f_const", False, "constant;low_iv", -5.2194453149),
    ("f_hi_miss", False, "high_missing", -1.3933704694),
    ("f_short_7d", True, "", -1.3933704694),
    ("f_lowiv", False, "low_iv", -5.2364440875),
    ("f_lowiv_save", True, "", 1.0354921625),
    ("f_hi_psi", False, "high_psi", 0.1436120887),
    ("f_fam_nan", True, "", 1.9660013287),
    ("f_fam_drop", False, "family_dropped_by_policy", 1.9660013287),
    ("f_grp_drop", False, "group_dropped_by_policy", 1.9660013287),
    ("f_tie_a", True, "", 1.9660013287),
    ("f_tie_b", False, "corr_dup_of:f_tie_a", 1.9660013287),
    ("f_clu_lo", False, "corr_dup_of:f_tie_a", -0.7324818821),
    ("f_miss_pen", True, "", 1.4660013287),
]


def test_hard_filters_and_auto_keep_edge_cases():
    out = _rank_and_auto_keep(_apply_hard_filters(_input_df(), CFG), CFG)
    got = [(r.feature, bool(r.auto_keep), r.drop_reason,
            round(float(r.rank_score), 10)) for r in out.itertuples()]
    assert got == EXPECTED


def test_helper_columns_are_dropped():
    out = _rank_and_auto_keep(_apply_hard_filters(_input_df(), CFG), CFG)
    assert "_hard_drop" not in out.columns
    assert "_hard_drop_reason" not in out.columns


def test_no_lift_keep_min_disables_soft_gate():
    cfg = {"analysis": dict(CFG["analysis"], lift_keep_min=None)}
    out = _apply_hard_filters(_input_df(), cfg)
    by_feat = dict(zip(out["feature"], out["_hard_drop_reason"]))
    assert by_feat["f_lowiv_save"] == "low_iv"
