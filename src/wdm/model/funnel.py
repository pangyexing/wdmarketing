"""Fused-funnel evaluation kernels (library half of scripts/run_funnel_eval.py).

Pure computation and report rendering live here so they are unit-testable:
per-stage absolute/conditional lift rows for a ranking score, bundle split
boundaries (manifest first, CSV re-derivation fallback), tier-value parsing
and the markdown report. The script keeps argument parsing, scoring
orchestration and console printing.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wdm.metrics.ranking import _top_k_mask
from wdm.utils.time_utils import split_by_yyyymmdd

logger = logging.getLogger(__name__)


def rate(mask_num, mask_den):
    """Mean of mask_num over the rows selected by mask_den; NaN when empty."""
    den = int(mask_den.sum())
    if den == 0:
        return float("nan")
    return float(mask_num[mask_den].mean())


def lift(top_rate, base_rate):
    if not np.isfinite(top_rate) or not np.isfinite(base_rate) or base_rate == 0:
        return float("nan")
    return top_rate / base_rate


def funnel_rows(score_name, scores, stage_flags, k, stages, value_vec=None):
    """Tidy metric rows for one ranking score at one top-K fraction.

    Absolute view: each stage flag's rate inside top-K vs the whole
    population. Conditional view: step conversions (stage | previous stages)
    inside top-K vs the population. value_vec adds a value_capture row
    (per-person mean business value in top-K vs population).
    """
    mask, k_int = _top_k_mask(scores, k)
    n = scores.size
    all_rows = np.ones(n, dtype=bool)
    rows = []

    for stage in stages:
        flag = stage_flags[stage]
        base = rate(flag, all_rows)
        top = rate(flag, mask)
        rows.append({
            "score": score_name, "top_k_pct": k, "view": "absolute",
            "stage": stage, "n_topk": k_int,
            "pos_topk": int(flag[mask].sum()),
            "base_rate": base, "topk_rate": top, "lift": lift(top, base),
        })

    if value_vec is not None:
        base_v = float(value_vec.mean())
        top_v = float(value_vec[mask].mean()) if k_int else float("nan")
        rows.append({
            "score": score_name, "top_k_pct": k, "view": "absolute",
            "stage": "value_capture", "n_topk": k_int,
            "pos_topk": int((value_vec[mask] > 0).sum()),
            "base_rate": base_v, "topk_rate": top_v, "lift": lift(top_v, base_v),
        })

    prev_all = all_rows
    prev_top = mask
    for stage in stages:
        flag = stage_flags[stage].astype(bool)
        base = rate(stage_flags[stage], prev_all)
        top = rate(stage_flags[stage], prev_top)
        rows.append({
            "score": score_name, "top_k_pct": k, "view": "conditional",
            "stage": stage, "n_topk": int(prev_top.sum()),
            "pos_topk": int(stage_flags[stage][prev_top].sum()),
            "base_rate": base, "topk_rate": top, "lift": lift(top, base),
        })
        prev_all = prev_all & flag
        prev_top = prev_top & flag
    return rows


def derive_boundaries(cfg):
    """Fallback for old bundles: re-derive valid/oot start dts by re-running
    the time split on the model's CURRENT training table. Returns
    {valid_min_dt, oot_min_dt} or None for non-time splits."""
    split_cfg = cfg["training"]["split"]
    if split_cfg.get("strategy") != "time":
        return None
    time_col = cfg["data"]["time_column"]
    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    dt = pd.read_csv(path, usecols=[time_col])[time_col]
    _m_tr, m_va, m_oot = split_by_yyyymmdd(
        dt, list(split_cfg["ratios"]),
        embargo_days=int(split_cfg.get("embargo_days", 0) or 0))
    nums = pd.to_numeric(dt, errors="coerce")
    out = {}
    for key, mask in (("valid_min_dt", m_va), ("oot_min_dt", m_oot)):
        out[key] = int(nums[mask].min()) if mask.any() else None
    return out


def model_boundaries(name, cfg, bundle_dir):
    """Read split_boundaries from the bundle manifest; fall back to deriving
    from the current CSV (warned — drifts if the CSV was rebuilt)."""
    manifest_path = Path(bundle_dir) / "run_manifest.json"
    if manifest_path.is_file():
        with open(str(manifest_path), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        sb = manifest.get("split_boundaries")
        if sb and sb.get("oot_min_dt") is not None:
            return {"valid_min_dt": sb.get("valid_min_dt"),
                    "oot_min_dt": sb.get("oot_min_dt"), "source": "manifest"}
    derived = derive_boundaries(cfg)
    if derived is None:
        return None
    logger.warning("%s bundle has no persisted split_boundaries; re-derived from "
                   "the current CSV (valid>=%s, oot>=%s) — stale if the table "
                   "was rebuilt after training", name,
                   derived["valid_min_dt"], derived["oot_min_dt"])
    derived["source"] = "re-derived"
    return derived


def parse_tier_values(spec):
    """'1:120,2:290,3:790' -> {1.0: 120.0, 2.0: 290.0, 3.0: 790.0}"""
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition(":")
        out[float(key)] = float(val)
    if not out:
        raise ValueError("--tier-values parsed to an empty mapping: {0!r}".format(spec))
    return out


def write_markdown(path, df, info, final_stage):
    """Render funnel_eval.md from the tidy result frame + run info pairs."""
    lines = ["# Fused funnel evaluation", ""]
    for key, val in info:
        lines.append("- **{0}**: {1}".format(key, val))
    lines.append("")
    lines.append("`fused` = calibrated resp x qual product; `fused_alpha` = "
                 "resp^alpha x qual^(1-alpha) with alpha fit on the pre-OOT fit "
                 "window. The `absolute` view compares each funnel stage rate in "
                 "top-K vs the population (the `{0}` row is the end-to-end "
                 "综合提升; `value_capture`, when present, compares per-person "
                 "business value); the `conditional` view compares step "
                 "conversions (reg, finish|reg, credit|finish) inside top-K vs "
                 "the population.".format(final_stage))
    for k in sorted(df["top_k_pct"].unique()):
        for view in ("absolute", "conditional"):
            block = df[(df["top_k_pct"] == k) & (df["view"] == view)]
            lines.append("")
            lines.append("## Top {0:.0%} — {1}".format(k, view))
            lines.append("")
            lines.append("| score | stage | n_topk | pos_topk | base_rate | topk_rate | lift |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in block.iterrows():
                lines.append("| {0} | {1} | {2} | {3} | {4:.4f} | {5:.4f} | {6:.4f} |".format(
                    r["score"], r["stage"], r["n_topk"], r["pos_topk"],
                    r["base_rate"], r["topk_rate"], r["lift"]))
    with open(str(path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
