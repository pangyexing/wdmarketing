"""Fused-model evaluation of the FULL analysis funnel.

Modeling is two models on a TWO-stage funnel (response 响应 = is_finish_task +
qualification 资质 V1/V2, see build_xc_dataset.py). The ANALYSIS funnel is the
full process is_reg -> is_finish_task -> credit, where the final credit stage
matches the qualification model being evaluated (--qual-stage):

  is_credit_succ   qualification V1 (xc_qual_finish)       [default]
  is_credit_1v1    qualification V2 (xc_qual_finish_1v1)

This script:

  1. scores the full population (data/xc_full.csv) with both trained bundles,
     using each bundle's CALIBRATED score when calibration.json is present
     (raw probabilities are distorted differently per model by
     scale_pos_weight, so fusing raw scores distorts the ranking);
  2. fuses the scores two ways:
       fused        = calib_resp * calib_qual   (plain product, legacy form)
       fused_alpha  = calib_resp^alpha * calib_qual^(1-alpha), with alpha
         grid-fit on a FIT WINDOW that precedes the OOT evaluation window
         (the overlap of both models' valid periods), maximizing the credit
         stage's lift@K — so the OOT evaluation stays honest;
  3. ranks by each score and reports, for each top-K:
       - absolute view: top-K rate vs population base rate for EVERY funnel
         stage flag; the final credit row is the end-to-end (综合) lift
       - conditional view: step conversions inside top-K vs population
     resp_only / qual_only — and e2e_only when --e2e-product is given (the
     single end-to-end model baseline) — are reported alongside;
  4. optionally reports VALUE capture for the credit_1v1 stage
     (--tier-values "1:120,2:290,3:790"): top-K mean business value vs the
     population mean;
  5. writes fusion_spec.json (alpha, fit/eval windows, per-alpha fit curve)
     next to funnel_eval.{csv,md} — the serving side executes its
     serving_formula on the two bundles' score_calibrated columns.

Evaluation window: by default the OOT period only. Each model's boundaries are
read from its bundle's run_manifest.json (split_boundaries, persisted at
export); for old bundles they are re-derived from the current training CSV
with a warning. The eval window starts at the LATEST OOT start across models.
Override with --start-dt / --end-dt (yyyymmdd ints). Non-time splits require
an explicit --start-dt (or --full-window).

Usage:
    PYTHONPATH=src python3 scripts/run_funnel_eval.py \
        --resp-product xc_resp_finish --resp-run-id r01 \
        --qual-product xc_qual_finish --qual-run-id q01

    # qualification V2 (credit_1v1 label) + e2e baseline + value capture:
    PYTHONPATH=src python3 scripts/run_funnel_eval.py \
        --resp-product xc_resp_finish --resp-run-id r01 \
        --qual-product xc_qual_finish_1v1 --qual-run-id q01 \
        --qual-stage is_credit_1v1 \
        --e2e-product xc_e2e_credit_1v1 --e2e-run-id e01 \
        --tier-values "1:120,2:290,3:790"
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from predict_template import Predictor  # noqa: E402  (scripts/ on path)
from wdm.config import load_config  # noqa: E402
from wdm.metrics.ranking import lift_at_k  # noqa: E402
from wdm.model.funnel import (  # noqa: E402
    funnel_rows, model_boundaries, parse_tier_values, write_markdown,
)
from wdm.model.fusion import fit_alpha, fuse  # noqa: E402
from wdm.utils.logging import setup_logging  # noqa: E402
from wdm.utils.paths import model_run_dir  # noqa: E402

logger = logging.getLogger(__name__)

# Default analysis funnel; the final credit stage is swapped via --qual-stage
# to match the qualification model under evaluation.
FUNNEL_STAGES = ["is_reg", "is_finish_task", "is_credit_succ"]
QUAL_STAGES = ["is_credit_succ", "is_credit_1v1"]
RAW_TIER_COLUMN = "credit_1v1"


def _print_block(df, k, final_stage):
    sub = df[df["top_k_pct"] == k]
    for view in ("absolute", "conditional"):
        block = sub[sub["view"] == view][
            ["score", "stage", "n_topk", "pos_topk", "base_rate", "topk_rate", "lift"]]
        title = ("Top {0:.0%} — absolute (top-K rate vs population; " + final_stage + " = 综合提升)"
                 if view == "absolute" else
                 "Top {0:.0%} — conditional (step conversion inside top-K vs population)")
        print()
        print(title.format(k))
        print(block.to_string(index=False, float_format=lambda v: "{0:.4f}".format(v)))


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the FULL analysis funnel with fused response x qualification scores.")
    ap.add_argument("--resp-product", required=True, help="Response model product (e.g. xc_resp_finish)")
    ap.add_argument("--resp-run-id", required=True)
    ap.add_argument("--qual-product", required=True,
                    help="Qualification model product (e.g. xc_qual_finish or xc_qual_finish_1v1)")
    ap.add_argument("--qual-run-id", required=True)
    ap.add_argument("--qual-stage", choices=QUAL_STAGES, default="is_credit_succ",
                    help="Final funnel stage = the qualification model's label. "
                         "Use is_credit_1v1 when evaluating xc_qual_finish_1v1.")
    ap.add_argument("--e2e-product", default=None,
                    help="Optional end-to-end single-model baseline product "
                         "(e.g. xc_e2e_credit); adds an e2e_only ranking.")
    ap.add_argument("--e2e-run-id", default=None)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Manual fusion alpha in [0,1] (skips grid fitting).")
    ap.add_argument("--alpha-k", type=float, default=0.10,
                    help="Top-K fraction targeted by the alpha grid fit.")
    ap.add_argument("--no-alpha", action="store_true",
                    help="Skip the fused_alpha ranking entirely (plain product only).")
    ap.add_argument("--tier-values", default=None,
                    help="Business value per credit_1v1 tier, e.g. '1:120,2:290,3:790'. "
                         "Adds a value_capture row (requires --qual-stage is_credit_1v1).")
    ap.add_argument("--data", default="data/xc_full.csv",
                    help="Full-population table with features + the funnel flags.")
    ap.add_argument("--top-k", default="0.05,0.10,0.20",
                    help="Comma-separated top-K fractions.")
    ap.add_argument("--start-dt", type=int, default=None,
                    help="Eval window start (yyyymmdd, inclusive). Default: auto OOT boundary.")
    ap.add_argument("--end-dt", type=int, default=None,
                    help="Eval window end (yyyymmdd, inclusive). Default: open.")
    ap.add_argument("--full-window", action="store_true",
                    help="Evaluate on ALL rows (includes both models' training periods — lift is optimistic).")
    ap.add_argument("--out", default=None,
                    help="Output dir. Default: artifacts/funnel_eval/<resp-run>__<qual-run>/")
    args = ap.parse_args()

    setup_logging()
    if bool(args.e2e_product) != bool(args.e2e_run_id):
        raise SystemExit("--e2e-product and --e2e-run-id must be given together.")
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be within [0, 1].")
    funnel_stages = FUNNEL_STAGES[:-1] + [args.qual_stage]
    resp_cfg = load_config(args.resp_product)
    qual_cfg = load_config(args.qual_product)
    if args.qual_product.endswith("_1v1") and args.qual_stage != "is_credit_1v1":
        logger.warning("qual product %s looks like the credit_1v1 model but --qual-stage is %s; "
                       "end-to-end lift will be measured against the wrong label.",
                       args.qual_product, args.qual_stage)
    tier_values = None
    if args.tier_values:
        if args.qual_stage != "is_credit_1v1":
            raise SystemExit("--tier-values only applies to --qual-stage is_credit_1v1.")
        tier_values = parse_tier_values(args.tier_values)

    models = [("resp", args.resp_product, args.resp_run_id, resp_cfg),
              ("qual", args.qual_product, args.qual_run_id, qual_cfg)]
    if args.e2e_product:
        models.append(("e2e", args.e2e_product, args.e2e_run_id,
                       load_config(args.e2e_product)))

    preds = {}
    bounds = {}
    for name, product, run_id, cfg in models:
        bundle = model_run_dir(cfg, run_id)
        if not bundle.is_dir():
            raise FileNotFoundError("{0} bundle not found: {1}".format(name, bundle))
        preds[name] = Predictor(bundle)
        bounds[name] = model_boundaries(name, cfg, bundle)
        logger.info("%s bundle: %s (%d features, calibration=%s)", name, bundle,
                    len(preds[name].base_features), preds[name].has_calibration)

    use_calibrated = all(p.has_calibration for p in preds.values())
    if not use_calibrated:
        missing = [n for n, p in preds.items() if not p.has_calibration]
        logger.warning("bundle(s) %s have NO calibration.json — fusing RAW scores "
                       "(rank-distorted product; retrain/export to get calibrated fusion)",
                       missing)

    # ---- evaluation + alpha-fit windows -------------------------------------
    start_dt = args.start_dt
    window_desc = "full"
    if start_dt is None and not args.full_window:
        oot_mins = [bounds[n]["oot_min_dt"] if bounds[n] else None for n in preds]
        if any(v is None for v in oot_mins):
            raise SystemExit(
                "Cannot auto-derive the OOT window (a model uses a non-time split). "
                "Pass --start-dt explicitly, or --full-window for an in-sample eval.")
        start_dt = max(oot_mins)
        logger.info("OOT boundaries %s -> eval window starts %s",
                    {n: bounds[n]["oot_min_dt"] for n in preds}, start_dt)
    elif args.full_window:
        logger.warning("--full-window: eval includes training-period rows; lift is OPTIMISTIC.")

    alpha_mode = "skip" if args.no_alpha else ("manual" if args.alpha is not None else "fit")
    fit_start = None
    if alpha_mode == "fit":
        valid_mins = [bounds[n].get("valid_min_dt") if bounds[n] else None for n in preds]
        if start_dt is None or any(v is None for v in valid_mins):
            logger.warning("alpha fit disabled: no honest fit window "
                           "(--full-window, non-time split or missing boundaries) "
                           "-> fused_alpha uses alpha=0.5")
            alpha_mode = "fallback"
        else:
            fit_start = max(valid_mins)
            if fit_start >= start_dt:
                logger.warning("alpha fit disabled: degenerate fit window "
                               "[%s, %s) -> fused_alpha uses alpha=0.5",
                               fit_start, start_dt)
                alpha_mode = "fallback"
                fit_start = None

    # ---- load + score ------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO / data_path
    time_col = resp_cfg["data"]["time_column"]
    needed = set(funnel_stages) | {time_col}
    for p in preds.values():
        needed |= set(p.base_features)
    if tier_values is not None:
        needed |= {RAW_TIER_COLUMN}
    df = pd.read_csv(data_path, usecols=sorted(needed))
    n_all = len(df)

    dt_vals = pd.to_numeric(df[time_col], errors="coerce")
    load_start = fit_start if fit_start is not None else start_dt
    keep = pd.Series(True, index=df.index)
    if load_start is not None:
        keep &= dt_vals >= load_start
    if args.end_dt is not None:
        keep &= dt_vals <= args.end_dt
    df = df[keep].reset_index(drop=True)
    dt_vals = dt_vals[keep].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("Selected window (dt >= {0}) selected 0 rows.".format(load_start))

    if start_dt is not None:
        eval_mask = (dt_vals >= start_dt).values
        window_desc = "dt >= {0}".format(start_dt)
        if args.end_dt is not None:
            window_desc += " and dt <= {0}".format(args.end_dt)
    else:
        eval_mask = np.ones(len(df), dtype=bool)
        if args.end_dt is not None:
            window_desc = "full and dt <= {0}".format(args.end_dt)
    fit_mask = ((dt_vals >= fit_start).values & ~eval_mask) if fit_start is not None \
        else np.zeros(len(df), dtype=bool)
    if not eval_mask.any():
        raise SystemExit("Evaluation window ({0}) selected 0 rows.".format(window_desc))
    logger.info("Loaded %d / %d rows; eval window %s: %d rows, alpha-fit window: %d rows",
                len(df), n_all, window_desc, int(eval_mask.sum()), int(fit_mask.sum()))

    raw_scores = {}
    cal_scores = {}
    for name, p in preds.items():
        raw = np.asarray(p.predict_proba(df[p.base_features]), dtype=np.float64)
        raw_scores[name] = raw
        cal_scores[name] = p.calibrate(raw) if use_calibrated else raw

    stage_flags = {}
    for stage in funnel_stages:
        stage_flags[stage] = (pd.to_numeric(df[stage], errors="coerce").fillna(0) > 0).values

    value_vec = None
    if tier_values is not None:
        tiers = pd.to_numeric(df[RAW_TIER_COLUMN], errors="coerce")
        value_vec = tiers.map(tier_values).fillna(0.0).values.astype(np.float64)

    # ---- alpha -------------------------------------------------------------
    alpha = None
    alpha_source = None
    fit_results = []
    if alpha_mode == "manual":
        alpha, alpha_source = float(args.alpha), "manual"
    elif alpha_mode == "fit":
        alpha, alpha_source, fit_results = fit_alpha(
            cal_scores["resp"][fit_mask], cal_scores["qual"][fit_mask],
            stage_flags[args.qual_stage][fit_mask], k_pct=args.alpha_k)
    elif alpha_mode == "fallback":
        alpha, alpha_source = 0.5, "default_fallback"
    if alpha is not None:
        logger.info("fusion alpha=%.2f (%s)", alpha, alpha_source)

    # ---- rankings on the EVAL window only -----------------------------------
    rankings = [("fused", cal_scores["resp"][eval_mask] * cal_scores["qual"][eval_mask]),
                ("resp_only", cal_scores["resp"][eval_mask]),
                ("qual_only", cal_scores["qual"][eval_mask])]
    if alpha is not None:
        rankings.insert(1, ("fused_alpha",
                            fuse(cal_scores["resp"][eval_mask],
                                 cal_scores["qual"][eval_mask], alpha)))
    if "e2e" in cal_scores:
        rankings.append(("e2e_only", cal_scores["e2e"][eval_mask]))
    flags_eval = {s: f[eval_mask] for s, f in stage_flags.items()}
    value_eval = value_vec[eval_mask] if value_vec is not None else None

    ks = [float(s) for s in str(args.top_k).split(",") if s.strip()]
    rows = []
    for k in ks:
        for name, scores in rankings:
            rows.extend(funnel_rows(name, scores, flags_eval, k, funnel_stages,
                                    value_vec=value_eval))
    result = pd.DataFrame(rows, columns=[
        "score", "top_k_pct", "view", "stage", "n_topk", "pos_topk",
        "base_rate", "topk_rate", "lift"])

    # End-to-end credit lift at the alpha-fit K, per ranking (summary + spec).
    y_stage_eval = stage_flags[args.qual_stage][eval_mask]
    eval_lift = {name: float(lift_at_k(y_stage_eval, scores, float(args.alpha_k)))
                 for name, scores in rankings}

    # ---- report ------------------------------------------------------------
    out_dir = Path(args.out) if args.out else (
        REPO / "artifacts" / "funnel_eval" /
        "{0}-{1}__{2}-{3}".format(args.resp_product, args.resp_run_id,
                                  args.qual_product, args.qual_run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    info = [
        ("response model", "{0} / {1}".format(args.resp_product, args.resp_run_id)),
        ("qualification model", "{0} / {1}".format(args.qual_product, args.qual_run_id)),
        ("qualification stage", args.qual_stage),
        ("data", str(data_path)),
        ("scores", "calibrated" if use_calibrated else "RAW (no calibration.json)"),
        ("eval window", window_desc),
        ("rows evaluated", "{0} (of {1})".format(int(eval_mask.sum()), n_all)),
    ]
    if args.e2e_product:
        info.insert(2, ("e2e model", "{0} / {1}".format(args.e2e_product, args.e2e_run_id)))
    if alpha is not None:
        info.append(("fusion alpha", "{0:.2f} ({1}; fit window {2} rows, k={3:.0%})".format(
            alpha, alpha_source, int(fit_mask.sum()), args.alpha_k)))
    info.append(("{0} lift@{1:.0%} by ranking".format(args.qual_stage, args.alpha_k),
                 ", ".join("{0}={1:.3f}".format(n, v) for n, v in eval_lift.items())))
    print()
    print("=" * 72)
    print("Fused funnel evaluation (analysis funnel: {0})".format(" -> ".join(funnel_stages)))
    for key, val in info:
        print("  {0:<22}: {1}".format(key, val))
    for k in ks:
        _print_block(result, k, args.qual_stage)
    print()
    print("=" * 72)

    csv_path = out_dir / "funnel_eval.csv"
    result.to_csv(str(csv_path), index=False)
    write_markdown(out_dir / "funnel_eval.md", result, info, args.qual_stage)
    print("Wrote {0}".format(csv_path))
    print("Wrote {0}".format(out_dir / "funnel_eval.md"))

    # ---- fusion_spec.json (serving contract) --------------------------------
    spec = {
        "version": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "serving_formula": ("score = calib_resp^alpha * calib_qual^(1-alpha) "
                            "(score_calibrated columns from the two bundles)"
                            if use_calibrated else
                            "score = resp^alpha * qual^(1-alpha) (RAW scores — "
                            "bundles lack calibration; retrain to fix)"),
        "alpha": alpha,
        "alpha_source": alpha_source,
        "alpha_grid": {"start": 0.0, "stop": 1.0, "step": 0.05},
        "objective": {"metric": "lift_at_k", "k_pct": float(args.alpha_k),
                      "stage": args.qual_stage},
        "calibrated": use_calibrated,
        "fit_window": {"start_dt": int(fit_start) if fit_start is not None else None,
                       "end_dt_exclusive": int(start_dt) if (fit_start is not None
                                                             and start_dt is not None) else None,
                       "n_rows": int(fit_mask.sum()),
                       "n_pos_stage": int(stage_flags[args.qual_stage][fit_mask].sum())},
        "eval_window": {"start_dt": int(start_dt) if start_dt is not None else None,
                        "end_dt": int(args.end_dt) if args.end_dt is not None else None,
                        "n_rows": int(eval_mask.sum())},
        "models": {name: {"product": product, "run_id": run_id,
                          "calibrated": preds[name].has_calibration,
                          "boundaries": bounds[name]}
                   for name, product, run_id, _cfg in models},
        "tier_values": ({str(int(k)): v for k, v in tier_values.items()}
                        if tier_values else None),
        "fit_results": fit_results,
        "eval_lift_at_k": eval_lift,
    }
    spec_path = out_dir / "fusion_spec.json"
    with open(str(spec_path), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print("Wrote {0}".format(spec_path))


if __name__ == "__main__":
    main()
