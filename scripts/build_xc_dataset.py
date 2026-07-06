"""Build the xc FUNNEL training tables from a feature file + a label file.

The two raw inputs share the row key (id + date):

  features:  id, dt(yyyy-mm-dd or yyyymmdd), feat1, ..., feat1000
  labels:    id, dt(yyyymmdd), is_reg, is_finish_task, is_credit_succ, credit_1v1

MODELING funnel: two-stage, ONE response model (响应) + TWO qualification
versions (资质 V1/V2):

  response          all rows                         label = is_finish_task  (xc_resp_finish)
  qualification V1  rows where is_finish_task == 1   label = is_credit_succ  (xc_qual_finish)
                    (is_credit_succ: 1 -> 正样本, 0 -> 负样本)
  qualification V2  rows where is_finish_task == 1   label = is_credit_1v1   (xc_qual_finish_1v1)
                    (credit_1v1: 1/2/3 -> 正样本, 0/-1 -> 负样本)

is_credit_1v1 is DERIVED here from the raw credit_1v1 column (1/2/3 -> 1,
0/-1 -> 0, anything else -> 0 with a warning); credit_1v1 itself is carried
through for auditing but must never become a feature or label.

Emitted tables (a config reads a whole table and uses every row — there is no
row-filter hook, so the conditioning population is physical):

  data/xc_full.csv         all rows                        response, funnel eval
  data/xc_qual_finish.csv  rows where is_finish_task == 1  qualification V1 + V2

Filtering at build time (not just at train time) is required because Stage-1
feature selection must also run on the conditioning population, else IV/PSI/Lift
are computed on the wrong denominator.

ANALYSIS funnel: the full process is_reg -> is_finish_task -> credit is NOT
modeled stage-by-stage (is_reg is analysis-only, never a label). It is reported
below as conversion stats for BOTH credit endpoints, and after training it is
evaluated with the fused model score (response x qualification) by
scripts/run_funnel_eval.py — per-stage and end-to-end top-K lift.

Each table keeps all outcome columns; the matching product config parks the
non-target ones in id_columns so the feature scanner excludes them (downstream,
intermediate and parallel outcomes must never become features). See
configs/products/xc_{resp_finish,qual_finish,qual_finish_1v1}.yaml.

Join:  both time columns are named 'dt' (features may be yyyy-mm-dd or
       yyyymmdd, labels yyyymmdd) and are normalized to int yyyymmdd 'dt'
       first, then merge on (id, dt). Robust to stray formatting and matches
       the config (time_column: dt, time_format: yyyymmdd). A legacy feature
       file still using 'apply_time' can be joined via --feat-time-col.

Usage:
    PYTHONPATH=src python3 scripts/build_xc_dataset.py \
        --features data/xc_features.csv \
        --labels   data/xc_labels.csv \
        --out-dir  data
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Funnel outcome flags (binarized: >0 -> 1).
TARGET_COLS = ["is_reg", "is_finish_task", "is_credit_succ"]
# Raw multi-valued credit outcome (kept for auditing, never a feature/label).
RAW_LABEL_COLS = ["credit_1v1"]
# Binary label derived from credit_1v1 for qualification V2.
CREDIT_1V1_FLAG = "is_credit_1v1"

# Modeling tables: (output_suffix, condition_column or None for full pop,
# configs served). Labels are NOT fixed per table — each product config picks
# its own label_column from the table.
MODELING_TABLES = [
    ("full",        None,             "xc_resp_finish(is_finish_task)"),
    ("qual_finish", "is_finish_task", "xc_qual_finish(is_credit_succ) xc_qual_finish_1v1(is_credit_1v1)"),
]

# Analysis funnels: the full sequential process, reported (never modeled),
# once per credit endpoint.
ANALYSIS_FUNNELS = [
    ["is_reg", "is_finish_task", "is_credit_succ"],
    ["is_reg", "is_finish_task", CREDIT_1V1_FLAG],
]

PRODUCTS = ["xc_resp_finish", "xc_qual_finish", "xc_qual_finish_1v1"]


def _to_yyyymmdd_int(time_series):
    """yyyy-mm-dd (or yyyymmdd) string/series -> nullable Int64 yyyymmdd."""
    s = time_series.astype(str).str.strip().str.replace("-", "", regex=False)
    ints = pd.to_numeric(s, errors="coerce")
    bad = int(ints.isna().sum())
    if bad:
        logger.warning("%d rows have an unparseable date -> dt is NaN (won't join)", bad)
    return ints.astype("Int64")


def _binarize(series):
    """Coerce a label column to {0,1}: numeric>0 -> 1, NaN/<=0 -> 0."""
    num = pd.to_numeric(series, errors="coerce")
    n_nan = int(num.isna().sum())
    if n_nan:
        logger.warning("  label has %d NaN/unparseable values -> treated as 0", n_nan)
    return (num.fillna(0) > 0).astype(np.int8)


def _binarize_credit_1v1(series):
    """credit_1v1 -> {0,1}: values 1/2/3 -> 1 (正样本), 0/-1 -> 0 (负样本).
    Anything else (NaN / unexpected codes) -> 0 with a warning."""
    num = pd.to_numeric(series, errors="coerce")
    pos = num.isin([1, 2, 3])
    neg = num.isin([0, -1])
    n_other = int((~(pos | neg)).sum())
    if n_other:
        logger.warning("  credit_1v1 has %d values outside {1,2,3} U {0,-1} "
                       "(incl. NaN) -> treated as negative", n_other)
    return pos.astype(np.int8)


def main():
    ap = argparse.ArgumentParser(description="Join xc feature + label files into modeling tables (response + qualification V1/V2).")
    ap.add_argument("--features", default="data/xc_features.csv",
                    help="Feature file: id, dt, feat1..featN")
    ap.add_argument("--labels", default="data/xc_labels.csv",
                    help="Label file: id, dt, is_reg, is_finish_task, is_credit_succ, credit_1v1")
    ap.add_argument("--out-dir", default="data",
                    help="Directory for xc_full.csv / xc_qual_finish.csv")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--feat-time-col", default="dt",
                    help="Date column in the FEATURE file (yyyy-mm-dd or yyyymmdd); "
                         "normalized to 'dt'. Legacy files: --feat-time-col apply_time.")
    ap.add_argument("--label-time-col", default="dt",
                    help="Date column in the LABEL file (yyyymmdd or yyyy-mm-dd); normalized to 'dt'.")
    ap.add_argument("--join", choices=["inner", "left"], default="inner",
                    help="inner: keep rows present in both (default). "
                         "left: keep all feature rows, NaN labels where unmatched.")
    args = ap.parse_args()

    setup_logging()
    repo = Path(__file__).resolve().parents[1]

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (repo / p)

    feat_path = _resolve(args.features)
    label_path = _resolve(args.labels)
    out_dir = _resolve(args.out_dir)

    id_col = args.id_col
    for p, what in ((feat_path, "features"), (label_path, "labels")):
        if not p.is_file():
            raise FileNotFoundError("{0} file not found: {1}".format(what, p))

    # Read keys as strings so big int IDs merge cleanly (no float coercion).
    logger.info("Loading features: %s", feat_path)
    feats = pd.read_csv(feat_path, dtype={id_col: str, args.feat_time_col: str})
    logger.info("  features shape: %s", feats.shape)

    logger.info("Loading labels: %s", label_path)
    labels = pd.read_csv(label_path, dtype={id_col: str, args.label_time_col: str})
    logger.info("  labels shape: %s", labels.shape)

    for col, name, df in ((id_col, "features", feats),
                          (args.feat_time_col, "features", feats),
                          (id_col, "labels", labels),
                          (args.label_time_col, "labels", labels)):
        if col not in df.columns:
            raise ValueError("join key '{0}' missing from {1} file".format(col, name))

    # All four label columns are required: the three funnel flags + credit_1v1
    # (the qualification V2 label is derived from it).
    required_labels = TARGET_COLS + RAW_LABEL_COLS
    missing_targets = [c for c in required_labels if c not in labels.columns]
    if missing_targets:
        raise ValueError("label file is missing required label columns: {0}".format(missing_targets))
    logger.info("  label columns found: %s", required_labels)

    # Normalize both date columns to int yyyymmdd 'dt', then join on (id, dt).
    feats["dt"] = _to_yyyymmdd_int(feats[args.feat_time_col])
    labels["dt"] = _to_yyyymmdd_int(labels[args.label_time_col])
    join_keys = [id_col, "dt"]

    for name, df in (("features", feats), ("labels", labels)):
        dup = int(df.duplicated(subset=join_keys).sum())
        if dup:
            logger.warning("%s file has %d duplicate (%s) keys", name, dup, ", ".join(join_keys))

    feature_cols = [c for c in feats.columns if c not in (id_col, args.feat_time_col, "dt")]
    logger.info("  %d feature columns detected", len(feature_cols))

    # Keep only join keys + label columns from the labels side.
    labels_slim = labels[join_keys + required_labels]
    feats_slim = feats[[id_col, "dt"] + feature_cols]

    merged = feats_slim.merge(labels_slim, on=join_keys, how=args.join)
    matched = int(merged[TARGET_COLS[0]].notna().sum()) if args.join == "left" else len(merged)
    logger.info("Join (%s) on (%s): %d feat rows, %d label rows -> %d merged (%d with labels)",
                args.join, ", ".join(join_keys), len(feats), len(labels), len(merged), matched)
    if len(merged) == 0:
        raise RuntimeError("Join produced 0 rows. Check that id / dt match across files.")

    # Binarize the three funnel flags; derive the qualification V2 label.
    for col in TARGET_COLS:
        merged[col] = _binarize(merged[col])
    merged[CREDIT_1V1_FLAG] = _binarize_credit_1v1(merged["credit_1v1"])
    merged["credit_1v1"] = pd.to_numeric(merged["credit_1v1"], errors="coerce")

    # Downcast feature float64 -> float32 to shrink the CSVs.
    for c in feature_cols:
        if merged[c].dtype == np.float64:
            merged[c] = merged[c].astype(np.float32)

    n_dt_nan = int(merged["dt"].isna().sum())
    if n_dt_nan:
        logger.warning("%d merged rows have NaN dt; time-split sorts them last.", n_dt_nan)

    ordered = [id_col, "dt"] + feature_cols + TARGET_COLS + [CREDIT_1V1_FLAG] + RAW_LABEL_COLS
    merged = merged[ordered]

    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 64)
    print("Modeling tables (merged base: {0} rows x {1} cols, {2} features)".format(
        len(merged), merged.shape[1], len(feature_cols)))
    print("dt range: {0} .. {1}".format(
        int(merged["dt"].min()) if merged["dt"].notna().any() else "NA",
        int(merged["dt"].max()) if merged["dt"].notna().any() else "NA"))
    print("-" * 64)

    for suffix, cond, serves in MODELING_TABLES:
        if cond is None:
            sub = merged
            pop_desc = "all rows"
        else:
            sub = merged[merged[cond] == 1]
            pop_desc = "{0} == 1".format(cond)
        out_path = out_dir / "xc_{0}.csv".format(suffix)
        sub.to_csv(out_path, index=False)
        print("xc_{0:<12} pop={1:<22} n={2:<8} -> {3}".format(
            suffix, pop_desc, len(sub), serves))
        logger.info("Wrote %s (%d rows)", out_path, len(sub))

    print("-" * 64)
    print("Analysis funnels (full process, conversion only — not modeled):")
    n_total = len(merged)
    for funnel in ANALYSIS_FUNNELS:
        print("  endpoint = {0}".format(funnel[-1]))
        pop = merged
        print("    {0:<16} n={1}".format("population", n_total))
        for label in funnel:
            n_prev = len(pop)
            pop = pop[pop[label] == 1]
            n_pass = len(pop)
            step_rate = (n_pass / n_prev) if n_prev else 0.0
            cum_rate = (n_pass / n_total) if n_total else 0.0
            print("    {0:<16} n={1:<8} step_cvr={2:.4f}  cum_cvr={3:.4f}".format(
                label, n_pass, step_rate, cum_rate))

    print("=" * 64)
    print("Next (Stage-1 then Stage-2 for each of the 3 models):")
    for prod in PRODUCTS:
        print("  PYTHONPATH=src python3 scripts/run_analysis.py --product {0}".format(prod))
    print("Then evaluate the FULL analysis funnel with the fused score:")
    print("  PYTHONPATH=src python3 scripts/run_funnel_eval.py \\")
    print("      --resp-product xc_resp_finish --resp-run-id <id> \\")
    print("      --qual-product xc_qual_finish --qual-run-id <id>")
    print("  (qualification V2: --qual-product xc_qual_finish_1v1 --qual-stage is_credit_1v1)")


if __name__ == "__main__":
    main()
