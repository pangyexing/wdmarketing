"""Build a wide Home Credit training table for wdmarketing.

Aggregates bureau / previous_application / installments / POS_CASH /
credit_card_balance onto the main application_train table, keyed on
SK_ID_CURR. Each related table is sliced into time windows
(30D / 90D / 180D / 360D / ALL) based on its days/months column, and each
numeric column is aggregated with a small set of functions. Feature names
are constructed as `<table>_<col>_<agg>_<window>` in lowercase, matching
the wdm default window_pattern regex.

Output: data/home_credit_wide.csv — one row per SK_ID_CURR, ~900 columns
(depends on which tables are present and aggregation choices).

Usage:
    PYTHONPATH=src python3 scripts/build_home_credit_wide.py
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

WINDOWS_DAYS = [30, 90, 180, 360, 9999]  # 9999 ~ "all"
WINDOW_LABELS = ["30d", "90d", "180d", "360d", "all"]


def _window_label(days):
    return WINDOW_LABELS[WINDOWS_DAYS.index(days)]


def _label_encode_categoricals(df):
    """Simple label encoding (not ideal for high-cardinality but fine here).

    Returns (encoded_df, list of columns encoded).
    """
    encoded = []
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category").cat.codes.astype(np.int32)
        encoded.append(c)
    return df, encoded


def _aggregate_by_window(sub_df, id_col, days_col, value_cols, aggs,
                        table_prefix, numeric_days_threshold=0):
    """Aggregate sub_df over time windows defined by (days_col <= threshold).

    sub_df rows whose days_col is missing are treated as "all-time only".
    Returns a DataFrame indexed by id_col with columns named
    `<table_prefix>_<col>_<agg>_<window>`.
    """
    out_frames = []
    # Coerce the days column to integer (some tables use int, some float with NaN)
    s = pd.to_numeric(sub_df[days_col], errors="coerce")
    abs_days = np.abs(s)  # HC convention: days are negative, more negative = older

    for days_cap in WINDOWS_DAYS:
        label = _window_label(days_cap)
        if days_cap == 9999:
            mask = np.ones(len(sub_df), dtype=bool)
        else:
            mask = abs_days.values <= days_cap
            # Treat NaN days as excluded from any finite window
            mask = mask & ~np.isnan(abs_days.values)
        if mask.sum() == 0:
            continue
        block = sub_df.loc[mask, [id_col] + value_cols]
        agg_df = block.groupby(id_col)[value_cols].agg(aggs)
        # Flatten multi-index columns: ('amt_credit', 'mean') → 'bureau_amt_credit_mean_30d'
        agg_df.columns = [
            "{0}_{1}_{2}_{3}".format(table_prefix, c.lower(), a, label)
            for c, a in agg_df.columns
        ]
        # Plus a count-per-window column
        count_df = block.groupby(id_col).size().to_frame(
            "{0}_cnt_{1}".format(table_prefix, label))
        out_frames.append(pd.concat([agg_df, count_df], axis=1))

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, axis=1)


def build_main(path):
    logger.info("Loading application_train.csv")
    df = pd.read_csv(path / "application_train.csv")
    logger.info("  shape: %s", df.shape)
    df, _ = _label_encode_categoricals(df)
    # Recode the well-known 365243 sentinel for DAYS_EMPLOYED to NaN
    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    # Lowercase columns so final wide has consistent casing (except SK_ID_CURR, TARGET stay)
    keep = {"SK_ID_CURR", "TARGET"}
    rename = {c: c.lower() for c in df.columns if c not in keep}
    df.rename(columns=rename, inplace=True)
    return df


def build_bureau(path):
    logger.info("Loading bureau.csv")
    bureau = pd.read_csv(path / "bureau.csv")
    logger.info("  shape: %s", bureau.shape)
    bureau, _ = _label_encode_categoricals(bureau)
    numeric_cols = [
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
    ]
    numeric_cols = [c for c in numeric_cols if c in bureau.columns]
    wide = _aggregate_by_window(
        bureau, id_col="SK_ID_CURR", days_col="DAYS_CREDIT",
        value_cols=numeric_cols, aggs=["mean", "max", "sum"],
        table_prefix="bureau")
    logger.info("  bureau wide shape: %s", wide.shape)
    return wide


def build_prev(path):
    logger.info("Loading previous_application.csv")
    prev = pd.read_csv(path / "previous_application.csv")
    logger.info("  shape: %s", prev.shape)
    prev, _ = _label_encode_categoricals(prev)
    numeric_cols = [
        "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE", "HOUR_APPR_PROCESS_START", "RATE_DOWN_PAYMENT",
        "DAYS_DECISION", "CNT_PAYMENT", "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    numeric_cols = [c for c in numeric_cols if c in prev.columns]
    # Recode 365243 sentinel across DAYS_* columns
    for c in [col for col in numeric_cols if col.startswith("DAYS_")]:
        prev.loc[prev[c] == 365243, c] = np.nan
    wide = _aggregate_by_window(
        prev, id_col="SK_ID_CURR", days_col="DAYS_DECISION",
        value_cols=numeric_cols, aggs=["mean", "max", "sum"],
        table_prefix="prev")
    logger.info("  prev wide shape: %s", wide.shape)
    return wide


def build_installments(path):
    logger.info("Loading installments_payments.csv")
    ins = pd.read_csv(path / "installments_payments.csv")
    logger.info("  shape: %s", ins.shape)
    # Derived delta features
    ins["AMT_UNDERPAY"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DAYS_LATE"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    numeric_cols = ["AMT_INSTALMENT", "AMT_PAYMENT", "AMT_UNDERPAY",
                    "DAYS_LATE", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
    numeric_cols = [c for c in numeric_cols if c in ins.columns]
    wide = _aggregate_by_window(
        ins, id_col="SK_ID_CURR", days_col="DAYS_INSTALMENT",
        value_cols=numeric_cols, aggs=["mean", "max", "sum"],
        table_prefix="instal")
    logger.info("  installments wide shape: %s", wide.shape)
    return wide


def build_pos_cash(path):
    logger.info("Loading POS_CASH_balance.csv")
    pos = pd.read_csv(path / "POS_CASH_balance.csv")
    logger.info("  shape: %s", pos.shape)
    pos, _ = _label_encode_categoricals(pos)
    # Convert months to days for window alignment
    pos["DAYS_APPROX"] = pos["MONTHS_BALANCE"] * 30
    numeric_cols = ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
    numeric_cols = [c for c in numeric_cols if c in pos.columns]
    wide = _aggregate_by_window(
        pos, id_col="SK_ID_CURR", days_col="DAYS_APPROX",
        value_cols=numeric_cols, aggs=["mean", "max", "sum"],
        table_prefix="pos")
    logger.info("  POS_CASH wide shape: %s", wide.shape)
    return wide


def build_credit_card(path):
    logger.info("Loading credit_card_balance.csv")
    cc = pd.read_csv(path / "credit_card_balance.csv")
    logger.info("  shape: %s", cc.shape)
    cc, _ = _label_encode_categoricals(cc)
    cc["DAYS_APPROX"] = cc["MONTHS_BALANCE"] * 30
    numeric_cols = [
        "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "SK_DPD", "SK_DPD_DEF",
    ]
    numeric_cols = [c for c in numeric_cols if c in cc.columns]
    wide = _aggregate_by_window(
        cc, id_col="SK_ID_CURR", days_col="DAYS_APPROX",
        value_cols=numeric_cols, aggs=["mean", "max", "sum"],
        table_prefix="cc")
    logger.info("  credit_card wide shape: %s", wide.shape)
    return wide


def main():
    ap = argparse.ArgumentParser(description="Build the Home Credit wide table.")
    ap.add_argument("--data-dir",
                    default="data/home-credit-default-risk")
    ap.add_argument("--out",
                    default="data/home_credit_wide.csv")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="Dev option: subsample to N customers for a quick smoke test.")
    args = ap.parse_args()

    setup_logging()
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / args.data_dir
    out_path = repo / args.out

    main_df = build_main(data_dir)
    if args.limit_rows:
        main_df = main_df.sample(args.limit_rows, random_state=0).reset_index(drop=True)
        logger.info("Subsampled main to %d rows for smoke test", len(main_df))

    merged = main_df.set_index("SK_ID_CURR")

    for build_fn in (build_bureau, build_prev, build_installments,
                     build_pos_cash, build_credit_card):
        w = build_fn(data_dir)
        if w.empty:
            continue
        # Filter to customers in main_df to reduce memory
        w = w.reindex(merged.index)
        merged = pd.concat([merged, w], axis=1)
        logger.info("Merged %s → total columns: %d", build_fn.__name__, len(merged.columns))

    merged = merged.reset_index()

    # Synthesize a yyyymmdd time column from SK_ID_CURR ordering.
    # Home Credit has no absolute application timestamp — only relative DAYS_*
    # offsets (each row's 0 = its own application day). SK_ID_CURR is a roughly
    # monotonic application id, so we use its rank as a chronological proxy
    # and spread rows evenly across 2022-01-01 → 2024-12-31.
    rank = merged["SK_ID_CURR"].rank(method="first").astype(np.int64) - 1
    span_days = (pd.Timestamp("2024-12-31") - pd.Timestamp("2022-01-01")).days
    denom = max(1, int(rank.max()))
    day_offset = (rank * span_days // denom).astype(np.int64)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(day_offset, unit="D")
    merged["yyyymmdd"] = dates.dt.strftime("%Y%m%d").astype(np.int64)
    logger.info("Synthesized yyyymmdd range: %d → %d",
                merged["yyyymmdd"].min(), merged["yyyymmdd"].max())

    # Downcast float64 to float32 where possible for smaller CSV
    for c in merged.select_dtypes(include=["float64"]).columns:
        merged[c] = merged[c].astype(np.float32)

    logger.info("Writing wide table: %s (shape=%s)", out_path, merged.shape)
    merged.to_csv(out_path, index=False)
    print("Done. Wide table: {0}".format(out_path))
    print("Shape: {0} rows × {1} columns".format(*merged.shape))
    print("Target pos rate: {0:.4f}".format(merged["TARGET"].mean()))
    print("yyyymmdd range: {0} → {1}".format(
        merged["yyyymmdd"].min(), merged["yyyymmdd"].max()))


if __name__ == "__main__":
    main()
