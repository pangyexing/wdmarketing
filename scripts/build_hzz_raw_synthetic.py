"""Generate synthetic raw hzz data for testing scripts/preprocess_hzz_raw.py.

Output layout (matches configs/preprocess/hzz_raw.yaml defaults):
    <out_dir>/day/hzz_d_feat{row}{table}.csv     (12 files: 3 row chunks × 4 tables)
    <out_dir>/mon/hzz_m_feat{table}.csv          (4 files: 1 per table)

Exercises every tricky preprocess path:
- Per-table time-column aliases       — t1/t4: cd_time, t2: dt, t3: statis_date.
- Per-table extra "redundant" columns — day_t1 has label_credit, day_t3 has
                                        label_credit_level, mon_t2 has label_credit.
- yyyy-mm-dd string cd_time values    — exercises the int conversion path.
- ~5% duplicate rows                  — same cust_id, different cd_time, identical
                                        feature values; dedup should collapse them.

Usage:
    PYTHONPATH=src python3 scripts/build_hzz_raw_synthetic.py --rows 600
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from wdm.utils.logging import setup_logging
except Exception:  # pragma: no cover
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

logger = logging.getLogger(__name__)


TABLE_FEATURES = {
    1: ["t1_amt_30d", "t1_cnt_7d", "t1_ratio"],
    2: ["t2_balance", "t2_amt_total"],
    3: ["t3_diff_days", "t3_flag_a"],
    4: ["t4_score"],
}
TABLE_TIME_ALIAS = {1: "cd_time", 2: "dt", 3: "statis_date", 4: "cd_time"}
TABLE_EXTRA_DROP_COLS = {
    ("day", 1): ["label_credit"],
    ("day", 3): ["label_credit_level"],
    ("mon", 2): ["label_credit"],
}


def gen_master(rows: int, seed: int, snapshots_per_id: int,
               start: str, span_days: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(start)
    cust_ids = np.arange(10_000_000, 10_000_000 + rows, dtype=np.int64)
    rep_ids = np.repeat(cust_ids, snapshots_per_id)
    n = rep_ids.size

    day_off = rng.integers(0, span_days, n)
    cd_time = (base + pd.to_timedelta(day_off.astype(np.int64), unit="D")
               ).strftime("%Y-%m-%d")

    df = pd.DataFrame({"cust_id": rep_ids, "cd_time": cd_time})
    df["label_register"] = (rng.random(n) < 0.07).astype(np.int8)
    for cols in TABLE_FEATURES.values():
        for c in cols:
            df[c] = rng.normal(size=n).astype(np.float32)
    return df


def make_duplicates(df: pd.DataFrame, dup_frac: float, seed: int,
                    span_days: int, start: str) -> pd.DataFrame:
    """Append `dup_frac × N` rows that copy existing rows but with a fresh
    cd_time. After dedup (ignore_time=True), they should collapse."""
    if dup_frac <= 0:
        return df
    n_dup = int(len(df) * dup_frac)
    if n_dup == 0:
        return df
    rng = np.random.default_rng(seed)
    src = df.sample(n=n_dup, random_state=seed).reset_index(drop=True)
    base = pd.Timestamp(start)
    new_off = rng.integers(0, span_days, n_dup)
    src["cd_time"] = (base + pd.to_timedelta(new_off.astype(np.int64), unit="D")
                      ).strftime("%Y-%m-%d")
    return pd.concat([df, src], ignore_index=True)


def split_rows_by_id(df: pd.DataFrame, n_chunks: int) -> list:
    """Partition rows by cust_id into n_chunks ~equal id-ranges. Same cust_id
    always lands in the same chunk (so per-id dedup is local to one chunk)."""
    if n_chunks == 1:
        return [df]
    ids = np.sort(df["cust_id"].unique())
    edges = np.array_split(ids, n_chunks)
    return [df.loc[df["cust_id"].isin(e)].reset_index(drop=True) for e in edges]


def write_table_files(master: pd.DataFrame, out_dir: Path, gran: str,
                      n_row_chunks: int, rng: np.random.Generator) -> None:
    gran_short = "d" if gran == "day" else "m"
    for table, cols in TABLE_FEATURES.items():
        sub = master[["cust_id", "cd_time", "label_register"] + cols].copy()

        # Per-table extra "redundant" columns the preprocess should drop.
        for extra in TABLE_EXTRA_DROP_COLS.get((gran, table), []):
            sub[extra] = rng.normal(size=len(sub)).astype(np.float32)

        alias = TABLE_TIME_ALIAS[table]
        if alias != "cd_time":
            sub = sub.rename(columns={"cd_time": alias})

        chunks = split_rows_by_id(sub, n_row_chunks)
        for i, chunk in enumerate(chunks, start=1):
            if n_row_chunks == 1:
                fname = "hzz_{0}_feat{1}.csv".format(gran_short, table)
            else:
                fname = "hzz_{0}_feat{1}{2}.csv".format(gran_short, i, table)
            path = out_dir / gran / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_csv(path, index=False)
            logger.info("wrote %s  (%d rows, %d cols)", path, len(chunk),
                        chunk.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600,
                    help="number of unique cust_ids")
    ap.add_argument("--snapshots-per-id-day", type=int, default=2)
    ap.add_argument("--snapshots-per-id-mon", type=int, default=2)
    ap.add_argument("--out-dir", default="data/hzz_raw_synth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--span-days", type=int, default=180)
    ap.add_argument("--dup-frac", type=float, default=0.05,
                    help="fraction of rows duplicated with a different cd_time "
                         "(after dedup, should collapse back)")
    args = ap.parse_args()

    setup_logging()
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir

    rng = np.random.default_rng(args.seed)

    master_day = gen_master(args.rows, args.seed,
                            args.snapshots_per_id_day,
                            args.start_date, args.span_days)
    master_day = make_duplicates(master_day, args.dup_frac, args.seed + 1,
                                 args.span_days, args.start_date)
    write_table_files(master_day, out_dir, "day", n_row_chunks=3, rng=rng)

    master_mon = gen_master(args.rows, args.seed + 100,
                            args.snapshots_per_id_mon,
                            args.start_date, args.span_days)
    master_mon = make_duplicates(master_mon, args.dup_frac, args.seed + 2,
                                 args.span_days, args.start_date)
    write_table_files(master_mon, out_dir, "mon", n_row_chunks=1, rng=rng)

    print("Done. Output: {0}".format(out_dir))


if __name__ == "__main__":
    main()
