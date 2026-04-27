"""End-to-end tests for scripts/preprocess_hzz_raw.py.

Each test builds a tiny raw-data layout under tmp_path, runs the pipeline,
and inspects the per-table outputs (and merged.csv when relevant).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load the script as a module — scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preprocess_hzz_raw as P  # noqa: E402


# ---------- helpers ------------------------------------------------------ #

def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _base_cfg(in_dir: Path, out_dir: Path, n_tables: int = 2) -> dict:
    return {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "sources": {
            "day": {
                "row_chunks": 3,
                "feature_tables": n_tables,
                "file_pattern": "hzz_d_feat{row}{table}.csv",
                "per_table_out": "day/feat_t{table}.csv",
                "merged_out": "day/merged.csv",
            },
            "mon": {
                "row_chunks": 1,
                "feature_tables": n_tables,
                "file_pattern": "hzz_m_feat{table}.csv",
                "per_table_out": "mon/feat_t{table}.csv",
                "merged_out": "mon/merged.csv",
            },
        },
        "id_col": "cust_id",
        "time_col": "cd_time",
        "time_col_aliases": ["cd_time", "dt", "statis_date"],
        "label_col": "label_register",
        "drop_cols": {"common": [], "per_table": {
            "day": {"t1": [], "t2": [], "t3": [], "t4": []},
            "mon": {"t1": [], "t2": [], "t3": [], "t4": []},
        }},
        "dedup": {"enabled": True, "ignore_time": True, "keep": "min_time"},
        "read_chunksize": 100,
        "shard_count": 4,
        "merge": {"enabled": False, "how": "outer"},
    }


def _make_day_master(n_ids: int = 6, dates=("2024-01-15", "2024-02-20")):
    """Master DF with one row per (cust_id, cd_time) and a stable label/feature
    grid. Each table picks columns by name."""
    cust_ids = list(range(10_000_001, 10_000_001 + n_ids))
    rows = []
    for cid in cust_ids:
        for d in dates:
            rows.append({"cust_id": cid, "cd_time": d,
                         "label_register": cid % 2})
    df = pd.DataFrame(rows)
    df["t1_a"] = np.arange(len(df), dtype=np.float32) + 0.1
    df["t1_b"] = np.arange(len(df), dtype=np.float32) + 1.1
    df["t2_x"] = np.arange(len(df), dtype=np.float32) + 2.1
    df["t2_y"] = np.arange(len(df), dtype=np.float32) + 3.1
    return df, cust_ids


def _write_day_files(in_dir: Path, master: pd.DataFrame, cust_ids,
                     table_to_cols, n_chunks=3, time_alias_per_table=None,
                     extra_cols_per_table=None) -> None:
    """Split master into n_chunks by id-range and one file per (chunk, table)."""
    time_alias_per_table = time_alias_per_table or {}
    extra_cols_per_table = extra_cols_per_table or {}
    chunks = np.array_split(np.array(cust_ids), n_chunks)
    for table, cols in table_to_cols.items():
        keep = ["cust_id", "cd_time", "label_register"] + list(cols)
        sub = master[keep].copy()
        for c, vals in extra_cols_per_table.get(table, {}).items():
            sub[c] = vals
        alias = time_alias_per_table.get(table, "cd_time")
        if alias != "cd_time":
            sub = sub.rename(columns={"cd_time": alias})
        for ri, ids in enumerate(chunks, start=1):
            chunk_df = sub.loc[sub["cust_id"].isin(list(ids))].reset_index(drop=True)
            path = in_dir / "day" / "hzz_d_feat{0}{1}.csv".format(ri, table)
            _write(path, chunk_df)


def _write_mon_files(in_dir: Path, master: pd.DataFrame,
                     table_to_cols, time_alias_per_table=None,
                     extra_cols_per_table=None) -> None:
    time_alias_per_table = time_alias_per_table or {}
    extra_cols_per_table = extra_cols_per_table or {}
    for table, cols in table_to_cols.items():
        keep = ["cust_id", "cd_time", "label_register"] + list(cols)
        sub = master[keep].copy()
        for c, vals in extra_cols_per_table.get(table, {}).items():
            sub[c] = vals
        alias = time_alias_per_table.get(table, "cd_time")
        if alias != "cd_time":
            sub = sub.rename(columns={"cd_time": alias})
        path = in_dir / "mon" / "hzz_m_feat{0}.csv".format(table)
        _write(path, sub)


# ---------- tests -------------------------------------------------------- #

def test_per_table_concat_day(tmp_path):
    """12-file day layout (here 6 = 2 tables × 3 chunks): rows preserved
    (no duplicates), cd_time int, no per-table drops applied."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    master, cids = _make_day_master()
    _write_day_files(in_dir, master, cids,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    P.run(cfg, tmp_path, ["day"])

    t1 = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    t2 = pd.read_csv(out_dir / "day" / "feat_t2.csv")
    assert len(t1) == len(master)
    assert len(t2) == len(master)
    assert t1["cd_time"].dtype == np.int64
    assert set(t1.columns) == {"cust_id", "cd_time", "label_register", "t1_a", "t1_b"}
    assert set(t2.columns) == {"cust_id", "cd_time", "label_register", "t2_x", "t2_y"}
    # cd_time should match yyyymmdd ints for the two dates we used.
    assert set(t1["cd_time"].unique()) == {20240115, 20240220}


def test_per_table_concat_mon(tmp_path):
    """mon layout: 1 file per table → 1 output per table."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    master, _ = _make_day_master()
    _write_mon_files(in_dir, master,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    P.run(cfg, tmp_path, ["mon"])

    t1 = pd.read_csv(out_dir / "mon" / "feat_t1.csv")
    assert len(t1) == len(master)
    assert t1["cd_time"].dtype == np.int64


def test_per_table_drop_independent(tmp_path):
    """day_t1 drops [extra_a]; day_t2 drops [extra_b]; common drops [common_z]."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    cfg["drop_cols"] = {
        "common": ["common_z"],
        "per_table": {
            "day": {"t1": ["extra_a"], "t2": ["extra_b"]},
            "mon": {"t1": [], "t2": []},
        },
    }
    master, cids = _make_day_master()
    n = len(master)
    _write_day_files(
        in_dir, master, cids,
        {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]},
        extra_cols_per_table={
            1: {"extra_a": np.arange(n), "common_z": np.arange(n)},
            2: {"extra_b": np.arange(n), "common_z": np.arange(n)},
        },
    )

    P.run(cfg, tmp_path, ["day"])

    t1 = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    t2 = pd.read_csv(out_dir / "day" / "feat_t2.csv")
    assert "extra_a" not in t1.columns
    assert "common_z" not in t1.columns
    assert "extra_a" not in t2.columns       # never existed in t2 → no-op (silent)
    assert "extra_b" not in t2.columns
    assert "common_z" not in t2.columns
    # day_t1's per-table drop should NOT remove t2-only column from t2:
    assert "t2_x" in t2.columns
    assert "t1_a" in t1.columns


def test_header_mismatch_raises(tmp_path):
    """Renaming a column in row-chunk 2 of t1 must raise during Stage A."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    master, cids = _make_day_master()
    _write_day_files(in_dir, master, cids,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    # corrupt chunk 2 of table 1: rename t1_b → t1_b_RENAMED
    bad = pd.read_csv(in_dir / "day" / "hzz_d_feat21.csv")
    bad = bad.rename(columns={"t1_b": "t1_b_RENAMED"})
    bad.to_csv(in_dir / "day" / "hzz_d_feat21.csv", index=False)

    with pytest.raises(ValueError, match="header mismatch"):
        P.run(cfg, tmp_path, ["day"])


def test_time_alias_rename(tmp_path):
    """All 3 row-chunks of all tables use 'dt' (an alias) — output is 'cd_time'."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    master, cids = _make_day_master()
    _write_day_files(
        in_dir, master, cids,
        {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]},
        time_alias_per_table={1: "dt", 2: "dt"},
    )

    P.run(cfg, tmp_path, ["day"])

    t1 = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    assert "cd_time" in t1.columns
    assert "dt" not in t1.columns


def test_time_alias_cross_table(tmp_path):
    """Different time-column aliases across the 4 tables, all normalized."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir, n_tables=4)
    cfg["drop_cols"]["per_table"]["day"] = {"t1": [], "t2": [], "t3": [], "t4": []}
    master, cids = _make_day_master()
    master["t3_p"] = master["t1_a"] + 100
    master["t4_q"] = master["t1_a"] + 200
    _write_day_files(
        in_dir, master, cids,
        {1: ["t1_a"], 2: ["t2_x"], 3: ["t3_p"], 4: ["t4_q"]},
        time_alias_per_table={1: "cd_time", 2: "dt", 3: "statis_date", 4: "cd_time"},
    )

    P.run(cfg, tmp_path, ["day"])

    for t in (1, 2, 3, 4):
        df = pd.read_csv(out_dir / "day" / "feat_t{0}.csv".format(t))
        assert "cd_time" in df.columns
        assert "dt" not in df.columns
        assert "statis_date" not in df.columns


def test_dedup_within_shard(tmp_path):
    """Same cust_id × different cd_time × identical features → collapse to 1
    row (keep=min_time keeps earliest cd_time)."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir, n_tables=1)
    cfg["drop_cols"]["per_table"]["day"] = {"t1": []}
    cfg["sources"]["day"]["row_chunks"] = 1
    cfg["sources"]["day"]["file_pattern"] = "hzz_d_feat{table}.csv"

    # 1 cust_id, 3 cd_times, all features identical → after dedup keep 1.
    df = pd.DataFrame({
        "cust_id": [10000001, 10000001, 10000001, 10000002, 10000002],
        "cd_time": ["2024-03-15", "2024-01-10", "2024-02-20",
                    "2024-04-05", "2024-04-05"],
        "label_register": [1, 1, 1, 0, 0],
        "t1_a": [5.5, 5.5, 5.5, 7.7, 7.7],
    })
    _write(in_dir / "day" / "hzz_d_feat1.csv", df)

    P.run(cfg, tmp_path, ["day"])

    out = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    # cust_id 10000001 dedups to 1 row (3→1); cust_id 10000002 dedups (2→1).
    assert len(out) == 2
    row1 = out[out["cust_id"] == 10000001].iloc[0]
    assert int(row1["cd_time"]) == 20240110  # min_time
    row2 = out[out["cust_id"] == 10000002].iloc[0]
    assert int(row2["cd_time"]) == 20240405


def test_dedup_disabled(tmp_path):
    """With dedup.enabled=False, every input row survives."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir, n_tables=1)
    cfg["drop_cols"]["per_table"]["day"] = {"t1": []}
    cfg["sources"]["day"]["row_chunks"] = 1
    cfg["sources"]["day"]["file_pattern"] = "hzz_d_feat{table}.csv"
    cfg["dedup"]["enabled"] = False

    df = pd.DataFrame({
        "cust_id": [10000001, 10000001, 10000001],
        "cd_time": ["2024-03-15", "2024-01-10", "2024-02-20"],
        "label_register": [1, 1, 1],
        "t1_a": [5.5, 5.5, 5.5],
    })
    _write(in_dir / "day" / "hzz_d_feat1.csv", df)

    P.run(cfg, tmp_path, ["day"])

    out = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    assert len(out) == 3


def test_merge_outer(tmp_path):
    """Full pipeline + horizontal merge: merged has 1 label_register and
    correctly-aligned features."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    cfg["merge"]["enabled"] = True
    master, cids = _make_day_master()
    _write_day_files(in_dir, master, cids,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    P.run(cfg, tmp_path, ["day"])

    merged = pd.read_csv(out_dir / "day" / "merged.csv")
    assert len(merged) == len(master)        # all (id, time) pairs match across t1/t2
    # exactly one label_register column
    assert sum(1 for c in merged.columns if c == "label_register") == 1
    # all features present
    assert set(merged.columns) >= {"cust_id", "cd_time", "label_register",
                                   "t1_a", "t1_b", "t2_x", "t2_y"}
    # spot-check alignment for one (id, time)
    one = merged[(merged["cust_id"] == 10000001)
                 & (merged["cd_time"] == 20240115)].iloc[0]
    src = master[(master["cust_id"] == 10000001)
                 & (master["cd_time"] == "2024-01-15")].iloc[0]
    for col in ("t1_a", "t1_b", "t2_x", "t2_y"):
        assert one[col] == pytest.approx(src[col])


def test_per_table_files_preserved_after_merge(tmp_path):
    """Stage C must NOT delete or modify the per-table outputs."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    cfg["merge"]["enabled"] = True
    master, cids = _make_day_master()
    _write_day_files(in_dir, master, cids,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    P.run(cfg, tmp_path, ["day"])

    t1_path = out_dir / "day" / "feat_t1.csv"
    t2_path = out_dir / "day" / "feat_t2.csv"
    assert t1_path.is_file() and t2_path.is_file()

    snap_t1 = t1_path.read_bytes()
    snap_t2 = t2_path.read_bytes()
    # Re-run merge separately (would re-trigger Stage B + C) and check that the
    # final per-table contents are identical (idempotent).
    P.run(cfg, tmp_path, ["day"])
    assert t1_path.read_bytes() == snap_t1
    assert t2_path.read_bytes() == snap_t2


def test_cd_time_conversion(tmp_path):
    """yyyy-mm-dd string → int yyyymmdd."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir, n_tables=1)
    cfg["drop_cols"]["per_table"]["day"] = {"t1": []}
    cfg["sources"]["day"]["row_chunks"] = 1
    cfg["sources"]["day"]["file_pattern"] = "hzz_d_feat{table}.csv"

    df = pd.DataFrame({
        "cust_id": [10000001, 10000002, 10000003],
        "cd_time": ["2024-03-15", "2024-12-31", "2024-01-01"],
        "label_register": [0, 1, 0],
        "t1_a": [1.0, 2.0, 3.0],
    })
    _write(in_dir / "day" / "hzz_d_feat1.csv", df)

    P.run(cfg, tmp_path, ["day"])

    out = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    assert out["cd_time"].dtype == np.int64
    assert sorted(out["cd_time"].tolist()) == [20240101, 20240315, 20241231]


def test_drop_missing_col_silent(tmp_path):
    """Listing a column that doesn't exist in any input must not raise."""
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cfg = _base_cfg(in_dir, out_dir)
    cfg["drop_cols"]["common"] = ["nonexistent_col_x", "another_missing"]
    master, cids = _make_day_master()
    _write_day_files(in_dir, master, cids,
                     {1: ["t1_a", "t1_b"], 2: ["t2_x", "t2_y"]})

    P.run(cfg, tmp_path, ["day"])  # should not raise

    t1 = pd.read_csv(out_dir / "day" / "feat_t1.csv")
    assert "t1_a" in t1.columns
    assert "nonexistent_col_x" not in t1.columns
