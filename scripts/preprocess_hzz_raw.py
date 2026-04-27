"""Preprocess raw hzz day/mon CSV files into per-table + optional merged tables.

Layout:
- day  : 12 files (3 row-chunks × 4 feature-tables)  → 4 per-table outputs.
- mon  : 4 files  (1 per feature-table)              → 4 per-table outputs.

Stage A — discover + validate
    Enumerate all expected inputs by config pattern; raise on missing.
    Read headers; rename the time column to its canonical name via aliases
    (production tables disagree on this name); for day, require the 3 row
    chunks of each table to share an identical canonical header.

Stage B — per-table processing (id-hash sharded streaming)
    For each (gran, table):
      For shard k in [0, shard_count):
        Stream all input files in chunksize=read_chunksize chunks,
        rename time alias, filter rows where hash(cust_id) % S == k,
        accumulate into a shard buffer (bounded ≈ table_rows / S × n_cols).
        On the buffer:  cd_time → yyyymmdd int, drop configured columns,
        optional dedup ("ignore time column" => same-id same-features rows
        collapse), append to <output_dir>/<gran>/feat_t{table}.csv.
    Per-table outputs are persistent products — never deleted by Stage C.

Stage C — optional horizontal merge
    For each gran (when merge.enabled): id-shard the 4 per-table outputs,
    drop label_register from t2/t3/t4, sequentially merge on (cust_id,
    cd_time) with configured how, append to <gran>/merged.csv.

Hash consistency: Stage B and Stage C share `shard_of()`; same id always
lands in the same shard, which makes (a) dedup correct without a global
pass and (b) horizontal merge complete within each shard.

Usage:
    PYTHONPATH=src python3 scripts/preprocess_hzz_raw.py \\
        --config configs/preprocess/hzz_raw.yaml \\
        --source both --merge
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from wdm.utils.logging import setup_logging
except Exception:  # pragma: no cover — script-only fallback
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

logger = logging.getLogger(__name__)


# ---------- config ------------------------------------------------------- #

def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def repo_root_of(config_path: Path) -> Path:
    # configs/preprocess/hzz_raw.yaml → repo root is parents[2].
    # Robust fallback: walk up until we hit a dir containing 'src/'.
    p = config_path.resolve()
    for parent in [p.parents[2]] + list(p.parents):
        if (parent / "src").is_dir():
            return parent
    return p.parents[2]


# ---------- shared utilities --------------------------------------------- #

def shard_of(ids: pd.Series, shard_count: int) -> np.ndarray:
    """Stable hash → shard index. Same id always maps to same shard regardless
    of dtype, so dedup and horizontal merge can each operate one shard at a time."""
    h = pd.util.hash_pandas_object(ids, index=False).values  # uint64
    return (h % np.uint64(shard_count)).astype(np.int64)


def normalize_cd_time(series: pd.Series) -> pd.Series:
    """Coerce cd_time to int64 yyyymmdd. Accepts int yyyymmdd, float yyyymmdd
    (whole-number values, e.g. when pandas inferred a numeric column as float
    due to a missing chunk), str yyyy-mm-dd, or str yyyymmdd. Assumes no NaN."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype(np.int64)
    if pd.api.types.is_float_dtype(series):
        return series.astype(np.int64)
    return series.astype(str).str.replace("-", "", regex=False).astype(np.int64)


def header_alias_target(columns: Sequence[str], aliases: Sequence[str],
                        canonical: str, where: str) -> str:
    """Pick the alias hit in `columns`, warn on multiple, raise on none."""
    hits = [a for a in aliases if a in columns]
    if not hits:
        raise ValueError(
            "{0}: no time column matched aliases {1}; columns={2}".format(
                where, list(aliases), list(columns)))
    if len(hits) > 1:
        logger.warning("%s: multiple time aliases matched %s; using %s",
                       where, hits, hits[0])
    return hits[0]


def rename_time_alias_inplace(df: pd.DataFrame, aliases: Sequence[str],
                              canonical: str, where: str) -> pd.DataFrame:
    if canonical in df.columns:
        return df
    hit = header_alias_target(df.columns, aliases, canonical, where)
    return df.rename(columns={hit: canonical})


def canonical_header(path: Path, aliases: Sequence[str], canonical: str) -> List[str]:
    df0 = pd.read_csv(path, nrows=0)
    cols = list(df0.columns)
    hit = header_alias_target(cols, aliases, canonical, str(path))
    return [canonical if c == hit else c for c in cols]


# ---------- Stage A ------------------------------------------------------ #

def enumerate_inputs(cfg: dict, gran: str, repo: Path) -> Dict[int, List[Path]]:
    """Map {table -> [Path, ...]} per the gran's file_pattern."""
    src = cfg["sources"][gran]
    pattern = src["file_pattern"]
    n_rows = int(src["row_chunks"])
    n_tables = int(src["feature_tables"])
    base = repo / cfg["input_dir"] / gran
    out: Dict[int, List[Path]] = {}
    for table in range(1, n_tables + 1):
        paths: List[Path] = []
        if n_rows == 1:
            p = base / pattern.format(table=table)
            paths.append(p)
        else:
            for row in range(1, n_rows + 1):
                p = base / pattern.format(row=row, table=table)
                paths.append(p)
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(
                    "expected raw file missing: {0}".format(p))
        out[table] = paths
    return out


def validate_headers(inputs: Dict[int, List[Path]],
                     aliases: Sequence[str], canonical: str) -> None:
    """For each table, all input row-chunks must share an identical canonical
    header. Mon (1 file) trivially passes."""
    for table, paths in inputs.items():
        canon = [canonical_header(p, aliases, canonical) for p in paths]
        if len(canon) <= 1:
            continue
        ref = canon[0]
        for i, h in enumerate(canon[1:], start=1):
            if tuple(h) != tuple(ref):
                ref_set, h_set = set(ref), set(h)
                raise ValueError(
                    "header mismatch in table {tbl}: '{p0}' vs '{pi}'; "
                    "missing in chunk={miss}; extra in chunk={extra}; "
                    "order_changed={order}".format(
                        tbl=table, p0=paths[0], pi=paths[i],
                        miss=sorted(ref_set - h_set),
                        extra=sorted(h_set - ref_set),
                        order=tuple(h) != tuple(ref) and ref_set == h_set,
                    ))
        logger.info("table %d: header OK across %d input(s)", table, len(paths))


# ---------- Stage B ------------------------------------------------------ #

def _effective_drop_cols(cfg: dict, gran: str, table: int) -> List[str]:
    common = list((cfg.get("drop_cols") or {}).get("common") or [])
    per_table = (cfg.get("drop_cols") or {}).get("per_table") or {}
    extra = list((per_table.get(gran) or {}).get("t{0}".format(table)) or [])
    seen, out = set(), []
    for c in common + extra:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _dedup(df: pd.DataFrame, time_col: str, ignore_time: bool, keep: str) -> pd.DataFrame:
    if ignore_time:
        subset = [c for c in df.columns if c != time_col]
    else:
        subset = list(df.columns)
    if not subset:
        return df
    if keep == "min_time":
        df = df.sort_values(time_col, kind="mergesort", na_position="last")
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    if keep == "max_time":
        df = df.sort_values(time_col, ascending=False, kind="mergesort", na_position="last")
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    if keep in ("first", "last"):
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    raise ValueError("unknown dedup.keep: {0}".format(keep))


def process_per_table(input_files: Sequence[Path], out_path: Path,
                      cfg: dict, gran: str, table: int) -> int:
    """Stage B for one (gran, table). Returns rows written."""
    id_col = cfg["id_col"]
    time_col = cfg["time_col"]
    aliases = cfg["time_col_aliases"]
    chunksize = int(cfg.get("read_chunksize", 100_000))
    shard_count = int(cfg.get("shard_count", 8))
    drop_cols = _effective_drop_cols(cfg, gran, table)
    dedup_cfg = cfg.get("dedup") or {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    header_written = False
    rows_written = 0

    for k in range(shard_count):
        shard_buf: List[pd.DataFrame] = []
        for in_path in input_files:
            for chunk in pd.read_csv(in_path, chunksize=chunksize):
                chunk = rename_time_alias_inplace(chunk, aliases, time_col, str(in_path))
                if id_col not in chunk.columns:
                    raise ValueError(
                        "{0}: id column '{1}' not in {2}".format(
                            in_path, id_col, list(chunk.columns)))
                mask = shard_of(chunk[id_col], shard_count) == k
                if mask.any():
                    shard_buf.append(chunk.loc[mask].copy())
        if not shard_buf:
            continue
        df = pd.concat(shard_buf, ignore_index=True)
        del shard_buf

        # cd_time → int yyyymmdd
        if df[time_col].isna().any():
            n_na = int(df[time_col].isna().sum())
            raise ValueError("{0}/t{1} shard {2}: {3} rows have NaN {4}".format(
                gran, table, k, n_na, time_col))
        df[time_col] = normalize_cd_time(df[time_col])

        # drop configured columns (silent on missing)
        drop_now = [c for c in drop_cols if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)

        # dedup
        if dedup_cfg.get("enabled", True):
            n_before = len(df)
            df = _dedup(df, time_col,
                        bool(dedup_cfg.get("ignore_time", True)),
                        str(dedup_cfg.get("keep", "min_time")))
            if n_before != len(df):
                logger.info("%s/t%d shard %d: dedup %d → %d",
                            gran, table, k, n_before, len(df))

        df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        rows_written += len(df)
        logger.info("%s/t%d shard %d/%d: wrote %d rows",
                    gran, table, k + 1, shard_count, len(df))

    if not header_written:
        # No data at all: still emit a header-only file so downstream merge works.
        canon = canonical_header(input_files[0], aliases, time_col)
        # apply drops to canon header
        canon = [c for c in canon if c not in set(drop_cols)]
        pd.DataFrame(columns=canon).to_csv(out_path, index=False)
        logger.warning("%s/t%d: no rows in any shard, wrote header-only file",
                       gran, table)
    logger.info("%s/t%d → %s (%d rows)", gran, table, out_path, rows_written)
    return rows_written


# ---------- Stage C ------------------------------------------------------ #

def merge_horizontal(per_table_paths: Sequence[Path], merged_path: Path,
                     cfg: dict) -> int:
    id_col = cfg["id_col"]
    time_col = cfg["time_col"]
    label_col = cfg["label_col"]
    on_cols = [id_col, time_col]
    how = (cfg.get("merge") or {}).get("how", "outer")
    chunksize = int(cfg.get("read_chunksize", 100_000))
    shard_count = int(cfg.get("shard_count", 8))

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        merged_path.unlink()
    header_written = False
    rows_written = 0

    for k in range(shard_count):
        shard_dfs: List[pd.DataFrame] = []
        for i, path in enumerate(per_table_paths):
            buf: List[pd.DataFrame] = []
            for chunk in pd.read_csv(path, chunksize=chunksize):
                if id_col not in chunk.columns:
                    raise ValueError(
                        "{0}: id column '{1}' missing".format(path, id_col))
                mask = shard_of(chunk[id_col], shard_count) == k
                if mask.any():
                    df = chunk.loc[mask].copy()
                    if i > 0 and label_col in df.columns:
                        df = df.drop(columns=[label_col])
                    buf.append(df)
            if buf:
                shard_dfs.append(pd.concat(buf, ignore_index=True))
            else:
                # Empty shard for this table — represent as id/time-only frame
                # so the outer/left merges still propagate other tables' rows.
                shard_dfs.append(pd.DataFrame(columns=on_cols))
        if all(len(d) == 0 for d in shard_dfs):
            continue
        merged = shard_dfs[0]
        for nxt in shard_dfs[1:]:
            merged = merged.merge(nxt, on=on_cols, how=how)
        if len(merged) == 0:
            continue
        merged.to_csv(merged_path, mode="a", header=not header_written, index=False)
        header_written = True
        rows_written += len(merged)
        logger.info("merge shard %d/%d: %d rows", k + 1, shard_count, len(merged))

    if not header_written:
        # Build header from the per-table headers (label_col only from t1).
        all_cols: List[str] = []
        seen = set()
        for i, path in enumerate(per_table_paths):
            cols = list(pd.read_csv(path, nrows=0).columns)
            if i > 0 and label_col in cols:
                cols = [c for c in cols if c != label_col]
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    all_cols.append(c)
        pd.DataFrame(columns=all_cols).to_csv(merged_path, index=False)
        logger.warning("%s: no merged rows, wrote header-only file", merged_path)
    logger.info("merged → %s (%d rows)", merged_path, rows_written)
    return rows_written


# ---------- driver ------------------------------------------------------- #

def run(cfg: dict, repo: Path, grans: List[str], dry_run: bool = False) -> None:
    aliases = cfg["time_col_aliases"]
    time_col = cfg["time_col"]

    # Stage A — for all selected grans first, so we fail fast.
    inputs_per_gran: Dict[str, Dict[int, List[Path]]] = {}
    for gran in grans:
        inputs = enumerate_inputs(cfg, gran, repo)
        validate_headers(inputs, aliases, time_col)
        inputs_per_gran[gran] = inputs
        logger.info("Stage A OK for %s: %d table(s), %d file(s)",
                    gran, len(inputs), sum(len(v) for v in inputs.values()))
    if dry_run:
        logger.info("dry-run: header validation passed; exiting before any writes")
        return

    output_dir = repo / cfg["output_dir"]

    # Stage B
    for gran in grans:
        per_table_tpl = cfg["sources"][gran]["per_table_out"]
        for table, in_files in inputs_per_gran[gran].items():
            out_path = output_dir / per_table_tpl.format(table=table)
            process_per_table(in_files, out_path, cfg, gran, table)

    # Stage C (optional)
    if (cfg.get("merge") or {}).get("enabled", False):
        for gran in grans:
            src = cfg["sources"][gran]
            n_tables = int(src["feature_tables"])
            per_table_paths = [
                output_dir / src["per_table_out"].format(table=t)
                for t in range(1, n_tables + 1)
            ]
            merged_path = output_dir / src["merged_out"]
            merge_horizontal(per_table_paths, merged_path, cfg)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess/hzz_raw.yaml")
    ap.add_argument("--source", choices=["day", "mon", "both"], default="both")

    mg = ap.add_mutually_exclusive_group()
    mg.add_argument("--merge", dest="merge", action="store_const", const=True, default=None)
    mg.add_argument("--no-merge", dest="merge", action="store_const", const=False, default=None)

    dg = ap.add_mutually_exclusive_group()
    dg.add_argument("--dedup", dest="dedup", action="store_const", const=True, default=None)
    dg.add_argument("--no-dedup", dest="dedup", action="store_const", const=False, default=None)

    ap.add_argument("--dedup-keep",
                    choices=["min_time", "max_time", "first", "last"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        # If a relative path was given, try resolving against script's repo root.
        alt = Path(__file__).resolve().parents[1] / args.config
        if alt.is_file():
            config_path = alt
        else:
            raise FileNotFoundError("config not found: {0}".format(args.config))
    repo = repo_root_of(config_path)
    cfg = load_cfg(config_path)
    cfg.setdefault("merge", {})
    cfg.setdefault("dedup", {})
    if args.merge is not None:
        cfg["merge"]["enabled"] = args.merge
    if args.dedup is not None:
        cfg["dedup"]["enabled"] = args.dedup
    if args.dedup_keep is not None:
        cfg["dedup"]["keep"] = args.dedup_keep

    grans = ["day", "mon"] if args.source == "both" else [args.source]
    logger.info("repo=%s  grans=%s  merge=%s  dedup=%s  shard_count=%s",
                repo, grans, cfg["merge"].get("enabled", False),
                cfg["dedup"].get("enabled", True),
                cfg.get("shard_count", 8))
    run(cfg, repo, grans, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
