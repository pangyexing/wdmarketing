"""CSR sparse-cache build/load shared by scripts and the probing model.

Why a cache:
- Wide tables (e.g. home_credit_wide.csv ~870 MB) do not fit in memory as
  float64 DataFrames (~2 GB). A CSR cache captures the same information in
  O(nnz) memory, which for HC's ~85% sparsity is <100 MB.
- The cache is **raw**: structural zeros are implicit (memory saved), NaN is
  stored explicitly in .data. The decision "0 = missing?" is NOT baked into
  the cache — the probing trainer resolves it at train time from
  cfg.missing.global.sentinels, so the same cache works for products that
  treat 0 as legitimate (home_credit) and products that treat 0 as missing
  (bank_marketing).

Output layout: data/cache/<product>/
    X.csr.npz          scipy.sparse.csr_matrix (feature block, float32)
    y.npy              label vector (original dtype)
    feature_names.npy  np.ndarray[str] in CSR column order
    <time_column>.npy  time values (if cfg.data.time_column is set)
    <id_col>.npy       one file per id column
    manifest.json      metadata incl. CSV size+mtime for staleness checks

CLI entry point: scripts/build_sparse_cache.py (thin wrapper around this
module).
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def resolve_cache_dir(cfg):
    """Default cache dir: data/cache/<product>/. Overridable via cfg.analysis.probing.cache_dir."""
    override = (cfg.get("analysis") or {}).get("probing", {}).get("cache_dir")
    if override:
        return Path(cfg["_repo_root"]) / override
    return Path(cfg["_repo_root"]) / "data" / "cache" / cfg["name"]


def build_sparse_cache(csv_path, out_dir, cfg, chunk_rows=50_000):
    """Stream-read csv_path; write CSR + sidecar arrays to out_dir.

    Memory profile: peak is roughly chunk_rows × n_feature_cols × 4 bytes
    (one dense chunk buffer in flight). For chunk_rows=50_000 and
    n_features=900 that's ~180 MB per chunk.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_col = cfg["data"]["label_column"]
    time_col = cfg["data"].get("time_column")
    id_cols = list(cfg["data"].get("id_columns") or [])
    treatment_col = cfg["data"].get("treatment_column")

    # Reserved: not serialized as features
    reserved = set([label_col])
    if time_col:
        reserved.add(time_col)
    if treatment_col:
        reserved.add(treatment_col)
    for c in id_cols:
        reserved.add(c)

    # 1) Peek header + dtype_map (features → float32; reserved keep pandas default)
    first = pd.read_csv(csv_path, nrows=1)
    all_cols = list(first.columns)
    feat_cols = [c for c in all_cols if c not in reserved]
    if not feat_cols:
        raise ValueError("No feature columns found after excluding reserved "
                         "columns {0}".format(reserved))
    dtype_map = {c: np.float32 for c in feat_cols}

    logger.info("Build cache: %d feature cols (reserved=%s)",
                len(feat_cols), sorted(reserved))

    # 2) Stream chunks → per-chunk CSR → accumulate
    x_blocks = []
    y_parts = []
    time_parts = []
    id_parts = {c: [] for c in id_cols}
    n_rows = 0

    reader = pd.read_csv(csv_path, chunksize=chunk_rows, dtype=dtype_map)
    for i, chunk in enumerate(reader):
        y_parts.append(chunk[label_col].to_numpy())
        if time_col:
            time_parts.append(chunk[time_col].to_numpy())
        for c in id_cols:
            id_parts[c].append(chunk[c].to_numpy())

        # Feature dense block → CSR (NaN kept as explicit nonzero; 0 → implicit)
        dense = chunk[feat_cols].to_numpy()
        x_blocks.append(sp.csr_matrix(dense))
        n_rows += dense.shape[0]
        del dense, chunk
        logger.info("  chunk %d: cumulative rows=%d", i + 1, n_rows)

    X = sp.vstack(x_blocks, format="csr")
    del x_blocks
    y = np.concatenate(y_parts)

    # 3) Write artifacts
    sp.save_npz(out_dir / "X.csr.npz", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "feature_names.npy", np.array(feat_cols, dtype=object))
    if time_col:
        np.save(out_dir / ("{0}.npy".format(time_col)),
                np.concatenate(time_parts))
    for c in id_cols:
        np.save(out_dir / ("{0}.npy".format(c)), np.concatenate(id_parts[c]))

    # 4) Manifest: used later to detect stale cache
    stat = csv_path.stat()
    nnz = int(X.nnz)
    total_cells = int(X.shape[0]) * int(X.shape[1])
    density = float(nnz) / float(total_cells) if total_cells else 0.0
    meta = {
        "product": cfg["name"],
        "csv_path": str(csv_path),
        "csv_size_bytes": stat.st_size,
        "csv_mtime": stat.st_mtime,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "nnz": nnz,
        "density": density,
        "dtype": str(X.dtype),
        "label_column": label_col,
        "time_column": time_col,
        "id_columns": id_cols,
        "treatment_column": treatment_col,
        "chunk_rows": int(chunk_rows),
        "sparse_convention": ("implicit_zero=0.0 (structural); NaN stored "
                              "explicit in .data. missing semantic is "
                              "resolved at probing train time from "
                              "cfg.missing.global.sentinels."),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info("Cache built: %s", out_dir)
    logger.info("  shape=%s  nnz=%d  density=%.4f  dtype=%s",
                X.shape, nnz, density, X.dtype)
    return meta


def load_cache(cache_dir, csv_path=None):
    """Load cache; optionally verify CSV size+mtime against manifest."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "manifest.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            "Cache manifest missing at {0}. Run "
            "scripts/build_sparse_cache.py --product <name> first.".format(meta_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if csv_path is not None:
        st = Path(csv_path).stat()
        if (meta.get("csv_size_bytes") != st.st_size
                or abs(float(meta.get("csv_mtime", 0)) - st.st_mtime) > 1e-3):
            raise RuntimeError(
                "Cache stale: source CSV has changed since cache was built.\n"
                "  cache built from : size={0}, mtime={1}\n"
                "  current CSV      : size={2}, mtime={3}\n"
                "Rebuild: PYTHONPATH=src python3 scripts/build_sparse_cache.py "
                "--product <name>".format(
                    meta.get("csv_size_bytes"), meta.get("csv_mtime"),
                    st.st_size, st.st_mtime))

    X = sp.load_npz(cache_dir / "X.csr.npz")
    y = np.load(cache_dir / "y.npy")
    feat_names = np.load(cache_dir / "feature_names.npy", allow_pickle=True)
    out = {"X": X, "y": y, "feature_names": list(feat_names), "meta": meta}
    time_col = meta.get("time_column")
    if time_col:
        p = cache_dir / ("{0}.npy".format(time_col))
        if p.is_file():
            out[time_col] = np.load(p, allow_pickle=True)
    for c in meta.get("id_columns") or []:
        p = cache_dir / ("{0}.npy".format(c))
        if p.is_file():
            out[c] = np.load(p, allow_pickle=True)
    return out
