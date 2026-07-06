"""Stream-build a CSR cache from a wide CSV (CLI).

The build/load logic lives in wdm.io.sparse_cache — this script is the CLI
wrapper. See that module's docstring for the cache design (raw CSR, implicit
zeros, NaN explicit, missing semantic resolved at train time) and the output
layout under data/cache/<product>/.

Usage:
    PYTHONPATH=src python3 scripts/build_sparse_cache.py --product home_credit
    PYTHONPATH=src python3 scripts/build_sparse_cache.py --product home_credit --chunk-rows 20000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.config import load_config
from wdm.io.sparse_cache import (  # noqa: F401  (re-exported for back-compat)
    build_sparse_cache,
    load_cache,
    resolve_cache_dir,
)
from wdm.utils.logging import setup_logging

# Back-compat alias: older callers imported the underscore name from here.
_resolve_cache_dir = resolve_cache_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--product", required=True,
                    help="Product name matching configs/products/<name>.yaml")
    ap.add_argument("--chunk-rows", type=int, default=50_000,
                    help="Rows per pd.read_csv chunk (default 50000). Lower "
                         "if the per-chunk dense buffer is too large.")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)
    csv_path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    if not csv_path.is_file():
        raise FileNotFoundError("data.train_path not found: {0}".format(csv_path))

    out_dir = resolve_cache_dir(cfg)
    meta = build_sparse_cache(csv_path, out_dir, cfg, chunk_rows=args.chunk_rows)

    print()
    print("=" * 60)
    print("Sparse cache built.")
    print("  dir     :", out_dir)
    print("  shape   : {0} rows × {1} features".format(meta["n_rows"], meta["n_features"]))
    print("  density : {0:.4f}  ({1} nnz)".format(meta["density"], meta["nnz"]))
    print("  csv     :", meta["csv_path"])
    print()
    print("Next: run_analysis.py will pick up the cache automatically when "
          "analysis.probing.enabled=true.")


if __name__ == "__main__":
    main()
