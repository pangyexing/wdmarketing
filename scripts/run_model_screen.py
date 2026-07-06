#!/usr/bin/env python
"""Stage-1.5 CLI: model-based feature screen (target-permutation null importance).

Consumes the Stage-1 feature list (default: selected_features.active_version)
and writes a refined selected_features/<out_version>.txt (default v2_model).
Stage-1 must have been run first for the product.

Requires xgboost — run with the ML conda environment (env36, see README
"环境" for the $PY convention), e.g.:

    PYTHONPATH=src $PY scripts/run_model_screen.py --product xc_e2e_credit

Then train on the refined list:

    scripts/run_training.py --product xc_e2e_credit --run-id <id> \\
        --features-version v2_model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.analysis.null_importance import run_null_importance
from wdm.config import load_config
from wdm.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser(
        description="Stage-1.5 model-based feature screen (null importance).")
    ap.add_argument("--product", required=True,
                    help="Product name matching configs/products/<name>.yaml")
    ap.add_argument("--features-version", default=None,
                    help="Base feature list to screen (default: "
                         "selected_features.active_version).")
    ap.add_argument("--out-version", default=None,
                    help="Output list name (default: "
                         "analysis.null_importance.out_version).")
    ap.add_argument("--n-actual-runs", type=int, default=None)
    ap.add_argument("--n-null-runs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)
    result = run_null_importance(cfg,
                                 base_version=args.features_version,
                                 out_version=args.out_version,
                                 n_actual_runs=args.n_actual_runs,
                                 n_null_runs=args.n_null_runs,
                                 seed=args.seed)

    print()
    print("=" * 60)
    print("Model screen complete.")
    print("  features in         :", result["n_features_in"])
    print("  kept by null screen :", result["n_kept"])
    print("  written to list     :", result["n_written"])
    print("  report csv          :", result["report_csv"])
    print("  features txt        :", result["features_txt"])
    print()
    print("Next: scripts/run_training.py --product {0} --run-id <id> "
          "--features-version {1}".format(
              args.product, Path(result["features_txt"]).stem))


if __name__ == "__main__":
    main()
