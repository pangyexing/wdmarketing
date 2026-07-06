"""Stage-1 end-to-end CLI: run feature analysis for a configured product.

Usage:
    PYTHONPATH=src python3 scripts/run_analysis.py --product bank_marketing
"""
import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.config import load_config
from wdm.analysis.selector import run_stage1
from wdm.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser(description="Stage-1 feature analysis.")
    ap.add_argument("--product", required=True, help="Product name matching configs/products/<name>.yaml")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip per-feature plot generation (faster).")
    ap.add_argument("--model-screen", action="store_true",
                    help="After Stage 1, run the model-based null-importance "
                         "screen (needs xgboost; see scripts/run_model_screen.py).")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)
    result = run_stage1(cfg)

    print()
    print("=" * 60)
    print("Stage 1 complete.")
    print("  n_features analyzed :", result["n_features"])
    print("  n_auto_kept         :", result["n_auto_kept"])
    print("  report dir          :", result["report_dir"])
    print("  auto features       :", result["auto_features"])
    print()
    print("Next: inspect {0}/index.html, optionally copy v1_auto.txt to v1_manual.txt"
          " and edit, then run scripts/run_training.py --product {1} --run-id <id>".format(
          result["report_dir"], args.product))

    ni_cfg = (cfg["analysis"].get("null_importance") or {})
    if args.model_screen or bool(ni_cfg.get("enabled", False)):
        from wdm.analysis.null_importance import run_null_importance
        try:
            screen = run_null_importance(cfg)
        except RuntimeError as e:
            # Stage-1 artifacts are already on disk — only the screen failed.
            print("Model screen FAILED (Stage-1 artifacts are intact): {0}".format(e),
                  file=sys.stderr)
            sys.exit(1)
        print()
        print("Model screen: kept {0} / {1} features → {2}".format(
            screen["n_kept"], screen["n_features_in"], screen["features_txt"]))

    # Per-feature plots are optionally run here after Stage 1 closes.
    if not args.no_plots:
        from wdm.plots.feature_plots import run_per_feature_plots
        run_per_feature_plots(cfg, result["bin_specs"])


if __name__ == "__main__":
    main()
