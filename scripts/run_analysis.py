"""Stage-1 end-to-end CLI: run feature analysis for a configured product.

Usage:
    PYTHONPATH=src python3 scripts/run_analysis.py --product bank_marketing
    PYTHONPATH=src python3 scripts/run_analysis.py --product home_credit --probing
    PYTHONPATH=src python3 scripts/run_analysis.py --product home_credit --no-probing
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.config import load_config
from wdm.analysis.selector import run_stage1
from wdm.utils.logging import setup_logging
from wdm.utils.paths import ensure_dirs, report_dir

logger = logging.getLogger(__name__)


def _resolve_probing_enabled(cfg, cli_flag):
    """CLI flag takes precedence; otherwise fall back to config."""
    if cli_flag is not None:
        return bool(cli_flag)
    return bool((cfg.get("analysis") or {}).get("probing", {}).get("enabled", False))


def _resolve_cache_dir(cfg):
    """Mirror scripts/build_sparse_cache._resolve_cache_dir to stay in sync."""
    override = (cfg.get("analysis") or {}).get("probing", {}).get("cache_dir")
    if override:
        return Path(cfg["_repo_root"]) / override
    return Path(cfg["_repo_root"]) / "data" / "cache" / cfg["name"]


def _maybe_run_probing(cfg, enabled):
    """Run Stage-1 probing before run_stage1 so selector.py can pick up
    probing_importance.csv when computing rank_score.
    """
    if not enabled:
        logger.info("Probing disabled — skipping.")
        return None

    cache_dir = _resolve_cache_dir(cfg)
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        print(
            "[probing] Cache not found at {0}.\n"
            "[probing] Build it first:\n"
            "[probing]   PYTHONPATH=src python3 scripts/build_sparse_cache.py "
            "--product {1}\n"
            "[probing] Proceeding without probing signal.".format(
                cache_dir, cfg["name"]),
            file=sys.stderr)
        return None

    rdir = report_dir(cfg)
    ensure_dirs(rdir)

    from wdm.analysis.probing import run_probing
    logger.info("Running Stage-1 probing (cache=%s → report=%s)", cache_dir, rdir)
    return run_probing(cfg, cache_dir=cache_dir, out_dir=rdir)


def main():
    ap = argparse.ArgumentParser(description="Stage-1 feature analysis.")
    ap.add_argument("--product", required=True, help="Product name matching configs/products/<name>.yaml")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip per-feature plot generation (faster).")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--probing", dest="probing", action="store_true",
                       default=None,
                       help="Force-enable Stage-1 probing model. Requires a "
                            "prebuilt CSR cache (scripts/build_sparse_cache.py).")
    group.add_argument("--no-probing", dest="probing", action="store_false",
                       default=None,
                       help="Force-disable Stage-1 probing (default: follow config).")
    ap.add_argument("--model-screen", action="store_true",
                    help="After Stage 1, run the model-based null-importance "
                         "screen (needs xgboost; see scripts/run_model_screen.py).")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)

    # Probing runs before the statistical pipeline so its importance CSV
    # is on disk when selector.py assembles rank_score.
    probing_enabled = _resolve_probing_enabled(cfg, args.probing)
    probing_result = _maybe_run_probing(cfg, probing_enabled)

    result = run_stage1(cfg)

    print()
    print("=" * 60)
    print("Stage 1 complete.")
    print("  n_features analyzed :", result["n_features"])
    print("  n_auto_kept         :", result["n_auto_kept"])
    print("  report dir          :", result["report_dir"])
    print("  auto features       :", result["auto_features"])
    if probing_result:
        print("  probing best AUCPR  : {0:.4f}".format(
            probing_result.get("best_valid_aucpr", float("nan"))))
        print("  probing importance  :", probing_result["importance_path"])
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
