"""Stage-2 end-to-end CLI: build dataset → tune → train → evaluate → export.

Usage:
    PYTHONPATH=src python3 scripts/run_training.py --product bank_marketing --run-id smoke01
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.config import load_config
from wdm.model.dataset import build_dataset
from wdm.model.evaluator import (
    evaluate_all, family_importance_audit, write_metrics_artifacts,
)
from wdm.model.exporter import export_bundle
from wdm.model.feature_pruner import maybe_prune_to_final
from wdm.model.trainer import train_final
from wdm.model.tuner import run_hyperopt
from wdm.plots.model_plots import make_all_model_plots
from wdm.utils.logging import setup_logging
from wdm.utils.paths import model_run_dir
from wdm.utils.progress import StageProgress


def main():
    ap = argparse.ArgumentParser(description="Stage-2 training + export.")
    ap.add_argument("--product", required=True)
    ap.add_argument("--run-id", required=True,
                    help="Artifact subdirectory under artifacts/<product>/models/")
    ap.add_argument("--features-version", default=None,
                    help="Override selected_features.active_version from config.")
    ap.add_argument("--max-evals", type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)
    version = args.features_version or cfg["selected_features"]["active_version"]

    run_dir = model_run_dir(cfg, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    prog = StageProgress("Stage 2", total=6)

    # 1. Dataset
    with prog.step("build dataset (features {0})".format(version)):
        data = build_dataset(cfg, version=version)

    # 1b. Stage-2 candidate → final funnel. Fires only when
    # training.stage2_candidate_count is set AND the loaded pool exceeds
    # final_feature_count; otherwise a pass-through.
    data = maybe_prune_to_final(data, cfg, run_dir)

    # 2. Hyperopt
    with prog.step("hyperopt search"):
        trials_path = run_dir / "trials.pkl"
        best_params, best_loss, trials = run_hyperopt(
            data.X_train, data.y_train, cfg,
            trials_path=str(trials_path),
            max_evals=args.max_evals,
            dt_train=data.dt_train,
            w_train=data.w_train,
        )

    # 3. Final train (early stopping on valid; weights enter the loss only)
    with prog.step("final train"):
        booster, evals_result = train_final(
            best_params, data.X_train, data.y_train,
            data.X_valid, data.y_valid, cfg, w_tr=data.w_train)

    # 4. Evaluate
    with prog.step("evaluate + metrics artifacts"):
        metrics_df, binned, scores, imp_df = evaluate_all(booster, data, cfg)
        audit_df = family_importance_audit(imp_df, cfg)
        write_metrics_artifacts(run_dir, metrics_df, binned, imp_df, audit_df,
                                best_params)

    # 5. Plots
    with prog.step("model plots"):
        make_all_model_plots(cfg, booster, data, scores, binned, imp_df,
                             out_dir=run_dir / "plots")

    # 6. Export deploy bundle
    with prog.step("export deploy bundle"):
        bundle = export_bundle(
            cfg, data, booster, evals_result, best_params, best_loss,
            selected_features_version=version, run_id=args.run_id,
            scores=scores,
        )

    prog.finish()

    objective = cfg["training"].get("tuner_objective", "aucpr")
    print()
    print("=" * 60)
    print("Stage 2 complete.")
    print("  run_id              :", args.run_id)
    print("  features version    :", version)
    print("  run dir             :", run_dir)
    print("  best CV {0:<12}:".format(objective), round(-best_loss, 4))
    print()
    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print()
    print("Deploy smoke test:")
    print("  python {0}/predict.py --validate".format(run_dir))


if __name__ == "__main__":
    main()
