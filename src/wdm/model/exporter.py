"""Stage-2 exporter: assemble the deployable bundle.

Emits under artifacts/<product>/models/<run_id>/:
  booster.json             — native xgboost model
  feature_list.txt         — the final column order (incl. __isnan indicators)
  missing_spec.json        — training-time rules + fit stats for replay
  predict.py               — copy of the template with the bundle layout baked in
  validation_samples.csv   — N raw rows with y_true + y_pred_expected
  importance.csv           — gain/weight/cover
  run_manifest.json        — reproducibility snapshot
  trials.pkl               — hyperopt trials (written separately by run_training)
  best_params.json
  metrics.json / .md
  plots/*.png

Key contract: validation_samples.csv uses RAW features (no missing-value
handling applied). predict.py reads raw CSV and applies its own missing logic,
so deployers never need to understand how features are transformed.
"""
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from wdm.preprocess.missing import dump_missing_spec
from wdm.utils.paths import load_column_mapping, model_run_dir, ensure_dirs

logger = logging.getLogger(__name__)


def _write_feature_list(path, feature_list):
    header = [
        "# Feature list used by the model. DO NOT REORDER.",
        "# Lines starting with '#' are comments and ignored by predict.py.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + list(feature_list)) + "\n")


def _raw_validation_samples(cfg, data, booster, n_samples):
    """Sample N rows from the RAW CSV (pre-missing-handling), score them with
    the current booster, and emit a CSV with y_true + y_pred_expected.

    The row pool is drawn preferentially from valid/oot segments so deployers
    see distribution close to inference time.
    """
    from wdm.io.chunked_reader import read_full

    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    label_col = cfg["data"]["label_column"]
    df_raw = read_full(path, columns=data.base_feature_list + [label_col])

    rng = np.random.RandomState(cfg["training"]["random_seed"] + 1)
    pool_mask = data.valid_mask | data.oot_mask
    pool_idx = np.where(pool_mask)[0]
    if pool_idx.size < n_samples:
        logger.warning("Pool smaller than requested sample count; drawing from full dataset")
        pool_idx = np.arange(len(df_raw))

    # Stratified sample across label to guarantee coverage of positives
    y_pool = df_raw.iloc[pool_idx][label_col].values
    pos_idx = pool_idx[y_pool == 1]
    neg_idx = pool_idx[y_pool == 0]
    n_pos = max(1, int(round(n_samples * y_pool.mean())))
    n_neg = n_samples - n_pos
    if pos_idx.size < n_pos:
        n_pos = pos_idx.size
        n_neg = n_samples - n_pos
    if neg_idx.size < n_neg:
        n_neg = neg_idx.size
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)

    sample_raw = df_raw.iloc[sel].reset_index(drop=True)

    # Score these rows THROUGH the same pipeline they'll go through at predict time.
    # To do this we replay apply_missing_for_training using the already-fitted stats.
    from wdm.preprocess.missing import apply_missing_for_training, to_nan_array
    applied = apply_missing_for_training(
        sample_raw[data.base_feature_list],
        data.spec_map, data.fitted).astype(np.float32)

    # Add indicator columns from raw, consistent with build_dataset
    frames = [applied]
    for ind in data.indicator_features:
        base = ind[:-len("__isnan")]
        from wdm.preprocess.missing import get_spec
        spec = get_spec(data.spec_map, base)
        _arr, mask = to_nan_array(sample_raw[base], spec)
        frames.append(pd.DataFrame({ind: mask.astype(np.int8)}))
    full = pd.concat(frames, axis=1)[data.feature_list].values.astype(np.float32)

    dmat = xgb.DMatrix(full)
    try:
        best_iter = booster.best_iteration + 1
        scores = booster.predict(dmat, iteration_range=(0, best_iter))
    except Exception:
        scores = booster.predict(dmat, ntree_limit=booster.best_ntree_limit)

    out = sample_raw.copy()
    out.rename(columns={label_col: "y_true"}, inplace=True)
    out["y_pred_expected"] = scores.astype(np.float64)
    return out


def export_bundle(cfg, data, booster, evals_result, best_params, best_params_loss,
                  selected_features_version, run_id, evaluator_artifacts=None):
    """Materialize the deploy bundle. Returns the bundle path."""
    run_dir = model_run_dir(cfg, run_id)
    ensure_dirs(run_dir, run_dir / "plots")

    # 1. booster
    booster_path = run_dir / "booster.json"
    booster.save_model(str(booster_path))

    # 2. feature list
    _write_feature_list(run_dir / "feature_list.txt", data.feature_list)

    # 3. missing_spec.json
    dump_missing_spec(run_dir / "missing_spec.json", data.spec_map, data.fitted)

    # 4. predict.py (template copy)
    template_path = Path(cfg["_repo_root"]) / "scripts" / "predict_template.py"
    shutil.copyfile(str(template_path), str(run_dir / "predict.py"))

    # 5. validation_samples.csv (RAW features + y_true + y_pred_expected)
    n_samples = int(cfg["export"].get("validation_sample_count", 100))
    val_df = _raw_validation_samples(cfg, data, booster, n_samples)
    val_df.to_csv(run_dir / "validation_samples.csv", index=False)

    # 6. run_manifest.json
    import yaml
    manifest = {
        "run_id": run_id,
        "product": cfg["name"],
        "selected_features_version": selected_features_version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "random_seed": cfg["training"]["random_seed"],
        "xgb_version": xgb.__version__,
        "n_features_total": len(data.feature_list),
        "n_features_base": len(data.base_feature_list),
        "n_features_indicator": len(data.indicator_features),
        "base_features": list(data.base_feature_list),
        "indicator_features": list(data.indicator_features),
        "split_strategy": cfg["training"]["split"]["strategy"],
        "split_ratios": cfg["training"]["split"]["ratios"],
        "top_k_pct": cfg["training"]["top_k_pct"],
        "best_params": best_params,
        "best_cv_pr_auc": -float(best_params_loss),
        "family_policy": cfg["feature_groups"].get("family_policy", {}),
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    # 7. importance.csv — written by evaluator, but decorate with Chinese names here
    imp_path = run_dir / "importance.csv"
    if imp_path.is_file():
        mapping = load_column_mapping(cfg)
        imp_df = pd.read_csv(imp_path)
        if "feature_cn" not in imp_df.columns:
            imp_df.insert(1, "feature_cn", imp_df["feature"].map(lambda f: mapping.get(f, f)))
            imp_df.to_csv(imp_path, index=False)

    logger.info("Exported bundle to %s", run_dir)
    return run_dir
