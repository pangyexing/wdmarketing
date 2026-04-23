"""Stage-2 dataset builder.

Loads the full CSV (feature count ≤ 200 after selection → no chunking needed),
applies the same missing spec that Stage 1 used for NaN rules, fits stats on
TRAIN only, then transforms train/valid/oot. Missing indicator columns are
added here (not in Stage 1) to avoid double-counting signal in analysis.
"""
import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from wdm.preprocess.missing import (
    apply_missing_for_training, build_missing_spec, fit_missing,
    get_spec, to_nan_array,
)
from wdm.utils.time_utils import split_by_yyyymmdd, split_stratified

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StageTwoData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_oot: np.ndarray
    y_oot: np.ndarray
    feature_list: List[str]               # final feature order (incl. __isnan if any)
    base_feature_list: List[str]          # without the __isnan columns
    fitted: Dict                          # {feature: FittedStats}
    spec_map: Dict                        # {feature: MissingSpec}
    indicator_features: List[str]         # names of added __isnan columns
    raw_index: np.ndarray                 # row indices into the source CSV, for val-sample export
    train_mask: np.ndarray
    valid_mask: np.ndarray
    oot_mask: np.ndarray
    # All-NA row diagnostics. Train/valid rows with every base feature NaN are
    # dropped (no signal for learning); OOT rows are kept so evaluation stays
    # honest to production distribution. oot_all_na_mask is aligned with
    # X_oot/y_oot and lets the evaluator carve out an "excl. all-NA" subset.
    all_na_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    all_na_rates: Dict[str, float] = dataclasses.field(default_factory=dict)
    oot_all_na_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=bool))


def _load_selected_features(cfg, version=None):
    from wdm.utils.paths import selected_features_file
    p = selected_features_file(cfg, version)
    if not p.is_file():
        raise FileNotFoundError("Selected features file not found: {0}".format(p))
    feats = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    if not feats:
        raise ValueError("Empty selected features file: {0}".format(p))
    return feats, p


def _split_masks(df, cfg):
    split_cfg = cfg["training"]["split"]
    strategy = split_cfg["strategy"]
    ratios = list(split_cfg["ratios"])
    seed = int(cfg["training"]["random_seed"])
    if strategy == "stratified":
        if cfg["data"].get("time_column"):
            logger.warning("time_column is configured but split.strategy='stratified' — using stratified anyway")
        return split_stratified(df[cfg["data"]["label_column"]].values, ratios, seed=seed)
    if strategy == "time":
        time_col = cfg["data"].get("time_column")
        if not time_col:
            raise ValueError("split.strategy='time' requires data.time_column")
        return split_by_yyyymmdd(df[time_col], ratios)
    raise ValueError("Unknown split strategy: {0}".format(strategy))


def build_dataset(cfg, version=None):
    """Build StageTwoData for the given selected-features version."""
    feats, feats_path = _load_selected_features(cfg, version)
    logger.info("Loaded selected features (%d) from %s", len(feats), feats_path)

    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    label_col = cfg["data"]["label_column"]
    time_col = cfg["data"].get("time_column")
    needed = list(dict.fromkeys(feats + [label_col] + ([time_col] if time_col else [])))
    df = pd.read_csv(path, usecols=needed)

    m_tr, m_va, m_oot = _split_masks(df, cfg)
    logger.info("Split sizes: train=%d valid=%d oot=%d",
                int(m_tr.sum()), int(m_va.sum()), int(m_oot.sum()))

    spec_map = build_missing_spec(cfg)

    missing_cfg = cfg["missing"]["global"]
    generate_indicator = bool(missing_cfg.get("generate_missing_indicator", False))
    indicator_threshold = float(missing_cfg.get("indicator_threshold", 0.10))

    # Compute missing mask per feature on TRAIN; indicator if miss_rate exceeds threshold
    df_tr = df.loc[m_tr, feats].copy()
    df_va = df.loc[m_va, feats].copy()
    df_oot = df.loc[m_oot, feats].copy()

    indicator_feats = []
    if generate_indicator:
        for feat in feats:
            spec = get_spec(spec_map, feat)
            arr, mask = to_nan_array(df_tr[feat], spec)
            if float(mask.mean()) >= indicator_threshold:
                indicator_feats.append("{0}__isnan".format(feat))
    if indicator_feats:
        logger.info("Generating %d missing indicators: %s", len(indicator_feats),
                    indicator_feats[:5] + (["..."] if len(indicator_feats) > 5 else []))

    # Fit missing stats on TRAIN only
    fitted = fit_missing(df_tr, spec_map)

    # Build a little helper that applies missing plus adds indicator columns
    def _apply(frame):
        out = apply_missing_for_training(frame, spec_map, fitted)
        for ind in indicator_feats:
            base = ind[:-len("__isnan")]
            spec = get_spec(spec_map, base)
            _arr, mask = to_nan_array(frame[base], spec)
            out[ind] = mask.astype(np.int8)
        return out

    def _all_na_row_mask(frame):
        # True where every base feature is NaN after sentinel/negative/empty
        # rules. Drives the all-NA row filter below.
        n_rows = len(frame)
        all_na = np.ones(n_rows, dtype=bool)
        for feat in feats:
            spec = get_spec(spec_map, feat)
            _arr, mask = to_nan_array(frame[feat], spec)
            all_na &= mask
            if not all_na.any():
                break
        return all_na

    X_tr = _apply(df_tr).values.astype(np.float32)
    X_va = _apply(df_va).values.astype(np.float32)
    X_oot = _apply(df_oot).values.astype(np.float32)

    y_all = df[label_col].astype(np.int64).values
    y_tr = y_all[m_tr]
    y_va = y_all[m_va]
    y_oot = y_all[m_oot]

    # Detect rows where every selected feature is NaN under the resolved spec.
    # Strategy is asymmetric across splits:
    #   - train / valid : drop (no signal for learning / early-stopping).
    #   - oot           : keep (evaluation must reflect production distribution;
    #                     evaluator reports an extra "oot_excl_all_na" row).
    all_na_tr = _all_na_row_mask(df_tr)
    all_na_va = _all_na_row_mask(df_va)
    all_na_oot = _all_na_row_mask(df_oot)

    train_mask = np.asarray(m_tr).copy()
    valid_mask = np.asarray(m_va).copy()
    oot_mask = np.asarray(m_oot).copy()

    all_na_counts = {
        "train": int(all_na_tr.sum()),
        "valid": int(all_na_va.sum()),
        "oot":   int(all_na_oot.sum()),
    }
    all_na_rates = {
        "train": all_na_counts["train"] / max(1, len(all_na_tr)),
        "valid": all_na_counts["valid"] / max(1, len(all_na_va)),
        "oot":   all_na_counts["oot"]   / max(1, len(all_na_oot)),
    }

    # Empty-split guard only applies to splits we actually drop from.
    for split_name, n_total, n_drop in (
        ("train", len(all_na_tr), all_na_counts["train"]),
        ("valid", len(all_na_va), all_na_counts["valid"]),
    ):
        if n_total - n_drop == 0:
            raise ValueError(
                "After dropping all-NA rows, {0} has 0 samples — feature list "
                "likely excludes every populated column for this split".format(split_name)
            )

    if sum(all_na_counts.values()) > 0:
        logger.info(
            "All-NA rows: train=%d/%d (%.2f%%) [drop] valid=%d/%d (%.2f%%) [drop] "
            "oot=%d/%d (%.2f%%) [keep]",
            all_na_counts["train"], len(all_na_tr), 100.0 * all_na_rates["train"],
            all_na_counts["valid"], len(all_na_va), 100.0 * all_na_rates["valid"],
            all_na_counts["oot"],   len(all_na_oot), 100.0 * all_na_rates["oot"],
        )
        for split_name in ("train", "valid", "oot"):
            if all_na_rates[split_name] > 0.05:
                logger.warning(
                    "High all-NA rate in %s (%.2f%%) — review the selected feature list",
                    split_name, 100.0 * all_na_rates[split_name],
                )

    def _apply_drop(X, y, all_na_row, split_bool_mask):
        if not all_na_row.any():
            return X, y, split_bool_mask
        kept = ~all_na_row
        true_positions = np.where(split_bool_mask)[0]
        split_bool_mask[true_positions[all_na_row]] = False
        return X[kept], y[kept], split_bool_mask

    X_tr, y_tr, train_mask = _apply_drop(X_tr, y_tr, all_na_tr, train_mask)
    X_va, y_va, valid_mask = _apply_drop(X_va, y_va, all_na_va, valid_mask)
    # OOT rows are NOT dropped; retain mask so evaluator can cut a complement.

    feat_order = list(feats) + indicator_feats

    return StageTwoData(
        X_train=X_tr, y_train=y_tr,
        X_valid=X_va, y_valid=y_va,
        X_oot=X_oot, y_oot=y_oot,
        feature_list=feat_order,
        base_feature_list=list(feats),
        fitted=fitted,
        spec_map=spec_map,
        indicator_features=indicator_feats,
        raw_index=np.arange(len(df)),
        train_mask=train_mask,
        valid_mask=valid_mask,
        oot_mask=oot_mask,
        all_na_counts=all_na_counts,
        all_na_rates=all_na_rates,
        oot_all_na_mask=np.asarray(all_na_oot).copy(),
    )
