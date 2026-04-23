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

    X_tr = _apply(df_tr).values.astype(np.float32)
    X_va = _apply(df_va).values.astype(np.float32)
    X_oot = _apply(df_oot).values.astype(np.float32)

    y_all = df[label_col].astype(np.int64).values
    y_tr = y_all[m_tr]
    y_va = y_all[m_va]
    y_oot = y_all[m_oot]

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
        train_mask=np.asarray(m_tr),
        valid_mask=np.asarray(m_va),
        oot_mask=np.asarray(m_oot),
    )
