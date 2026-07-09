"""Stage-2 dataset builder.

Loads the full CSV (feature count ≤ 200 after selection → no chunking needed),
applies the same missing spec that Stage 1 used for NaN rules, fits stats on
TRAIN only, then transforms train/valid/oot. Missing indicator columns are
added here (not in Stage 1) to avoid double-counting signal in analysis.
"""
import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from wdm.preprocess.missing import (
    apply_missing_for_training, build_missing_spec, fit_missing,
    get_spec, to_nan_array,
)
from wdm.utils.labels import validate_binary_label
from wdm.utils.split_masks import compute_split_masks

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
    # Per-row yyyymmdd (float64, NaN allowed) per split; None without a time column.
    dt_train: Optional[np.ndarray] = None
    dt_valid: Optional[np.ndarray] = None
    dt_oot: Optional[np.ndarray] = None
    # Per-row training-loss weights per split; None unless training.sample_weight is set.
    w_train: Optional[np.ndarray] = None
    w_valid: Optional[np.ndarray] = None
    w_oot: Optional[np.ndarray] = None
    # Calibration holdout carved from valid (training.calibration_split_fraction,
    # only when export.calibration is enabled): X_valid/y_valid keep the
    # early-stopping/selection half; this is the time-later half reserved for
    # isotonic fitting, so the calibration curve is never fit on the set that
    # drove early stopping. None when disabled. calib_mask marks the holdout
    # rows at raw-CSV length (valid_mask excludes them, keeping the invariant
    # valid_mask.sum() == len(y_valid)).
    X_calib: Optional[np.ndarray] = None
    y_calib: Optional[np.ndarray] = None
    dt_calib: Optional[np.ndarray] = None
    w_calib: Optional[np.ndarray] = None
    calib_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=bool))


def _load_selected_features(cfg, version=None):
    from wdm.utils.paths import selected_features_file
    p = selected_features_file(cfg, version)
    if not p.is_file():
        hint = ""
        if p.stem == "v2_model":
            hint = (" v2_model.txt is produced by the Stage-1.5 model screen — "
                    "run `scripts/run_analysis.py --product {0} --model-screen` "
                    "(or scripts/run_model_screen.py) first.".format(cfg["name"]))
        elif p.stem == "v1_auto":
            hint = (" v1_auto.txt is produced by Stage-1 — run "
                    "`scripts/run_analysis.py --product {0}` first.".format(cfg["name"]))
        raise FileNotFoundError(
            "Selected features file not found: {0}.{1}".format(p, hint))
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


def build_sample_weights(series, sw_cfg):
    """Map a raw column to per-row weights via training.sample_weight.

    Numeric equality match against mapping keys; NaN and unmapped values get
    `default` (1.0 when omitted). Returns a float64 array.
    """
    default = float(sw_cfg.get("default", 1.0))
    num = pd.to_numeric(pd.Series(series), errors="coerce")
    w = np.full(num.size, default, dtype=np.float64)
    for key, weight in sw_cfg["mapping"].items():
        try:
            kf = float(key)
        except (TypeError, ValueError):
            raise ValueError("sample_weight.mapping key {0!r} is not numeric".format(key))
        w[(num == kf).values] = float(weight)
    return w


def build_dataset(cfg, version=None):
    """Build StageTwoData for the given selected-features version."""
    feats, feats_path = _load_selected_features(cfg, version)
    logger.info("Loaded selected features (%d) from %s", len(feats), feats_path)

    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    label_col = cfg["data"]["label_column"]
    time_col = cfg["data"].get("time_column")
    sw_cfg = cfg["training"].get("sample_weight")
    exclude_rows = cfg["data"].get("exclude_rows") or []
    extra_cols = ([sw_cfg["column"]] if sw_cfg else []) + [r["column"] for r in exclude_rows]
    needed = list(dict.fromkeys(feats + [label_col]
                                + ([time_col] if time_col else []) + extra_cols))
    df = pd.read_csv(path, usecols=needed)
    n_raw = len(df)

    # Shared with Stage-1 (wdm.pipeline.stage1) so both stages agree on which
    # rows are "train": exclude_rows are False in all three masks (exporter
    # indexes the raw CSV with them) and time splits purge embargo_days.
    m_tr, m_va, m_oot, included = compute_split_masks(df, cfg)
    validate_binary_label(df.loc[included, label_col], label_col)
    if exclude_rows:
        logger.info("exclude_rows kept %d / %d rows", int(included.sum()), n_raw)
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

    # Per-row time (for time_forward CV) and loss weights, aligned to the raw
    # split masks; the all-NA drop below re-filters the train/valid slices.
    dt_tr = dt_va = dt_oot = None
    if time_col:
        dt_all = pd.to_numeric(
            df[time_col].astype(str).str.replace("-", "", regex=False)
            if not pd.api.types.is_numeric_dtype(df[time_col]) else df[time_col],
            errors="coerce").values.astype(np.float64)
        dt_tr, dt_va, dt_oot = dt_all[m_tr], dt_all[m_va], dt_all[m_oot]

    w_tr = w_va = w_oot = None
    if sw_cfg:
        w_all = build_sample_weights(df[sw_cfg["column"]], sw_cfg)
        w_tr, w_va, w_oot = w_all[m_tr], w_all[m_va], w_all[m_oot]
        logger.info("sample_weight on %s: train weight sum=%.1f (n=%d), distinct weights=%s",
                    sw_cfg["column"], float(w_tr.sum()), w_tr.size,
                    sorted(set(np.unique(w_tr).tolist())))

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

    # Keep per-row time / weight arrays aligned with the dropped train/valid rows.
    def _filter_aux(a, all_na_row):
        if a is None or not all_na_row.any():
            return a
        return a[~all_na_row]

    dt_tr = _filter_aux(dt_tr, all_na_tr)
    dt_va = _filter_aux(dt_va, all_na_va)
    w_tr = _filter_aux(w_tr, all_na_tr)
    w_va = _filter_aux(w_va, all_na_va)

    # Carve a calibration holdout out of valid so isotonic calibration is not
    # fit on the same rows that drive early stopping / feature pruning. With a
    # time column the LATER tail becomes the holdout (closest to OOT /
    # production); otherwise a seeded random subset. Only carved when
    # export.calibration is enabled — without calibration the holdout would
    # shrink the early-stopping set for nothing. calibration_split_fraction=0
    # disables (legacy: calibration then fits on the full valid).
    X_cal = y_cal = dt_cal = w_cal = None
    calib_mask = np.zeros(len(valid_mask), dtype=bool)
    calib_frac = float(cfg["training"].get("calibration_split_fraction", 0.5) or 0.0)
    calib_enabled = bool(((cfg.get("export") or {}).get("calibration") or {})
                         .get("enabled", False))
    if calib_frac > 0 and calib_enabled:
        n_va_rows = int(y_va.shape[0])
        n_cal = int(round(n_va_rows * calib_frac))
        # Precheck with the same thresholds fit_isotonic_table will apply:
        # if the holdout could never fit a curve, don't pay for it — carving
        # would shrink the early-stopping set AND leave the model uncalibrated.
        calib_cfg = ((cfg.get("export") or {}).get("calibration") or {})
        cal_min_rows = int(calib_cfg.get("min_valid_rows", 200))
        cal_min_pos = int(calib_cfg.get("min_valid_pos", 10))
        if n_cal >= 1 and n_va_rows - n_cal >= 1:
            if dt_va is not None:
                order = np.argsort(np.where(np.isnan(dt_va), np.inf, dt_va),
                                   kind="mergesort")
            else:
                order = np.random.RandomState(
                    int(cfg["training"]["random_seed"])).permutation(n_va_rows)
            cal_sel = np.zeros(n_va_rows, dtype=bool)
            cal_sel[order[n_va_rows - n_cal:]] = True
            n_pos_cal = int(np.sum(y_va[cal_sel] == 1))
            if n_cal < cal_min_rows or n_pos_cal < cal_min_pos \
                    or n_pos_cal == n_cal:
                logger.warning(
                    "calibration holdout would be unusable (n=%d pos=%d vs "
                    "export.calibration min_valid_rows=%d min_valid_pos=%d) "
                    "— carve skipped; early stopping keeps the full valid "
                    "set and calibration falls back to it.",
                    n_cal, n_pos_cal, cal_min_rows, cal_min_pos)
            else:
                X_cal, y_cal = X_va[cal_sel], y_va[cal_sel]
                dt_cal = dt_va[cal_sel] if dt_va is not None else None
                w_cal = w_va[cal_sel] if w_va is not None else None
                X_va, y_va = X_va[~cal_sel], y_va[~cal_sel]
                dt_va = dt_va[~cal_sel] if dt_va is not None else None
                w_va = w_va[~cal_sel] if w_va is not None else None
                # Raw-length bookkeeping: valid rows (post all-NA drop) map
                # 1:1, in order, onto valid_mask's True positions.
                vpos = np.where(valid_mask)[0]
                calib_mask[vpos[cal_sel]] = True
                valid_mask[vpos[cal_sel]] = False
                logger.info(
                    "Valid carved: early-stop/selection n=%d, calibration "
                    "holdout n=%d (calibration_split_fraction=%.2f, %s).",
                    int(y_va.shape[0]), n_cal, calib_frac,
                    "time-ordered tail" if dt_cal is not None
                    else "random subset")
        else:
            logger.warning(
                "calibration_split_fraction=%.2f would leave an empty half "
                "(n_valid=%d) — carve skipped; calibration will fall back to "
                "the early-stopping valid set.", calib_frac, n_va_rows)

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
        dt_train=dt_tr, dt_valid=dt_va, dt_oot=dt_oot,
        w_train=w_tr, w_valid=w_va, w_oot=w_oot,
        X_calib=X_cal, y_calib=y_cal, dt_calib=dt_cal, w_calib=w_cal,
        calib_mask=calib_mask,
    )
