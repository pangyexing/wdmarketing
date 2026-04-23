"""Missing value handling — strict separation between analysis and training.

Key contract:
  * Stage-1 analysis modules call `to_nan_array(series, spec)` and work on the
    NaN-masked array. NaN never gets replaced by a fill value during analysis
    — this prevents -999 from polluting correlation / PSI / IV statistics.
  * Stage-2 `model/dataset.py` calls `fit_missing` on train only, then
    `apply_missing_for_training` on train/valid/oot to produce the final
    matrix fed to XGBoost.
  * `exporter.py` persists the resolved rules + fit stats to missing_spec.json
    so `predict.py` can replay (never re-fit) at deploy time.

The default rules (0/negative/empty treated as missing, fill with -999) are
aggressive and only valid when users confirm features are supposed to be
positive. For UCI bank_marketing, per-column overrides disable the defaults
on columns where 0/negative are legitimate (balance, previous, duration, ...).
"""
import dataclasses
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FILL_STRATEGIES = ("constant", "median", "mean", "zero", "special", "keep_nan")


@dataclasses.dataclass
class MissingSpec:
    sentinels: List[Any] = dataclasses.field(default_factory=list)
    treat_negative_as_missing: bool = True
    treat_empty_as_missing: bool = True
    fill_strategy: str = "constant"
    fill_constant: float = -999.0
    fill_value: Optional[float] = None            # only for 'special'
    treat_as_missing_in_woe: bool = False

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def build_missing_spec(cfg):
    """Return Dict[feature_name, MissingSpec]. For every declared or inferred
    feature column, resolve global defaults + per-column overrides.
    """
    missing_cfg = cfg["missing"]
    g = missing_cfg["global"]
    global_spec = MissingSpec(
        sentinels=list(g.get("sentinels", [])),
        treat_negative_as_missing=bool(g.get("treat_negative_as_missing", True)),
        treat_empty_as_missing=bool(g.get("treat_empty_as_missing", True)),
        fill_strategy=str(g.get("fill_strategy", "constant")),
        fill_constant=float(g.get("fill_constant", -999.0)),
        fill_value=None,
        treat_as_missing_in_woe=bool(g.get("treat_as_missing_in_woe", False)),
    )
    if global_spec.fill_strategy not in FILL_STRATEGIES:
        raise ValueError("Invalid fill_strategy: {0}".format(global_spec.fill_strategy))

    per_col = missing_cfg.get("per_column") or {}
    specs = {}
    for feat, overrides in per_col.items():
        d = dataclasses.asdict(global_spec)
        d.update(overrides or {})
        specs[feat] = MissingSpec.from_dict(d)

    # Columns not in per_column get the global spec on-demand (via get_spec)
    specs["__default__"] = global_spec
    return specs


def get_spec(spec_map, feature):
    if feature in spec_map:
        return spec_map[feature]
    return spec_map["__default__"]


# ---------- core: pre-fill NaN conversion used by ALL Stage-1 analysis ----------

def to_nan_array(series, spec):
    """Convert a pandas Series to (float64 numpy array with NaN for missing,
    boolean mask of 'is_missing').

    Rules applied in order (each can be disabled independently by the spec):
      1. empty strings / None → NaN       (if treat_empty_as_missing)
      2. pandas.NA / numpy.nan already    (always handled — NaN is missing)
      3. values in spec.sentinels → NaN   (exact match)
      4. negative values → NaN            (if treat_negative_as_missing)

    This function does NOT fill anything. Its output feeds PSI/IV/WOE/correlation.
    """
    s = pd.Series(series)
    # Coerce non-numeric (string) to float where possible; non-parseable → NaN
    if not pd.api.types.is_numeric_dtype(s):
        # treat empty string as NaN before coercion
        if spec.treat_empty_as_missing:
            s = s.replace({"": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan})
        s = pd.to_numeric(s, errors="coerce")
    arr = s.astype(np.float64).values
    mask = np.isnan(arr)

    # Sentinels
    if spec.sentinels:
        for v in spec.sentinels:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            mask |= (arr == vf)

    # Negatives
    if spec.treat_negative_as_missing:
        with np.errstate(invalid="ignore"):
            mask |= (arr < 0)

    arr = arr.copy()
    arr[mask] = np.nan
    return arr, mask


# ---------- fit + apply for Stage 2 training ----------

@dataclasses.dataclass
class FittedStats:
    """Per-column fitted statistic that gets persisted to missing_spec.json."""
    feature: str
    fill_strategy: str
    fill_value: float                    # resolved scalar used at predict time
    sentinels: List[Any]
    treat_negative_as_missing: bool
    treat_empty_as_missing: bool
    n_missing_train: int
    missing_rate_train: float

    def to_dict(self):
        d = dataclasses.asdict(self)
        # Ensure JSON-serializable (numpy types → python types)
        d["fill_value"] = None if d["fill_value"] is None or (
            isinstance(d["fill_value"], float) and math.isnan(d["fill_value"])
        ) else float(d["fill_value"])
        d["n_missing_train"] = int(d["n_missing_train"])
        d["missing_rate_train"] = float(d["missing_rate_train"])
        return d


def _resolve_fill_value(arr_nan, spec):
    """Compute the scalar fill value to apply; returns None for keep_nan."""
    non_nan = arr_nan[~np.isnan(arr_nan)]
    if spec.fill_strategy == "keep_nan":
        return None
    if spec.fill_strategy == "constant":
        return float(spec.fill_constant)
    if spec.fill_strategy == "zero":
        return 0.0
    if spec.fill_strategy == "special":
        if spec.fill_value is None:
            raise ValueError("fill_strategy='special' requires fill_value")
        return float(spec.fill_value)
    if spec.fill_strategy == "median":
        if non_nan.size == 0:
            return float(spec.fill_constant)
        return float(np.median(non_nan))
    if spec.fill_strategy == "mean":
        if non_nan.size == 0:
            return float(spec.fill_constant)
        return float(np.mean(non_nan))
    raise ValueError("Unknown fill_strategy: {0}".format(spec.fill_strategy))


def sanity_check_fill_value(arr_nan, fill_value, feature, fill_constant=None):
    """Raise if fill_value falls inside observed [p01, max] — it would be
    indistinguishable from a real value at predict time.
    """
    if fill_value is None:
        return
    non_nan = arr_nan[~np.isnan(arr_nan)]
    if non_nan.size == 0:
        return
    p01 = float(np.quantile(non_nan, 0.01))
    mx = float(np.max(non_nan))
    if p01 <= fill_value <= mx:
        raise ValueError(
            "Fill value {0} for feature '{1}' falls inside observed range "
            "[p01={2:.4f}, max={3:.4f}]. Change fill_strategy or fill_constant."
            .format(fill_value, feature, p01, mx))


def fit_missing(df_train, spec_map, run_sanity_check=True):
    """Fit per-column statistics on the training frame. Returns Dict[feature, FittedStats].

    df_train is an in-memory pandas DataFrame (Stage 2 has <=200 features so it fits).
    """
    stats = {}
    # Sanity check only applies to user-specified sentinel-style fills, not
    # data-driven imputations (median/mean) which are SUPPOSED to land inside
    # the observed range.
    sanity_eligible = {"constant", "zero", "special"}
    for feature in df_train.columns:
        spec = get_spec(spec_map, feature)
        arr, mask = to_nan_array(df_train[feature], spec)
        fill_val = _resolve_fill_value(arr, spec)
        if run_sanity_check and spec.fill_strategy in sanity_eligible:
            sanity_check_fill_value(arr, fill_val, feature, spec.fill_constant)
        stats[feature] = FittedStats(
            feature=feature,
            fill_strategy=spec.fill_strategy,
            fill_value=fill_val if fill_val is not None else float("nan"),
            sentinels=list(spec.sentinels),
            treat_negative_as_missing=spec.treat_negative_as_missing,
            treat_empty_as_missing=spec.treat_empty_as_missing,
            n_missing_train=int(mask.sum()),
            missing_rate_train=float(mask.mean()),
        )
    return stats


def apply_missing_for_training(df, spec_map, fitted):
    """Apply sentinel/negative/empty → NaN → fill per column. Returns a new DataFrame.

    Stage 2 contract: valid/oot DataFrames MUST be transformed with the same
    `fitted` produced on train — never re-fit.
    """
    out = {}
    for feature in df.columns:
        spec = get_spec(spec_map, feature)
        arr, _mask = to_nan_array(df[feature], spec)
        fs = fitted.get(feature)
        if fs is None or fs.fill_strategy == "keep_nan" or (
                isinstance(fs.fill_value, float) and math.isnan(fs.fill_value)
                and fs.fill_strategy == "keep_nan"):
            out[feature] = arr
        else:
            arr = arr.copy()
            arr[np.isnan(arr)] = float(fs.fill_value)
            out[feature] = arr
    return pd.DataFrame(out, index=df.index)


# ---------- persistence (for exporter + predict.py) ----------

# Bump when MissingSpec / FittedStats gain a field that predict.py must understand.
# Older predict.py bundles read this key and refuse to run on a newer, unknown version.
MISSING_SPEC_SCHEMA_VERSION = 1


def dump_missing_spec(path, spec_map, fitted):
    """Write a self-contained JSON bundle usable by predict.py at deploy time."""
    payload = {
        "schema_version": MISSING_SPEC_SCHEMA_VERSION,
        "specs": {feat: s.to_dict() for feat, s in spec_map.items()},
        "fitted": {feat: fs.to_dict() for feat, fs in fitted.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_missing_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    version = int(payload.get("schema_version", 0))
    if version > MISSING_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "missing_spec.json schema_version={0} is newer than this code "
            "understands ({1}). Upgrade the wdm package.".format(
                version, MISSING_SPEC_SCHEMA_VERSION))
    spec_map = {k: MissingSpec.from_dict(v) for k, v in payload["specs"].items()}
    # fitted stays as dicts — predict.py only reads scalars from it
    return spec_map, payload["fitted"]
