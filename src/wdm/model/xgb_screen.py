"""Shared kernel for the three model-driven feature screens.

probing (Stage-1 signal), null_importance (Stage-1.5 screen) and the Stage-2
exploratory pruner each fit a cheap XGBoost and read per-feature importance
from it. Two pieces must stay identical across all three — the class-imbalance
default (scale_pos_weight = neg/pos, matching the deployed model's tuning
regime) and the importance extraction aligned to the feature list (including
the fN-key fallback when feature names were not propagated to the DMatrix).
This module is the single implementation of both.
"""
import numpy as np
import pandas as pd


def apply_scale_pos_weight_default(params, y):
    """Set scale_pos_weight = neg/pos unless explicitly configured.

    Aligns every screening model with the deployed model's imbalance regime:
    an unweighted screen under-ranks features whose signal concentrates in
    the rare positive class. An explicit xgb_params.scale_pos_weight wins.
    Mutates and returns params.
    """
    if "scale_pos_weight" not in params:
        y = np.asarray(y)
        n_pos = float(np.sum(y == 1))
        n_neg = float(y.size - n_pos)
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos
    return params


def named_importance(booster, feature_names, importance_type="gain"):
    """Per-feature importance dict aligned to feature_names.

    Features unused by any split are 0.0. Keys returned as fN (feature names
    not propagated to the DMatrix) are mapped by position; unknown keys are
    ignored.
    """
    raw = booster.get_score(importance_type=importance_type)
    known = set(feature_names)
    out = {name: 0.0 for name in feature_names}
    for k, v in raw.items():
        if k in known:
            out[k] = float(v)
        elif k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_names):
                out[feature_names[idx]] = float(v)
    return out


def gain_vector(booster, feature_names, importance_type="gain"):
    """float64 ndarray aligned to feature_names."""
    d = named_importance(booster, feature_names, importance_type)
    return np.array([d[f] for f in feature_names], dtype=np.float64)


def gain_series(booster, feature_names, importance_type="gain"):
    """pd.Series indexed by feature_names."""
    d = named_importance(booster, feature_names, importance_type)
    return pd.Series([d[f] for f in feature_names], index=list(feature_names))


def importance_frame(booster, feature_names):
    """DataFrame[feature, gain, weight, cover] aligned to feature_names."""
    rows = {name: {"gain": 0.0, "weight": 0.0, "cover": 0.0}
            for name in feature_names}
    for imp_type in ("gain", "weight", "cover"):
        d = named_importance(booster, feature_names, imp_type)
        for name in feature_names:
            rows[name][imp_type] = d[name]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "feature"
    return df.reset_index()
