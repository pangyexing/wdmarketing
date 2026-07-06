"""Stage-2 candidate → final feature reduction.

When Stage-1 emits stage2_candidate_count features (e.g. 800), score the
candidates with a configurable multivariate signal and slice the
StageTwoData down to final_feature_count base features. Indicator columns
ride along with their surviving bases so the (base, __isnan) pair stays
coherent — matches the historical meaning of final_feature_count
("how many base features end up in the model").

Ranking methods (training.stage2_pruning.ranking_method):
  * "gain"                  — single XGB, raw gain importance. Cheapest;
                              biased toward high-cardinality / many-split
                              features; sensitive to the random seed.
  * "stability" (default)   — train n_seeds XGBs with different seeds, take
                              gain rank-percentile per run, average across
                              runs. Same signal type as gain but absorbs
                              XGB's stochasticity (subsample/colsample).
  * "permutation"           — train one XGB; for each feature, shuffle that
                              column on valid n_permutation_repeats times
                              and average the PR-AUC drop. Closer to "does
                              this column actually move predictions" but
                              costs ~O(n_features × n_repeats) predictions.
  * "permutation_stability" — n_seeds × permutation. Most stable, most
                              expensive. Use when the funnel's selection
                              materially affects downstream metrics.
  * "shap"                  — single XGB; importance = mean(|shap_values|)
                              over a sampled valid set via TreeExplainer.
                              When SHAP is installed this is usually the
                              best signal *and* faster than permutation
                              (TreeSHAP is polynomial in tree depth). When
                              SHAP import fails we fall back to the method
                              named by shap_fallback (default "stability");
                              set shap_fallback="raise" to fail loud.
  * "shap_stability"        — n_seeds × shap. Same fallback contract.

No-op when training.stage2_candidate_count is not set (the funnel is opt-in)
or when the candidate pool is already ≤ final_feature_count. The legacy path
then sees this function as a pass-through.
"""
import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from wdm.metrics.pr_auc import pr_auc
from wdm.model.dataset import StageTwoData

logger = logging.getLogger(__name__)


# --- exploratory XGB ----------------------------------------------------------

def _train_exploratory(X_tr, y_tr, X_va, y_va, prune_cfg, seed=0):
    params = dict(prune_cfg.get("xgb_params") or {})
    params["seed"] = int(seed)
    # Align the imbalance regime with the deployed model (which tunes
    # scale_pos_weight): an unweighted exploratory ranker under-ranks
    # features whose signal concentrates in the rare positive class.
    # Explicit xgb_params.scale_pos_weight still wins.
    if "scale_pos_weight" not in params:
        n_pos = float(np.sum(np.asarray(y_tr) == 1))
        n_neg = float(np.asarray(y_tr).size - n_pos)
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos
    n_rounds = int(prune_cfg.get("num_boost_round", 200))
    early_stop = int(prune_cfg.get("early_stopping_rounds", 30))
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )


def _predict_valid(booster, X):
    dmat = xgb.DMatrix(X)
    try:
        return booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
    except (TypeError, xgb.core.XGBoostError):
        return booster.predict(dmat, ntree_limit=booster.best_ntree_limit)


def _gain_series(booster, feature_list):
    raw = booster.get_score(importance_type="gain")
    rename = {"f{0}".format(i): feature_list[i] for i in range(len(feature_list))}
    gain = {rename.get(k, k): v for k, v in raw.items()}
    return pd.Series(
        [float(gain.get(f, 0.0)) for f in feature_list], index=feature_list)


# --- ranking signals ---------------------------------------------------------

def _rank_gain(data, prune_cfg, seed=0):
    booster = _train_exploratory(
        data.X_train, data.y_train, data.X_valid, data.y_valid, prune_cfg, seed=seed)
    return _gain_series(booster, data.feature_list)


def _rank_stability(data, prune_cfg, n_seeds, base_seed=0):
    """Mean rank-percentile of gain across n_seeds XGB runs.

    Rank-percentile (not raw gain) because gain magnitudes vary across seeds;
    averaging percentiles gives every run an equal vote on relative ordering.
    """
    accum = pd.Series(0.0, index=data.feature_list)
    for i in range(n_seeds):
        gain = _rank_gain(data, prune_cfg, seed=base_seed + i)
        accum += gain.rank(method="average", pct=True)
    return accum / float(n_seeds)


def _rank_permutation(data, prune_cfg, seed=0):
    booster = _train_exploratory(
        data.X_train, data.y_train, data.X_valid, data.y_valid,
        prune_cfg, seed=seed)
    n_repeats = int(prune_cfg.get("n_permutation_repeats", 3))
    rng = np.random.RandomState(int(prune_cfg.get("permutation_seed", seed)))

    base_score = pr_auc(data.y_valid, _predict_valid(booster, data.X_valid))

    # One writable copy of X_valid; restore each column after shuffling so the
    # next feature's measurement starts from the unmodified valid matrix.
    X_perm = data.X_valid.copy()
    n_features = X_perm.shape[1]
    drops = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        col_backup = X_perm[:, j].copy()
        repeat_drops = np.zeros(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            rng.shuffle(X_perm[:, j])
            score = pr_auc(data.y_valid, _predict_valid(booster, X_perm))
            repeat_drops[r] = base_score - score
        X_perm[:, j] = col_backup
        drops[j] = float(repeat_drops.mean())
    return pd.Series(drops, index=data.feature_list)


def _rank_permutation_stability(data, prune_cfg, n_seeds, base_seed=0):
    accum = pd.Series(0.0, index=data.feature_list)
    for i in range(n_seeds):
        perm = _rank_permutation(data, prune_cfg, seed=base_seed + i)
        accum += perm.rank(method="average", pct=True)
    return accum / float(n_seeds)


def _try_import_shap():
    """Return the shap module, or None plus the captured exception. Mirrors
    the guard already used by plots/model_plots.py — SHAP imports fail on
    some machines (notably with mismatched numba/llvmlite stacks)."""
    try:
        import shap
        return shap, None
    except Exception as exc:
        return None, exc


def _shap_importance(booster, X_sample, shap_mod):
    """mean(|shap_values|) per feature. Defensive against SHAP variants that
    return a list (multi-class) or an Explanation object instead of a 2D array."""
    explainer = shap_mod.TreeExplainer(booster)
    sv = explainer.shap_values(X_sample)
    if hasattr(sv, "values"):  # shap.Explanation
        sv = sv.values
    if isinstance(sv, list):   # multi-class API: pick the positive-class slice
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:           # (n_classes, n_samples, n_features) on some versions
        sv = sv[-1]
    return np.abs(sv).mean(axis=0)


def _sample_valid(X_valid, sample_size, seed):
    if not sample_size or sample_size >= len(X_valid):
        return X_valid
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(len(X_valid), size=int(sample_size), replace=False)
    return X_valid[idx]


def _rank_shap(data, prune_cfg, seed=0):
    shap_mod, err = _try_import_shap()
    if shap_mod is None:
        raise RuntimeError("shap import failed: {0}".format(err))
    booster = _train_exploratory(
        data.X_train, data.y_train, data.X_valid, data.y_valid,
        prune_cfg, seed=seed)
    X_sample = _sample_valid(
        data.X_valid, prune_cfg.get("shap_sample_size", 5000), seed)
    importance = _shap_importance(booster, X_sample, shap_mod)
    return pd.Series(importance, index=data.feature_list)


def _rank_shap_stability(data, prune_cfg, n_seeds, base_seed=0):
    accum = pd.Series(0.0, index=data.feature_list)
    for i in range(n_seeds):
        s = _rank_shap(data, prune_cfg, seed=base_seed + i)
        accum += s.rank(method="average", pct=True)
    return accum / float(n_seeds)


_RANKERS = {
    "gain":                  lambda d, c, n: _rank_gain(d, c),
    "stability":             lambda d, c, n: _rank_stability(d, c, n),
    "permutation":           lambda d, c, n: _rank_permutation(d, c),
    "permutation_stability": lambda d, c, n: _rank_permutation_stability(d, c, n),
    "shap":                  lambda d, c, n: _rank_shap(d, c),
    "shap_stability":        lambda d, c, n: _rank_shap_stability(d, c, n),
}

_SHAP_METHODS = {"shap", "shap_stability"}


def _resolve_shap_method(method, prune_cfg):
    """If `method` requires SHAP and SHAP is unavailable, return the fallback
    method (or raise). Otherwise return `method` unchanged."""
    if method not in _SHAP_METHODS:
        return method
    shap_mod, err = _try_import_shap()
    if shap_mod is not None:
        return method
    fallback = prune_cfg.get("shap_fallback", "stability")
    if fallback in (None, "raise"):
        raise RuntimeError(
            "ranking_method={0!r} but shap import failed ({1}). Set "
            "stage2_pruning.shap_fallback to a non-shap method (e.g. "
            "'stability') to allow graceful degradation."
            .format(method, err))
    fallback = str(fallback).lower()
    if fallback in _SHAP_METHODS:
        raise ValueError(
            "shap_fallback={0!r} also requires SHAP; pick a non-shap method."
            .format(fallback))
    if fallback not in _RANKERS:
        non_shap = sorted(set(_RANKERS) - _SHAP_METHODS)
        raise ValueError(
            "Unknown shap_fallback: {0!r} (expected one of {1})"
            .format(fallback, non_shap))
    logger.warning("SHAP unavailable (%s) — falling back to ranking_method=%r.",
                   err, fallback)
    return fallback


def _score_features(data, prune_cfg):
    requested = str(prune_cfg.get("ranking_method", "stability")).lower()
    if requested not in _RANKERS:
        raise ValueError(
            "Unknown stage2_pruning.ranking_method: {0!r} "
            "(expected one of {1})".format(requested, sorted(_RANKERS)))
    method = _resolve_shap_method(requested, prune_cfg)
    n_seeds = int(prune_cfg.get("n_seeds", 3))
    logger.info("Stage-2 pruning ranker: %s (requested=%s, n_seeds=%d)",
                method, requested, n_seeds)
    series = _RANKERS[method](data, prune_cfg, n_seeds)
    return method, series.rename("score")


# --- main entrypoint ---------------------------------------------------------

def maybe_prune_to_final(data: StageTwoData, cfg: Dict,
                         run_dir: Optional[Path] = None) -> StageTwoData:
    """Shrink the candidate pool to final_feature_count via the configured ranker.

    The funnel fires only when ``training.stage2_candidate_count`` is set
    (the documented opt-in) AND the candidate pool exceeds
    ``final_feature_count``. With the funnel disabled the loaded feature list
    is trained as-is — a screen the config never asked for must not run.
    When pruning fires and ``run_dir`` is supplied, writes
    ``exploratory_importance.csv`` (with ``score`` + ``kept``) and
    ``pruned_features.txt`` so the reduction is auditable.
    """
    training_cfg = cfg["training"]
    final_n = int(training_cfg["final_feature_count"])
    candidate_n = training_cfg.get("stage2_candidate_count")
    n_base = len(data.base_feature_list)
    if not candidate_n:
        if n_base > final_n:
            logger.info(
                "Stage-2 funnel disabled (stage2_candidate_count not set) — "
                "training on all %d loaded features. final_feature_count=%d "
                "only gates the funnel; set stage2_candidate_count to enable "
                "the exploratory pruning.", n_base, final_n)
        return data
    if n_base <= final_n:
        logger.info("Stage-2 pruning skipped: candidate pool (%d base features) "
                    "≤ final_feature_count (%d).", n_base, final_n)
        return data

    prune_cfg = training_cfg.get("stage2_pruning") or {}
    logger.info("Stage-2 pruning: %d candidates → top %d base.", n_base, final_n)
    method, scores = _score_features(data, prune_cfg)

    base_set = set(data.base_feature_list)
    base_scores = scores[scores.index.isin(base_set)].sort_values(ascending=False)
    kept_bases = base_scores.head(final_n).index.tolist()
    kept_base_set = set(kept_bases)

    kept_indicators = [
        ind for ind in data.indicator_features
        if ind[: -len("__isnan")] in kept_base_set
    ]
    new_feature_list = list(kept_bases) + kept_indicators

    pos = {f: i for i, f in enumerate(data.feature_list)}
    cols = np.array([pos[f] for f in new_feature_list], dtype=np.int64)

    # Slice spec_map / fitted so the deploy bundle's missing_spec.json only
    # carries entries for surviving bases. spec_map["__default__"] is always
    # kept because get_spec() falls back to it for any not-explicitly-mapped
    # feature; dropping it would silently change predict-time behavior.
    kept_spec_map = {
        k: v for k, v in data.spec_map.items()
        if k == "__default__" or k in kept_base_set
    }
    kept_fitted = {
        k: v for k, v in data.fitted.items() if k in kept_base_set
    }

    pruned = dataclasses.replace(
        data,
        X_train=data.X_train[:, cols],
        X_valid=data.X_valid[:, cols],
        X_oot=data.X_oot[:, cols],
        X_calib=(data.X_calib[:, cols]
                 if getattr(data, "X_calib", None) is not None else None),
        feature_list=new_feature_list,
        base_feature_list=kept_bases,
        indicator_features=kept_indicators,
        spec_map=kept_spec_map,
        fitted=kept_fitted,
    )

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        kept_set = set(new_feature_list)
        imp_out = pd.DataFrame({
            "feature": list(scores.index),
            "score": scores.values,
        }).sort_values("score", ascending=False).reset_index(drop=True)
        imp_out["kept"] = imp_out["feature"].isin(kept_set)
        imp_out["ranking_method"] = method
        imp_out.to_csv(run_dir / "exploratory_importance.csv", index=False)
        (run_dir / "pruned_features.txt").write_text(
            "\n".join(new_feature_list) + "\n", encoding="utf-8")

    logger.info("Stage-2 pruning done: kept %d base + %d indicator "
                "(dropped %d base).",
                len(kept_bases), len(kept_indicators), n_base - len(kept_bases))
    return pruned
