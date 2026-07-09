"""Hyperopt driver for XGBoost top-K ranking.

Objective (training.tuner_objective):
  * aucpr          — maximize CV PR-AUC (legacy default).
  * precision_at_k — maximize CV Precision@K (custom feval at
    training.top_k_pct), matching the top-K deployment goal directly.

CV folds (training.cv_strategy):
  * stratified   — shuffled stratified xgb.cv (legacy default).
  * time_forward — forward-chaining expanding-window folds built from per-row
    dt, so parameter selection respects the time-based deployment split.

Per-row weights (training.sample_weight) enter the training loss only; the
precision_at_k feval reads labels unweighted (per-person deployment metric).

Trials are pickled to disk so tuning can be resumed after interruption — but
only under the SAME objective (losses across objectives are incomparable).
"""
import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from wdm.metrics.ranking import precision_at_k
from wdm.utils.progress import fmt_duration
from wdm.utils.time_utils import build_forward_chaining_folds

logger = logging.getLogger(__name__)


def make_precision_at_k_feval(k_pct):
    """xgb.cv custom feval: unweighted Precision@K (k as fraction)."""
    k = float(k_pct)

    def feval(preds, dmat):
        y = dmat.get_label()
        return "p_at_k", float(precision_at_k(y, preds, k))
    return feval


def _search_space(cfg):
    space = {
        "max_depth": hp.choice("max_depth", [3, 4, 5, 6]),
        "min_child_weight": hp.quniform("min_child_weight", 1, 50, 1),
        "subsample": 1.0,
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "learning_rate": hp.loguniform("learning_rate",
                                       math.log(0.01), math.log(0.2)),
        "reg_lambda": hp.loguniform("reg_lambda", math.log(0.1), math.log(10.0)),
        "reg_alpha": hp.loguniform("reg_alpha", math.log(1e-4), math.log(1.0)),
        "gamma": hp.loguniform("gamma", math.log(1e-3), math.log(5.0)),
        "scale_pos_weight_choice": hp.choice("scale_pos_weight_choice",
                                             ["auto", "half_auto", "double_auto", "one"]),
        "n_estimators": hp.quniform("n_estimators", 200, 300, 25),
    }
    return space


def _resolve_params(h_params, auto_spw):
    spw_choice = h_params["scale_pos_weight_choice"]
    if spw_choice == "auto":
        spw = float(auto_spw)
    elif spw_choice == "half_auto":
        spw = float(auto_spw) * 0.5
    elif spw_choice == "double_auto":
        spw = float(auto_spw) * 2.0
    else:
        spw = 1.0
    return {
        "max_depth": int(h_params["max_depth"]),
        "min_child_weight": float(h_params["min_child_weight"]),
        "subsample": float(h_params["subsample"]),
        "colsample_bytree": float(h_params["colsample_bytree"]),
        "learning_rate": float(h_params["learning_rate"]),
        "reg_lambda": float(h_params["reg_lambda"]),
        "reg_alpha": float(h_params["reg_alpha"]),
        "gamma": float(h_params["gamma"]),
        "scale_pos_weight": float(spw),
        "n_estimators": int(h_params["n_estimators"]),
    }


def run_hyperopt(X_train, y_train, cfg, trials_path=None, max_evals=None,
                 dt_train=None, w_train=None):
    """Run TPE search. Returns (best_resolved_params, best_loss, trials_obj).

    dt_train: per-row yyyymmdd for cv_strategy='time_forward' (else unused).
    w_train: per-row loss weights (training.sample_weight) or None.
    """
    max_evals = int(max_evals or cfg["training"]["hyperopt_max_evals"])
    seed = int(cfg["training"]["random_seed"])
    # hyperopt 0.2.6 expects the numpy.random.Generator API (np.random.default_rng),
    # not the legacy RandomState.
    rng = np.random.default_rng(seed)

    cv_folds = int(cfg["training"]["cv_folds"])
    eval_metrics = list(cfg["training"]["eval_metrics"])
    base = dict(cfg["training"]["xgb_base_params"])
    base["eval_metric"] = eval_metrics
    # Silence xgb CV log flood
    base.setdefault("verbosity", 0)
    # Explicit seed: don't rely on xgboost's implicit default (0) — CV trials
    # must be reproducible under colsample/subsample sampling.
    base.setdefault("seed", seed)

    objective_name = cfg["training"].get("tuner_objective", "aucpr")
    cv_strategy = cfg["training"].get("cv_strategy", "stratified")
    if objective_name == "precision_at_k":
        k_pct = float(cfg["training"].get("top_k_pct", 0.10))
        feval = make_precision_at_k_feval(k_pct)
        metric_col = "test-p_at_k-mean"
        logger.info("tuner objective: precision_at_k (k=%.2f)", k_pct)
    else:
        feval = None
        metric_col = "test-aucpr-mean"
        logger.info("tuner objective: aucpr")

    folds = None
    if cv_strategy == "time_forward":
        if dt_train is None:
            raise ValueError("cv_strategy='time_forward' requires per-row dt_train "
                             "(is data.time_column configured?)")
        folds = build_forward_chaining_folds(dt_train, cv_folds)
        logger.info("time_forward CV: %d folds, test sizes %s", len(folds),
                    [int(te.size) for _tr, te in folds])

    if w_train is not None:
        # Weight-aware class ratio so the scale_pos_weight choices keep their meaning.
        w = np.asarray(w_train, dtype=np.float64)
        auto_spw = float(w[y_train == 0].sum()) / max(1e-12, float(w[y_train == 1].sum()))
    else:
        n_pos = int(y_train.sum())
        n_neg = int(y_train.size - n_pos)
        auto_spw = n_neg / max(1, n_pos)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)

    # Resume trials if a pickle exists
    trials = Trials()
    already = 0
    if trials_path and Path(trials_path).is_file():
        try:
            with open(trials_path, "rb") as f:
                trials = pickle.load(f)
            already = len(trials.trials)
            logger.info("Resumed %d prior trials from %s", already, trials_path)
        except Exception as e:
            logger.warning("Could not resume trials: %s (starting fresh)", e)
        if already:
            prior_obj = trials.trials[0]["result"].get("objective", "aucpr")
            if prior_obj != objective_name:
                raise ValueError(
                    "trials.pkl at {0} was tuned with objective '{1}' but the config now "
                    "asks for '{2}' — losses are incomparable. Use a new --run-id (or "
                    "delete the stale trials.pkl).".format(trials_path, prior_obj,
                                                           objective_name))

    effective_evals = max(max_evals, already + 1)

    # Per-trial progress state (ETA based on trials run in THIS process only)
    progress = {"done": already, "best": -np.inf, "t0": time.time()}

    def _objective(h_params):
        p = _resolve_params(h_params, auto_spw)
        n_rounds = p.pop("n_estimators")
        params = dict(base)
        params.update({k: v for k, v in p.items() if k not in ("n_estimators",)})
        cv_kwargs = dict(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_rounds,
            metrics=["aucpr"],
            early_stopping_rounds=25,
            seed=seed,
            verbose_eval=False,
        )
        if folds is not None:
            cv_kwargs["folds"] = folds  # nfold/stratified/shuffle are ignored with folds
        else:
            cv_kwargs.update(nfold=cv_folds, stratified=True, shuffle=True)
        if feval is not None:
            # feval is appended LAST in eval results; with early stopping keyed on
            # the last metric, maximize=True is required (else it minimizes P@K).
            cv_kwargs.update(feval=feval, maximize=True)
        cv_result = xgb.cv(**cv_kwargs)
        best_score = float(cv_result[metric_col].max())
        best_round = int(cv_result[metric_col].idxmax() + 1)
        aucpr_at_best = float(cv_result["test-aucpr-mean"].iloc[best_round - 1])

        progress["done"] += 1
        progress["best"] = max(progress["best"], best_score)
        ran_here = progress["done"] - already
        elapsed = time.time() - progress["t0"]
        eta = elapsed / ran_here * (effective_evals - progress["done"])
        logger.info("Hyperopt trial %d/%d: cv %s=%.5f (best %.5f, aucpr@best=%.5f) "
                    "elapsed %s, ETA %s",
                    progress["done"], effective_evals, objective_name, best_score,
                    progress["best"], aucpr_at_best,
                    fmt_duration(elapsed), fmt_duration(eta))

        return {"loss": -best_score, "status": STATUS_OK,
                "params": p, "n_estimators": n_rounds,
                "objective": objective_name,
                "cv_aucpr_at_best": aucpr_at_best,
                "best_round": best_round}

    best = fmin(
        fn=_objective,
        space=_search_space(cfg),
        algo=tpe.suggest,
        max_evals=effective_evals,
        trials=trials,
        rstate=rng,
        show_progressbar=False,
    )

    if trials_path:
        Path(trials_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trials_path, "wb") as f:
            pickle.dump(trials, f)

    # Recover best resolved params from the top trial
    losses = [t["result"]["loss"] for t in trials.trials]
    best_idx = int(np.argmin(losses))
    best_result = trials.trials[best_idx]["result"]
    best_params = dict(best_result["params"])
    best_params["n_estimators"] = int(best_result.get("best_round",
                                                      best_result["n_estimators"]))
    return best_params, float(best_result["loss"]), trials
