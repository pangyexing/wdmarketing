"""Hyperopt driver for XGBoost top-K ranking.

Objective: maximize CV PR-AUC (aucpr). PR-AUC is a better proxy for top-K
precision on imbalanced data than ROC-AUC.

Trials are pickled to disk so tuning can be resumed after interruption.
"""
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

logger = logging.getLogger(__name__)


def _search_space(cfg):
    space = {
        "max_depth": hp.choice("max_depth", [3, 4, 5, 6]),
        "min_child_weight": hp.quniform("min_child_weight", 1, 50, 1),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "learning_rate": hp.loguniform("learning_rate",
                                       math.log(0.01), math.log(0.3)),
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


def run_hyperopt(X_train, y_train, cfg, trials_path=None, max_evals=None):
    """Run TPE search. Returns (best_resolved_params, best_loss, trials_obj)."""
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

    n_pos = int(y_train.sum())
    n_neg = int(y_train.size - n_pos)
    auto_spw = n_neg / max(1, n_pos)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    def _objective(h_params):
        p = _resolve_params(h_params, auto_spw)
        n_rounds = p.pop("n_estimators")
        params = dict(base)
        params.update({k: v for k, v in p.items() if k not in ("n_estimators",)})
        cv_result = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_rounds,
            nfold=cv_folds,
            stratified=True,
            metrics=["aucpr"],
            early_stopping_rounds=25,
            seed=seed,
            shuffle=True,
            verbose_eval=False,
        )
        best_score = float(cv_result["test-aucpr-mean"].max())
        return {"loss": -best_score, "status": STATUS_OK,
                "params": p, "n_estimators": n_rounds,
                "best_round": int(cv_result["test-aucpr-mean"].idxmax() + 1)}

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

    effective_evals = max(max_evals, already + 1)

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
