"""Final XGBoost training on selected features with early stopping on valid PR-AUC."""
import logging
from typing import Dict

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class _RoundProgress(xgb.callback.TrainingCallback):
    """Log train/valid metrics every `every` boosting rounds via the wdm logger
    (instead of xgboost's print-based verbose_eval)."""

    def __init__(self, total_rounds, every=50):
        self.total_rounds = int(total_rounds)
        self.every = int(every)

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.every == 0:
            parts = []
            for split, metrics in evals_log.items():
                for metric, values in metrics.items():
                    parts.append("{0}-{1}={2:.5f}".format(split, metric, values[-1]))
            logger.info("final train round %d/%d: %s",
                        epoch + 1, self.total_rounds, " ".join(parts))
        return False


def train_final(best_params, X_tr, y_tr, X_va, y_va, cfg, w_tr=None, w_va=None):
    """Final fit. w_tr enters the training loss only; dvalid stays UNWEIGHTED
    by default so early stopping selects rounds on the per-person valid metric
    (pass w_va explicitly to opt into weighted early stopping)."""
    base = dict(cfg["training"]["xgb_base_params"])
    # xgboost early-stops on the LAST entry of eval_metric. Reorder so the
    # configured early-stop target sits last; otherwise best_iteration would
    # silently track whatever metric happens to close the list (e.g. ROC-AUC
    # while the deployment goal is PR-AUC).
    metrics = list(cfg["training"]["eval_metrics"])
    es_metric = cfg["training"]["early_stop_metric"]
    metrics = [m for m in metrics if m != es_metric] + [es_metric]
    base["eval_metric"] = metrics
    # Explicit seed (training.random_seed) — reproducible final fits under
    # colsample/subsample sampling instead of xgboost's implicit default.
    base.setdefault("seed", int(cfg["training"]["random_seed"]))

    params = dict(base)
    resolved = dict(best_params)
    n_rounds = int(resolved.pop("n_estimators", 500))
    params.update(resolved)

    if w_tr is not None:
        logger.info("final train: weighted loss (sum w=%.1f over %d rows)",
                    float(np.sum(w_tr)), len(y_tr))
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)

    logger.info("final train: up to %d rounds, early stopping on valid %s "
                "after 50 stale rounds", n_rounds, es_metric)
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
        callbacks=[_RoundProgress(n_rounds, every=50)],
    )
    logger.info("final train done: best_iteration=%s best_score=%s",
                getattr(booster, "best_iteration", "n/a"),
                getattr(booster, "best_score", "n/a"))

    return booster, evals_result
