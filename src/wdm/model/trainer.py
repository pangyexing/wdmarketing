"""Final XGBoost training on selected features with early stopping on valid PR-AUC."""
import logging
from typing import Dict

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


def train_final(best_params, X_tr, y_tr, X_va, y_va, cfg):
    base = dict(cfg["training"]["xgb_base_params"])
    base["eval_metric"] = list(cfg["training"]["eval_metrics"])

    params = dict(base)
    resolved = dict(best_params)
    n_rounds = int(resolved.pop("n_estimators", 500))
    params.update(resolved)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )

    return booster, evals_result
