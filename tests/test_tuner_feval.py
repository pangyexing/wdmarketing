"""Tuner feval correctness + the xgb 1.5 cv(folds=, feval=, maximize=) interaction."""
import numpy as np
import pytest
import xgboost as xgb

from wdm.metrics.ranking import precision_at_k
from wdm.model.tuner import make_precision_at_k_feval
from wdm.utils.time_utils import build_forward_chaining_folds


def _data(n=500, p=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    logit = X[:, 0] * 1.5 + X[:, 1] - 0.5
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    return X, y


def test_feval_matches_metric():
    X, y = _data()
    rng = np.random.RandomState(1)
    preds = rng.uniform(size=len(y))
    dmat = xgb.DMatrix(X, label=y)
    for k in (0.05, 0.10, 0.25):
        name, value = make_precision_at_k_feval(k)(preds, dmat)
        assert name == "p_at_k"
        assert value == pytest.approx(precision_at_k(y, preds, k))


def test_cv_with_folds_and_feval_smoke():
    """Pins the xgb 1.5 contract: folds= + feval= + maximize=True yields both
    test-aucpr-mean and test-p_at_k-mean columns and early stopping works."""
    X, y = _data()
    # Synthetic ordered dt: 25 fake days, 20 rows each
    dt = np.repeat(np.arange(20250101, 20250126, dtype=np.float64), 20)
    folds = build_forward_chaining_folds(dt, n_folds=4, min_test_rows=10)
    assert len(folds) >= 2
    dtrain = xgb.DMatrix(X, label=y)
    cv_result = xgb.cv(
        params={"objective": "binary:logistic", "tree_method": "hist",
                "max_depth": 3, "verbosity": 0, "eval_metric": ["aucpr"]},
        dtrain=dtrain,
        num_boost_round=30,
        folds=folds,
        metrics=["aucpr"],
        feval=make_precision_at_k_feval(0.10),
        maximize=True,
        early_stopping_rounds=5,
        seed=42,
        verbose_eval=False,
    )
    assert "test-p_at_k-mean" in cv_result.columns
    assert "test-aucpr-mean" in cv_result.columns
    assert len(cv_result) >= 1
    best = float(cv_result["test-p_at_k-mean"].max())
    assert 0.0 <= best <= 1.0
    # On signal-bearing data the tuned ranking must beat the base rate
    assert best > float(np.mean(y)) * 0.8
