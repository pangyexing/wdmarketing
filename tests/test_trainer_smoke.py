"""Smoke coverage for wdm.model.trainer.train_final — the core Stage-2 fit
had no dedicated test. Tiny data, few rounds; asserts the contract, not the
model quality."""
import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from wdm.model.trainer import train_final  # noqa: E402


def _tiny_data(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4).astype(np.float32)
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.randn(n)
    y = (logit > 0.5).astype(np.int64)
    return X, y


def _cfg():
    return {
        "training": {
            "xgb_base_params": {"objective": "binary:logistic",
                                "tree_method": "hist", "verbosity": 0},
            "eval_metrics": ["aucpr", "auc"],
            "early_stop_metric": "aucpr",
            "random_seed": 42,
        },
    }


def test_train_final_smoke():
    X, y = _tiny_data()
    best_params = {"n_estimators": 20, "max_depth": 2, "eta": 0.3}
    booster, evals_result = train_final(
        best_params, X[:300], y[:300], X[300:], y[300:], _cfg())

    preds = booster.predict(xgb.DMatrix(X[300:]))
    assert preds.shape == (100,)
    assert np.all((preds >= 0) & (preds <= 1))
    # Signal is strong; the model must beat random ranking on valid.
    from wdm.metrics.pr_auc import roc_auc
    assert roc_auc(y[300:], preds) > 0.8
    # evals_result carries both configured metrics for both watchlist entries.
    assert set(evals_result) == {"train", "valid"}
    assert set(evals_result["valid"]) == {"aucpr", "auc"}


def test_train_final_early_stops_on_aucpr():
    """eval_metrics lists auc last, but best_iteration must track aucpr:
    train_final moves early_stop_metric to the end of eval_metric (xgboost
    early-stops on the last entry)."""
    X, y = _tiny_data()
    best_params = {"n_estimators": 60, "max_depth": 2, "eta": 0.3}
    booster, evals_result = train_final(
        best_params, X[:300], y[:300], X[300:], y[300:], _cfg())
    assert booster.best_score == pytest.approx(
        max(evals_result["valid"]["aucpr"]), abs=1e-9)
    assert booster.best_score == pytest.approx(
        evals_result["valid"]["aucpr"][booster.best_iteration], abs=1e-9)


def test_train_final_weighted_loss_changes_model():
    X, y = _tiny_data()
    best_params = {"n_estimators": 20, "max_depth": 2, "eta": 0.3}
    b_unw, _ = train_final(best_params, X[:300], y[:300], X[300:], y[300:], _cfg())
    w = np.where(y[:300] == 1, 10.0, 1.0)
    b_w, _ = train_final(best_params, X[:300], y[:300], X[300:], y[300:], _cfg(),
                         w_tr=w)
    p_unw = b_unw.predict(xgb.DMatrix(X[300:]))
    p_w = b_w.predict(xgb.DMatrix(X[300:]))
    # Upweighting positives must shift predicted probabilities upward on average.
    assert p_w.mean() > p_unw.mean()
