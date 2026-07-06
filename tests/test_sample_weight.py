"""Sample weights: config mapping semantics + weights actually reach training."""
import numpy as np
import pandas as pd
import xgboost as xgb

from wdm.model.dataset import build_sample_weights


SW_CFG = {"column": "credit_1v1",
          "mapping": {3: 6.5833, 2: 2.4167, 1: 1.0},
          "default": 1.0}


def test_mapping_default_nan_and_negative():
    col = pd.Series([3, 2, 1, 0, -1, np.nan, 7, "2", "bad"])
    w = build_sample_weights(col, SW_CFG)
    expected = np.array([6.5833, 2.4167, 1.0,   # tiers
                         1.0, 1.0,              # 0 / -1 -> default
                         1.0, 1.0,              # NaN / unmapped -> default
                         2.4167,                # numeric string matches
                         1.0])                  # unparseable -> default
    assert w.dtype == np.float64
    assert np.allclose(w, expected)


def test_dmatrix_weight_roundtrip():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.randint(0, 2, size=100)
    w = build_sample_weights(pd.Series(rng.choice([0, 1, 2, 3], size=100)), SW_CFG)
    dmat = xgb.DMatrix(X, label=y, weight=w)
    assert np.allclose(dmat.get_weight(), w.astype(np.float32))


def test_weights_change_training():
    """Extreme weights on a planted tier signal must move the predictions."""
    rng = np.random.RandomState(1)
    n = 2000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    # Two positive sub-populations driven by different features
    tier3 = X[:, 0] > 1.0
    tier1 = X[:, 1] > 1.0
    y = (tier3 | tier1).astype(np.int64)
    w_uniform = np.ones(n)
    w_tier3 = np.where(tier3, 50.0, 1.0)

    params = {"objective": "binary:logistic", "max_depth": 3,
              "verbosity": 0, "seed": 0}
    b_uni = xgb.train(params, xgb.DMatrix(X, label=y, weight=w_uniform),
                      num_boost_round=20)
    b_w = xgb.train(params, xgb.DMatrix(X, label=y, weight=w_tier3),
                    num_boost_round=20)
    probe = xgb.DMatrix(X)
    p_uni = b_uni.predict(probe)
    p_w = b_w.predict(probe)
    assert not np.allclose(p_uni, p_w), "weights had no effect on training"
    # The weighted model must score the up-weighted tier-3 positives higher
    # relative to the tier-1 positives than the uniform model does.
    gap_w = p_w[tier3 & (y == 1)].mean() - p_w[tier1 & ~tier3 & (y == 1)].mean()
    gap_uni = p_uni[tier3 & (y == 1)].mean() - p_uni[tier1 & ~tier3 & (y == 1)].mean()
    assert gap_w > gap_uni
