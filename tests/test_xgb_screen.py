"""wdm.model.xgb_screen — the shared kernel behind probing, null_importance
and the Stage-2 pruner. Importance extraction must align to the feature list
with the fN-key fallback, and the imbalance default must match neg/pos."""
import numpy as np
import pytest

from wdm.model.xgb_screen import (
    apply_scale_pos_weight_default, gain_series, gain_vector,
    importance_frame, named_importance)

xgb = pytest.importorskip("xgboost")


class _FakeBooster(object):
    def __init__(self, scores_by_type):
        self._scores = scores_by_type

    def get_score(self, importance_type="gain"):
        return dict(self._scores.get(importance_type, {}))


def test_named_importance_alignment_and_zero_fill():
    b = _FakeBooster({"gain": {"b": 3.0, "a": 1.5}})
    d = named_importance(b, ["a", "b", "c"])
    assert d == {"a": 1.5, "b": 3.0, "c": 0.0}


def test_named_importance_fn_key_fallback():
    b = _FakeBooster({"gain": {"f0": 2.0, "f2": 5.0, "f9": 7.0}})
    d = named_importance(b, ["a", "b", "c"])
    # f9 is out of range → ignored
    assert d == {"a": 2.0, "b": 0.0, "c": 5.0}


def test_gain_vector_and_series_shapes():
    b = _FakeBooster({"gain": {"a": 1.0}})
    v = gain_vector(b, ["a", "b"])
    assert v.dtype == np.float64 and list(v) == [1.0, 0.0]
    s = gain_series(b, ["a", "b"])
    assert list(s.index) == ["a", "b"] and list(s.values) == [1.0, 0.0]


def test_importance_frame_columns_and_order():
    b = _FakeBooster({"gain": {"a": 1.0}, "weight": {"b": 2.0},
                      "cover": {"f0": 3.0}})
    df = importance_frame(b, ["a", "b"])
    assert list(df.columns) == ["feature", "gain", "weight", "cover"]
    assert list(df["feature"]) == ["a", "b"]
    assert df.loc[0, "cover"] == 3.0  # fN fallback in one importance type


def test_scale_pos_weight_default_and_override():
    y = np.array([1, 0, 0, 0])
    p = apply_scale_pos_weight_default({}, y)
    assert p["scale_pos_weight"] == pytest.approx(3.0)
    p2 = apply_scale_pos_weight_default({"scale_pos_weight": 7.0}, y)
    assert p2["scale_pos_weight"] == 7.0
    # all-negative labels: no weight set (avoid division by zero)
    assert "scale_pos_weight" not in apply_scale_pos_weight_default(
        {}, np.zeros(5))


def test_real_booster_roundtrip():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 3).astype(np.float32)
    y = (X[:, 0] > 0.3).astype(int)
    names = ["x0", "x1", "x2"]
    dtrain = xgb.DMatrix(X, label=y, feature_names=names)
    booster = xgb.train({"objective": "binary:logistic", "verbosity": 0,
                         "max_depth": 2}, dtrain, num_boost_round=10)
    v = gain_vector(booster, names)
    assert v.shape == (3,)
    assert v[0] > v[1] and v[0] > v[2]  # the signal feature dominates
