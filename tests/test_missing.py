"""Missing-value rule and sanity-check tests."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from wdm.preprocess.missing import (
    MISSING_SPEC_SCHEMA_VERSION, MissingSpec, apply_missing_for_training,
    build_missing_spec, dump_missing_spec, fit_missing, load_missing_spec,
    sanity_check_fill_value, to_nan_array,
)


def test_default_rules_treat_zero_negative_empty_as_missing():
    spec = MissingSpec(sentinels=[0], treat_negative_as_missing=True,
                       treat_empty_as_missing=True, fill_strategy="constant",
                       fill_constant=-999.0)
    s = pd.Series([1, 0, -1, 2, 3, None, ""])
    arr, mask = to_nan_array(s, spec)
    # 0 (sentinel), -1 (neg), None and "" → NaN
    expected_mask = np.array([False, True, True, False, False, True, True])
    assert np.array_equal(mask, expected_mask)


def test_per_column_override_disables_negative_and_sentinel():
    spec = MissingSpec(sentinels=[], treat_negative_as_missing=False,
                       treat_empty_as_missing=True)
    s = pd.Series([1, 0, -1, 2])
    arr, mask = to_nan_array(s, spec)
    # Nothing flagged; 0 and -1 are legitimate
    assert not mask.any()
    assert np.array_equal(arr, np.array([1., 0., -1., 2.]))


def test_special_strategy_preserves_sentinel_as_fill():
    spec = MissingSpec(sentinels=[-1], treat_negative_as_missing=False,
                       fill_strategy="special", fill_value=-1.0)
    df = pd.DataFrame({"pdays": [-1, 5, 10, -1, 7]})
    fitted = fit_missing(df, {"pdays": spec, "__default__": spec})
    out = apply_missing_for_training(df, {"pdays": spec, "__default__": spec}, fitted)
    # The -1 values are masked to NaN then re-filled as -1 — round-trip.
    assert (out["pdays"].values == [-1, 5, 10, -1, 7]).all()


def test_sanity_check_rejects_fill_inside_observed_range():
    spec = MissingSpec(fill_strategy="constant", fill_constant=50.0)
    df = pd.DataFrame({"x": np.arange(100) + 1.0})  # range [1, 100]
    with pytest.raises(ValueError):
        fit_missing(df, {"x": spec, "__default__": spec})


def test_sanity_check_ok_for_default_neg999_on_positive_feature():
    spec = MissingSpec(fill_strategy="constant", fill_constant=-999.0)
    df = pd.DataFrame({"age": np.array([20, 30, 50, 70])})
    # Should not raise
    fitted = fit_missing(df, {"age": spec, "__default__": spec})
    assert fitted["age"].fill_value == -999.0


def test_median_strategy_skips_sanity_check():
    # median will naturally land inside [p01, max] — it's meant to; no error.
    spec = MissingSpec(fill_strategy="median", fill_constant=-999.0,
                       sentinels=[], treat_negative_as_missing=True)
    df = pd.DataFrame({"x": [-5, 10, 20, 30, 40]})
    fitted = fit_missing(df, {"x": spec, "__default__": spec})
    # Negatives masked; median of [10,20,30,40] = 25
    assert fitted["x"].fill_value == 25.0


def test_keep_nan_strategy_leaves_nan():
    spec = MissingSpec(fill_strategy="keep_nan", sentinels=[0],
                       treat_negative_as_missing=True)
    df = pd.DataFrame({"x": [0, 1, -1, 5]})
    fitted = fit_missing(df, {"x": spec, "__default__": spec})
    out = apply_missing_for_training(df, {"x": spec, "__default__": spec}, fitted)
    nan_mask = out["x"].isna().values
    assert list(nan_mask) == [True, False, True, False]


def test_dump_missing_spec_stamps_schema_version(tmp_path):
    spec = MissingSpec(fill_strategy="constant", fill_constant=-999.0,
                       sentinels=[], treat_negative_as_missing=True)
    df = pd.DataFrame({"age": np.array([20.0, 30.0, 50.0, 70.0])})
    fitted = fit_missing(df, {"age": spec, "__default__": spec})
    path = tmp_path / "missing_spec.json"
    dump_missing_spec(str(path), {"age": spec, "__default__": spec}, fitted)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MISSING_SPEC_SCHEMA_VERSION

    loaded_specs, loaded_fitted = load_missing_spec(str(path))
    assert "age" in loaded_specs
    assert loaded_fitted["age"]["fill_value"] == -999.0


def test_load_missing_spec_rejects_future_schema(tmp_path):
    path = tmp_path / "missing_spec.json"
    path.write_text(json.dumps({
        "schema_version": MISSING_SPEC_SCHEMA_VERSION + 99,
        "specs": {"__default__": {"fill_strategy": "constant", "fill_constant": -999.0}},
        "fitted": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_missing_spec(str(path))


def test_load_missing_spec_accepts_legacy_unversioned(tmp_path):
    # Bundles written before schema_version existed default to 0 and should
    # still load — we only refuse versions strictly newer than we support.
    path = tmp_path / "missing_spec.json"
    path.write_text(json.dumps({
        "specs": {"__default__": {"fill_strategy": "constant", "fill_constant": -999.0}},
        "fitted": {},
    }), encoding="utf-8")
    spec_map, fitted = load_missing_spec(str(path))
    assert "__default__" in spec_map
    assert fitted == {}
