"""Lock the deploy-time hand copies to their training-side originals.

scripts/predict_template.py deliberately re-implements the missing-value
replay (wdm.preprocess.missing) and the isotonic replay
(wdm.model.calibration) so the deployed bundle has no dependency on the wdm
package. Until now the two copies were kept in sync only by a comment and the
runtime validation self-test; these tests pin the equivalence in CI. No
xgboost required.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import predict_template  # noqa: E402

from wdm.model.calibration import apply_table, fit_isotonic_table, save_table  # noqa: E402
from wdm.preprocess.missing import (  # noqa: E402
    apply_missing_for_training, build_missing_spec, dump_missing_spec,
    fit_missing,
)


def _raw_frame(n=60, seed=0):
    """Raw business-ish frame: sentinels, negatives, empties, NaN, strings."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "amt": np.where(rng.rand(n) < 0.2, 0.0, rng.rand(n) * 100),      # 0 sentinel
        "cnt": np.where(rng.rand(n) < 0.15, -1.0, rng.randint(0, 9, n)), # negatives
        "score": np.where(rng.rand(n) < 0.25, np.nan, rng.randn(n)),     # NaN
        "txt_num": pd.Series(
            np.where(rng.rand(n) < 0.2, "", (rng.rand(n) * 10).round(2).astype(str)),
            dtype=object),                                               # empty strings
    })


def _cfg_missing(fill_strategy, sentinels, treat_negative):
    return {
        "missing": {
            "global": {
                "sentinels": sentinels,
                "treat_negative_as_missing": treat_negative,
                "treat_empty_as_missing": True,
                "fill_strategy": fill_strategy,
                "fill_constant": -999.0,
                "generate_missing_indicator": False,
                "fill_value_sanity_check": False,
            },
            "per_column": {},
        },
    }


@pytest.mark.parametrize("fill_strategy,sentinels,treat_negative", [
    ("constant", [0], True),
    ("keep_nan", [], False),
    ("median", [0], True),
])
def test_missing_rules_parity(tmp_path, fill_strategy, sentinels, treat_negative):
    cfg = _cfg_missing(fill_strategy, sentinels, treat_negative)
    df = _raw_frame()
    spec_map = build_missing_spec(cfg)
    fitted = fit_missing(df, spec_map)

    wdm_out = apply_missing_for_training(df, spec_map, fitted)

    # Round-trip through the JSON the bundle actually ships.
    spec_path = tmp_path / "missing_spec.json"
    dump_missing_spec(spec_path, spec_map, fitted)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    deploy_out = predict_template._apply_missing_rules(
        df, payload["specs"], payload["fitted"])

    assert list(wdm_out.columns) == list(deploy_out.columns)
    for col in wdm_out.columns:
        a = wdm_out[col].values.astype(np.float64)
        b = deploy_out[col].values.astype(np.float64)
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        assert np.array_equal(nan_a, nan_b), (
            "NaN placement differs for {0} ({1})".format(col, fill_strategy))
        assert np.allclose(a[~nan_a], b[~nan_b], atol=1e-9), (
            "values differ for {0} ({1})".format(col, fill_strategy))


def test_missing_spec_schema_version_supported():
    from wdm.preprocess.missing import MISSING_SPEC_SCHEMA_VERSION
    assert MISSING_SPEC_SCHEMA_VERSION <= predict_template._SUPPORTED_SPEC_SCHEMA, (
        "wdm bumped the missing-spec schema beyond what predict_template.py "
        "can read — update the deploy template together with the schema.")


def test_calibration_replay_parity(tmp_path):
    rng = np.random.RandomState(1)
    scores = rng.rand(500)
    y = (rng.rand(500) < scores * 0.6).astype(int)
    table = fit_isotonic_table(y, scores, min_rows=10, min_pos=5,
                               fit_split="valid_calib")
    assert table is not None
    assert table["fit_split"] == "valid_calib"

    # Ship through JSON exactly as the bundle does.
    p = tmp_path / "calibration.json"
    save_table(p, table)
    loaded = json.loads(p.read_text(encoding="utf-8"))

    fresh = rng.rand(200) * 1.4 - 0.2  # includes out-of-range values
    train_side = apply_table(fresh, table)
    # predict_template.Predictor.calibrate is np.interp over the JSON table —
    # replay the identical formula from the serialized artifact.
    deploy_side = np.interp(fresh,
                            np.asarray(loaded["x"], dtype=np.float64),
                            np.asarray(loaded["y"], dtype=np.float64))
    assert np.allclose(train_side, deploy_side, atol=0)
    # Monotone and clipped to [0, 1].
    order = np.argsort(fresh)
    assert np.all(np.diff(deploy_side[order]) >= -1e-12)
    assert deploy_side.min() >= 0.0 and deploy_side.max() <= 1.0
