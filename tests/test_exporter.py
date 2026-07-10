"""End-to-end coverage for wdm.model.exporter.export_bundle — previously the
entire deploy path (bundle files, run_manifest.json, calibration holdout fit,
validation_samples 1e-6 contract) had no test. Builds a tiny dataset, trains
a small booster, exports, then validates the bundle with the very
predict_template.Predictor that ships in it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import predict_template  # noqa: E402

from wdm.model.dataset import build_dataset  # noqa: E402
from wdm.model.exporter import export_bundle  # noqa: E402
from wdm.model.trainer import train_final  # noqa: E402

N_ROWS = 400
FEATS = ["f1", "f2", "f3"]


def _make_repo(tmp_path, seed=7):
    rng = np.random.RandomState(seed)
    n = N_ROWS
    f1 = rng.rand(n) * 10
    f2 = rng.randn(n)
    f3 = np.where(rng.rand(n) < 0.2, np.nan, rng.rand(n))
    logit = 0.8 * f1 - 5.0 + 1.2 * f2 + rng.randn(n)
    y = (logit > 0).astype(int)
    dt = pd.date_range("2024-01-01", periods=n, freq="H").strftime("%Y%m%d").astype(int)
    df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "y": y, "dt": dt})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    sf_dir = tmp_path / "selected_features"
    sf_dir.mkdir()
    (sf_dir / "v1.txt").write_text("\n".join(FEATS) + "\n", encoding="utf-8")

    # export_bundle copies <repo_root>/scripts/predict_template.py into the
    # bundle — mirror the real template into the fake repo.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    real_template = ROOT / "scripts" / "predict_template.py"
    (scripts_dir / "predict_template.py").write_text(
        real_template.read_text(encoding="utf-8"), encoding="utf-8")
    return csv, sf_dir


def _cfg(tmp_path, csv, sf_dir):
    return {
        "_repo_root": str(tmp_path),
        "name": "test_export",
        "data": {
            "train_path": str(csv.relative_to(tmp_path)),
            "label_column": "y",
            "time_column": "dt",
        },
        "training": {
            "split": {"strategy": "time", "ratios": [0.6, 0.2, 0.2]},
            "random_seed": 42,
            "top_k_pct": 0.10,
            "final_feature_count": 10,
            "calibration_split_fraction": 0.5,
            "tuner_objective": "aucpr",
            "cv_strategy": "stratified",
            "xgb_base_params": {"objective": "binary:logistic",
                                "tree_method": "hist", "verbosity": 0},
            "eval_metrics": ["aucpr", "auc"],
            "early_stop_metric": "aucpr",
        },
        "missing": {
            "global": {
                "sentinels": [],
                "treat_negative_as_missing": False,
                "treat_empty_as_missing": True,
                "fill_strategy": "keep_nan",
                "generate_missing_indicator": False,
            },
            "per_column": {},
        },
        "selected_features": {
            "active_version": "v1",
            "versions_dir": str(sf_dir.relative_to(tmp_path)),
        },
        "feature_groups": {"family_policy": {}, "semantic_groups": []},
        "analysis": {"corr_cutoff": 0.95},
        "export": {
            "validation_sample_count": 30,
            "model_format": ["json"],
            "calibration": {"enabled": True, "method": "isotonic",
                            "min_valid_rows": 10, "min_valid_pos": 3},
        },
    }


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("export_repo")
    csv, sf_dir = _make_repo(tmp_path)
    cfg = _cfg(tmp_path, csv, sf_dir)
    data = build_dataset(cfg, version="v1")
    best_params = {"n_estimators": 30, "max_depth": 2, "eta": 0.3}
    booster, evals_result = train_final(
        best_params, data.X_train, data.y_train,
        data.X_valid, data.y_valid, cfg)
    run_dir = export_bundle(
        cfg, data, booster, evals_result, best_params,
        best_params_loss=-0.5, selected_features_version="v1",
        run_id="testrun")
    return {"cfg": cfg, "data": data, "booster": booster,
            "run_dir": Path(run_dir)}


def test_bundle_files_written(bundle):
    run_dir = bundle["run_dir"]
    for name in ("booster.json", "feature_list.txt", "missing_spec.json",
                 "predict.py", "validation_samples.csv", "run_manifest.json",
                 "calibration.json"):
        assert (run_dir / name).is_file(), "missing bundle file: " + name


def test_manifest_provenance_fields(bundle):
    mf = json.loads((bundle["run_dir"] / "run_manifest.json")
                    .read_text(encoding="utf-8"))
    import numpy
    import pandas
    assert mf["numpy_version"] == numpy.__version__
    assert mf["pandas_version"] == pandas.__version__
    assert mf["xgb_version"] == xgb.__version__
    # In this git repo the commit must resolve (suffix -dirty allowed).
    assert mf["git_commit"] is None or len(mf["git_commit"]) >= 40
    assert mf["selected_features_version"] == "v1"
    assert mf["n_features_base"] == len(FEATS)
    # No funnel configured -> no exploratory ranker ran.
    assert mf["feature_funnel"]["stage2_candidate_count"] is None
    assert mf["feature_funnel"]["ranking_method_used"] is None


def test_calibration_fit_on_dedicated_holdout(bundle):
    data = bundle["data"]
    assert data.X_calib is not None and len(data.y_calib) > 0
    table = json.loads((bundle["run_dir"] / "calibration.json")
                       .read_text(encoding="utf-8"))
    assert table["fit_split"] == "valid_calib"
    assert table["n_fit"] == int(len(data.y_calib))
    mf = json.loads((bundle["run_dir"] / "run_manifest.json")
                    .read_text(encoding="utf-8"))
    assert mf["calibration"]["fit_split"] == "valid_calib"
    # The holdout is the time-later tail of valid.
    assert np.nanmin(data.dt_calib) >= np.nanmax(data.dt_valid)


def test_valid_carve_bookkeeping(bundle):
    data = bundle["data"]
    assert int(data.valid_mask.sum()) == len(data.y_valid)
    assert int(data.calib_mask.sum()) == len(data.y_calib)
    assert not np.any(data.valid_mask & data.calib_mask)


def test_deployed_predictor_reproduces_scores_1e6(bundle):
    """The shipped predict.py contract: raw validation rows re-scored through
    the deploy-side pipeline match y_pred_expected to 1e-6 (and the
    calibrated column when present)."""
    run_dir = bundle["run_dir"]
    pr = predict_template.Predictor(run_dir)
    df = pd.read_csv(run_dir / "validation_samples.csv")
    feature_cols = [c for c in df.columns
                    if c not in predict_template._NON_FEATURE_COLS]
    scores = pr.predict_proba(df[feature_cols])
    assert float(np.abs(scores - df["y_pred_expected"].values).max()) <= 1e-6
    assert pr.has_calibration
    assert "y_pred_calibrated_expected" in df.columns
    cal = pr.calibrate(scores)
    assert float(np.abs(cal - df["y_pred_calibrated_expected"].values).max()) <= 1e-6
