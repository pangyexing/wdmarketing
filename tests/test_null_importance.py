"""Null-importance screen: signal features must survive, most pure-noise
features must be rejected, and the output list must be Stage-2 consumable.

Needs xgboost — runs only under the ML environment:
    PYTHONPATH=src .../envs/env_ml/bin/python -m pytest tests/test_null_importance.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from wdm.analysis.null_importance import run_null_importance  # noqa: E402

N_SIGNAL = 5
N_NOISE = 40
N_ROWS = 4000


def _make_repo(repo_root):
    rng = np.random.RandomState(11)
    cols = {}
    signal_names = ["sig_{0}".format(i) for i in range(N_SIGNAL)]
    noise_names = ["noise_{0:02d}".format(i) for i in range(N_NOISE)]
    logit = -1.5
    for i, name in enumerate(signal_names):
        x = rng.randn(N_ROWS)
        cols[name] = x
        logit = logit + (0.9 - 0.1 * i) * x
    for name in noise_names:
        cols[name] = rng.randn(N_ROWS)
    p = 1.0 / (1.0 + np.exp(-logit))
    cols["y"] = (rng.rand(N_ROWS) < p).astype(int)
    df = pd.DataFrame(cols)

    data_path = repo_root / "data" / "screen.csv"
    data_path.parent.mkdir(parents=True)
    df.to_csv(data_path, index=False)

    features = signal_names + noise_names
    sf_dir = repo_root / "artifacts" / "screen_test" / "selected_features"
    sf_dir.mkdir(parents=True)
    with open(sf_dir / "v1_auto.txt", "w", encoding="utf-8") as f:
        f.write("# parent: null\n" + "\n".join(features) + "\n")
    return signal_names, noise_names


def _cfg(repo_root):
    return {
        "name": "screen_test",
        "_repo_root": str(repo_root),
        "data": {
            "train_path": "data/screen.csv",
            "label_column": "y",
            "time_column": None,
            "column_mapping": None,
        },
        "missing": {
            "global": {
                "sentinels": [],
                "treat_negative_as_missing": False,
                "treat_empty_as_missing": True,
                "fill_strategy": "keep_nan",
                "generate_missing_indicator": False,
            },
        },
        "analysis": {
            "null_importance": {
                "enabled": True,
                "n_actual_runs": 2,
                "n_null_runs": 10,
                "n_boost_rounds": 60,
                "keep_percentile": 75,
                "importance_type": "gain",
                "max_features": None,
                "out_version": "v2_model",
                "xgb_params": {},
            },
        },
        "training": {
            "random_seed": 42,
            "final_feature_count": 30,
            "split": {"strategy": "stratified", "ratios": [0.7, 0.15, 0.15]},
            "xgb_base_params": {"objective": "binary:logistic",
                                "tree_method": "hist", "verbosity": 0},
        },
        "selected_features": {"active_version": "v1_auto"},
    }


@pytest.fixture(scope="module")
def screen_run(tmp_path_factory):
    repo = tmp_path_factory.mktemp("screen_repo")
    signal_names, noise_names = _make_repo(repo)
    cfg = _cfg(repo)
    result = run_null_importance(cfg)
    return repo, cfg, result, signal_names, noise_names


def test_signal_features_all_kept(screen_run):
    repo, _, result, signal_names, _ = screen_run
    report = pd.read_csv(result["report_csv"])
    kept = set(report.loc[report["keep"] == True, "feature"])
    assert set(signal_names) <= kept


def test_most_noise_rejected(screen_run):
    _, _, result, _, noise_names = screen_run
    report = pd.read_csv(result["report_csv"])
    kept = set(report.loc[report["keep"] == True, "feature"])
    noise_kept = len(kept & set(noise_names))
    assert noise_kept < len(noise_names) * 0.5, (
        "too many noise features survived: {0}/{1}".format(
            noise_kept, len(noise_names)))


def test_output_list_format(screen_run):
    repo, cfg, result, signal_names, _ = screen_run
    txt = Path(result["features_txt"])
    assert txt.name == "v2_model.txt"
    lines = txt.read_text(encoding="utf-8").splitlines()
    assert any(l.startswith("# parent: v1_auto") for l in lines)
    assert any(l.startswith("# source: analysis/null_importance.py") for l in lines)
    feats = [l for l in lines if l and not l.startswith("#")]
    assert 0 < len(feats) <= cfg["training"]["final_feature_count"]
    # Stage-2 loader must accept the file and signal features must rank high.
    from wdm.model.dataset import _load_selected_features
    loaded, _ = _load_selected_features(cfg, version="v2_model")
    assert loaded == feats
    assert set(signal_names) <= set(feats)


def test_meta_json_written(screen_run):
    _, _, result, _, _ = screen_run
    import json
    with open(result["meta_json"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["n_null_runs"] == 10
    assert meta["base_version"] == "v1_auto"
    assert meta["n_kept"] >= N_SIGNAL