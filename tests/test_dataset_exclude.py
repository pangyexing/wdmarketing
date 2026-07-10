"""data.exclude_rows: masks stay aligned to the raw CSV after row exclusion."""
import numpy as np
import pandas as pd
import pytest

from wdm.model.dataset import build_dataset


def _make_env(tmp_path, exclude_rows=None, sample_weight=None):
    """Minimal repo layout + config for build_dataset."""
    rng = np.random.RandomState(0)
    n = 600
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "dt": np.repeat(np.arange(20250101, 20250131), n // 30),
        "credit_1v1": rng.choice([-1, 0, 1, 2, 3], size=n, p=[0.1, 0.5, 0.2, 0.1, 0.1]),
    })
    df["label"] = (df["f1"] + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    (tmp_path / "data").mkdir()
    df.to_csv(tmp_path / "data" / "table.csv", index=False)
    feats_dir = tmp_path / "artifacts" / "prod" / "selected_features"
    feats_dir.mkdir(parents=True)
    (feats_dir / "v1.txt").write_text("f1\nf2\n")

    cfg = {
        "_repo_root": str(tmp_path),
        "name": "prod",
        "data": {
            "train_path": "data/table.csv",
            "label_column": "label",
            "time_column": "dt",
        },
        "missing": {"global": {
            "sentinels": [], "treat_negative_as_missing": False,
            "treat_empty_as_missing": True, "fill_strategy": "keep_nan",
            "generate_missing_indicator": False,
        }},
        "training": {
            "random_seed": 42,
            "split": {"strategy": "time", "ratios": [0.7, 0.15, 0.15]},
            # required key (defaults live only in configs/global.yaml)
            "calibration_split_fraction": 0.5,
        },
        "selected_features": {"active_version": "v1",
                              "versions_dir": "artifacts/prod/selected_features"},
    }
    if exclude_rows:
        cfg["data"]["exclude_rows"] = exclude_rows
    if sample_weight:
        cfg["training"]["sample_weight"] = sample_weight
    return cfg, df


def test_masks_scatter_back_to_raw_length(tmp_path):
    cfg, raw = _make_env(
        tmp_path, exclude_rows=[{"column": "credit_1v1", "values": [-1]}])
    data = build_dataset(cfg, version="v1")
    n_raw = len(raw)
    n_excluded = int((raw["credit_1v1"] == -1).sum())
    assert n_excluded > 0

    for mask in (data.train_mask, data.valid_mask, data.oot_mask):
        assert mask.shape == (n_raw,), "masks must keep raw CSV length"
    excluded_idx = raw.index[raw["credit_1v1"] == -1].values
    union = data.train_mask | data.valid_mask | data.oot_mask
    assert not union[excluded_idx].any(), "excluded rows must be False in all masks"
    assert int(union.sum()) == n_raw - n_excluded

    assert data.X_train.shape[0] == int(data.train_mask.sum())
    assert data.X_valid.shape[0] == int(data.valid_mask.sum())
    assert data.X_oot.shape[0] == int(data.oot_mask.sum())


def test_no_exclusion_keeps_legacy_behavior(tmp_path):
    cfg, raw = _make_env(tmp_path)
    data = build_dataset(cfg, version="v1")
    union = data.train_mask | data.valid_mask | data.oot_mask
    assert int(union.sum()) == len(raw)
    # dt threading: per-split dt present and time-ordered between splits
    assert data.dt_train is not None
    assert np.nanmax(data.dt_train) <= np.nanmin(data.dt_valid)
    assert np.nanmax(data.dt_valid) <= np.nanmin(data.dt_oot)
    # no weights configured -> None
    assert data.w_train is None


def test_weights_threaded_per_split(tmp_path):
    sw = {"column": "credit_1v1",
          "mapping": {3: 6.5833, 2: 2.4167, 1: 1.0}, "default": 1.0}
    cfg, raw = _make_env(tmp_path, sample_weight=sw)
    data = build_dataset(cfg, version="v1")
    assert data.w_train is not None
    assert data.w_train.shape[0] == data.X_train.shape[0]
    # spot-check: weights in train match the raw column through the mask
    tiers = pd.to_numeric(raw.loc[data.train_mask, "credit_1v1"]).values
    expected = np.where(tiers == 3, 6.5833,
                        np.where(tiers == 2, 2.4167, 1.0))
    assert np.allclose(data.w_train, expected)
