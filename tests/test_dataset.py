"""Stage-2 dataset builder tests — all-NA row filtering.

These tests exercise `build_dataset` end-to-end with a tiny synthetic CSV,
avoiding `load_config` to isolate the row-filter behaviour from config loading.
Time-based split is used so bad rows land in predictable splits.

Policy covered:
  * train / valid : rows where every selected feature is NaN are dropped.
  * oot           : rows are kept so evaluation reflects production; their
                    count + mask are exposed via StageTwoData for the
                    evaluator's `oot_excl_all_na` companion row.
"""
import numpy as np
import pandas as pd
import pytest

from wdm.model.dataset import build_dataset


def _write_csv(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _write_selected_features(tmp_path, feats):
    d = tmp_path / "selected_features"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "v1.txt"
    p.write_text("\n".join(feats) + "\n", encoding="utf-8")
    return d


def _make_cfg(tmp_path, versions_dir, csv_path,
              fill_strategy="keep_nan",
              generate_missing_indicator=False,
              per_column=None,
              ratios=(0.6, 0.2, 0.2)):
    return {
        "_repo_root": str(tmp_path),
        "name": "test",
        "data": {
            "train_path": str(csv_path.relative_to(tmp_path)),
            "label_column": "y",
            "time_column": "yyyymmdd",
        },
        "training": {
            "split": {"strategy": "time", "ratios": list(ratios)},
            "random_seed": 42,
        },
        "missing": {
            "global": {
                "sentinels": [],
                "treat_negative_as_missing": False,
                "treat_empty_as_missing": True,
                "fill_strategy": fill_strategy,
                "fill_constant": -999.0,
                "generate_missing_indicator": generate_missing_indicator,
                "indicator_threshold": 0.10,
            },
            "per_column": per_column or {},
        },
        "selected_features": {
            "active_version": "v1",
            "versions_dir": str(versions_dir.relative_to(tmp_path)),
        },
    }


def _yyyymmdd(n):
    return pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y%m%d").astype(int)


# With n=20 and ratios (0.6, 0.2, 0.2): n_tr=12 → rows 0-11 train,
# 12-15 valid, 16-19 oot. Used to pick "bad rows" that land in the desired
# split below.
def test_drops_train_valid_all_na_rows_keeps_oot(tmp_path):
    n = 20
    df = pd.DataFrame({
        "f1": np.linspace(1, 20, n),
        "f2": np.linspace(10, 30, n),
        "f3": np.linspace(0.1, 2.0, n),
        "y": np.array([0, 1] * 10),
        "yyyymmdd": _yyyymmdd(n),
    })
    # rows 5 & 11 → train (dropped), row 14 → valid (dropped), row 17 → oot (kept).
    for r in [5, 11, 14, 17]:
        df.loc[r, ["f1", "f2", "f3"]] = np.nan
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2", "f3"])
    cfg = _make_cfg(tmp_path, versions, csv)

    data = build_dataset(cfg)

    # train: 12 - 2 = 10, valid: 4 - 1 = 3, oot: 4 (oot row kept).
    assert len(data.X_train) == 10
    assert len(data.X_valid) == 3
    assert len(data.X_oot) == 4

    # OOT still contains the all-NA row (retained).
    assert np.all(np.isnan(data.X_oot), axis=1).sum() == 1

    # Counts and rates populated for all three splits.
    assert data.all_na_counts == {"train": 2, "valid": 1, "oot": 1}
    assert data.all_na_rates["train"] == pytest.approx(2 / 12)
    assert data.all_na_rates["oot"] == pytest.approx(1 / 4)

    # oot_all_na_mask aligned with X_oot/y_oot, and marks exactly the kept all-NA row.
    assert data.oot_all_na_mask.shape == (len(data.y_oot),)
    assert int(data.oot_all_na_mask.sum()) == 1


def test_split_alignment_after_drop(tmp_path):
    n = 30
    df = pd.DataFrame({
        "f1": np.arange(n, dtype=float),
        "f2": np.arange(n, dtype=float) * 2.0,
        "y": np.random.RandomState(0).randint(0, 2, n),
        "yyyymmdd": _yyyymmdd(n),
    })
    for r in [2, 15, 25]:
        df.loc[r, ["f1", "f2"]] = np.nan
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2"])
    cfg = _make_cfg(tmp_path, versions, csv)

    data = build_dataset(cfg)

    assert len(data.X_train) == len(data.y_train) == int(data.train_mask.sum())
    assert len(data.X_valid) == len(data.y_valid) == int(data.valid_mask.sum())
    assert len(data.X_oot) == len(data.y_oot) == int(data.oot_mask.sum())
    assert not (data.train_mask & data.valid_mask).any()
    assert not (data.train_mask & data.oot_mask).any()
    assert not (data.valid_mask & data.oot_mask).any()
    # Only train + valid drops change the sum; OOT all-NA rows are retained.
    expected_dropped = data.all_na_counts["train"] + data.all_na_counts["valid"]
    assert int(data.train_mask.sum() + data.valid_mask.sum() + data.oot_mask.sum()) == n - expected_dropped


def test_sentinels_and_negatives_trigger_drop(tmp_path):
    # n=15, ratios (0.6, 0.2, 0.2) → train rows 0-8, valid 9-11, oot 12-14.
    n = 15
    df = pd.DataFrame({
        "f1": np.linspace(1, 15, n),
        "f2": np.linspace(5, 20, n),
        "y": np.array([0, 1, 0] * 5),
        "yyyymmdd": _yyyymmdd(n),
    })
    df.loc[7, ["f1", "f2"]] = -1.0       # train row, should drop
    df.loc[10, ["f1", "f2"]] = [-5.0, -3.0]  # valid row, should drop
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2"])
    cfg = _make_cfg(
        tmp_path, versions, csv,
        per_column={
            "f1": {"sentinels": [-1], "treat_negative_as_missing": True},
            "f2": {"sentinels": [-1], "treat_negative_as_missing": True},
        },
    )

    data = build_dataset(cfg)

    assert data.all_na_counts["train"] == 1
    assert data.all_na_counts["valid"] == 1
    assert data.all_na_counts["oot"] == 0


def test_indicator_columns_do_not_defeat_detection(tmp_path):
    n = 20
    f1 = np.linspace(1, 20, n)
    for r in [3, 9, 14, 18]:
        f1[r] = np.nan
    df = pd.DataFrame({
        "f1": f1,
        "f2": np.linspace(1, 10, n),
        "y": np.array([0, 1] * 10),
        "yyyymmdd": _yyyymmdd(n),
    })
    # Rows 3 (train) and 14 (valid) have BOTH features NaN → real all-NA rows.
    for r in [3, 14]:
        df.loc[r, "f2"] = np.nan
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2"])
    cfg = _make_cfg(tmp_path, versions, csv, generate_missing_indicator=True)

    data = build_dataset(cfg)

    assert data.all_na_counts["train"] + data.all_na_counts["valid"] == 2
    assert "f1__isnan" in data.feature_list


def test_empty_train_raises(tmp_path):
    # Make every training row all-NA by NaN-ing the training rows only.
    n = 15
    df = pd.DataFrame({
        "f1": np.linspace(1, 15, n),
        "f2": np.linspace(1, 15, n),
        "y": np.array([0, 1, 0] * 5),
        "yyyymmdd": _yyyymmdd(n),
    })
    # With ratios (0.6, 0.2, 0.2): train = rows 0-8.
    for r in range(9):
        df.loc[r, ["f1", "f2"]] = np.nan
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2"])
    cfg = _make_cfg(tmp_path, versions, csv)

    with pytest.raises(ValueError, match="train.*0 samples"):
        build_dataset(cfg)


def test_all_na_oot_rows_do_not_raise(tmp_path):
    # OOT being entirely all-NA is now a reported condition, not a fatal error
    # — it represents a feature-coverage regression that production will see.
    n = 15
    df = pd.DataFrame({
        "f1": np.linspace(1, 15, n),
        "f2": np.linspace(1, 15, n),
        "y": np.array([0, 1, 0] * 5),
        "yyyymmdd": _yyyymmdd(n),
    })
    # With ratios (0.6, 0.2, 0.2) on n=15: oot = rows 12-14.
    for r in [12, 13, 14]:
        df.loc[r, ["f1", "f2"]] = np.nan
    csv = _write_csv(tmp_path, df)
    versions = _write_selected_features(tmp_path, ["f1", "f2"])
    cfg = _make_cfg(tmp_path, versions, csv)

    data = build_dataset(cfg)  # must not raise
    assert data.all_na_counts["oot"] == 3
    assert int(data.oot_all_na_mask.sum()) == 3
    assert len(data.X_oot) == 3   # retained, not dropped
