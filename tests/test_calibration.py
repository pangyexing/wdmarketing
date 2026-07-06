"""Calibration table: fit/save/load roundtrip, np.interp == sklearn, guards."""
import json

import numpy as np
import pytest

from wdm.model.calibration import (
    apply_table, fit_isotonic_table, load_table, save_table,
)


def _synthetic(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    scores = rng.beta(2, 5, size=n)
    # Positive probability increases with score; sigmoid-ish, noisy labels.
    p = 1.0 / (1.0 + np.exp(-8 * (scores - 0.4)))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return y, scores


def test_table_validity_and_monotonicity():
    y, s = _synthetic()
    table = fit_isotonic_table(y, s)
    assert table is not None
    x = np.asarray(table["x"])
    yv = np.asarray(table["y"])
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"
    assert np.all(np.diff(yv) >= 0), "y must be non-decreasing"
    assert yv.min() >= 0.0 and yv.max() <= 1.0
    # Calibration preserves weak ordering of raw scores
    probe = np.sort(np.random.RandomState(1).uniform(0, 1, 500))
    cal = apply_table(probe, table)
    assert np.all(np.diff(cal) >= 0)


def test_interp_matches_sklearn_including_clip():
    from sklearn.isotonic import IsotonicRegression
    y, s = _synthetic(seed=3)
    table = fit_isotonic_table(y, s)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, y)
    # Probe inside AND outside the fitted score range (clip behavior).
    probe = np.concatenate([
        np.linspace(-0.5, 1.5, 401), s[:200],
    ])
    expected = iso.predict(probe)
    actual = apply_table(probe, table)
    assert np.allclose(actual, expected, atol=1e-12)


def test_save_load_roundtrip(tmp_path):
    y, s = _synthetic(seed=5)
    table = fit_isotonic_table(y, s)
    p = tmp_path / "calibration.json"
    save_table(p, table)
    loaded = load_table(p)
    assert loaded["x"] == table["x"]
    assert loaded["y"] == table["y"]
    # File is valid JSON with plain floats
    raw = json.loads(p.read_text())
    assert isinstance(raw["x"][0], float)
    probe = np.linspace(0, 1, 100)
    assert np.allclose(apply_table(probe, loaded), apply_table(probe, table))


def test_load_missing_returns_none(tmp_path):
    assert load_table(tmp_path / "nope.json") is None


@pytest.mark.parametrize("y,s", [
    (np.zeros(1000), np.linspace(0, 1, 1000)),            # no positives
    (np.ones(1000), np.linspace(0, 1, 1000)),             # single class (all pos)
    (np.array([0, 1] * 50), np.linspace(0, 1, 100)),      # n < min_rows
    (np.array([0] * 1995 + [1] * 5), np.linspace(0, 1, 2000)),  # pos < min_pos
])
def test_guards_return_none(y, s):
    assert fit_isotonic_table(y, s, min_rows=200, min_pos=10) is None


def test_constant_scores_return_none():
    y = np.array([0, 1] * 500)
    s = np.full(1000, 0.3)
    assert fit_isotonic_table(y, s) is None
