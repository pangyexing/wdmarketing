"""Isotonic score calibration exported as a portable np.interp lookup table.

Why: each model's raw probability is distorted differently (scale_pos_weight
is tuned per model), so multiplying raw scores across models distorts the
fused ranking. Isotonic calibration is monotone — it never changes a single
model's ranking — but puts the scores of different models on a common
probability scale so products / weighted fusions are meaningful.

The fitted curve is serialized as {x: thresholds, y: values} and replayed at
serving time with np.interp (constant extrapolation at the ends == sklearn's
out_of_bounds="clip"), so deployed predict.py needs no sklearn.
"""
import datetime
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILENAME = "calibration.json"


def fit_isotonic_table(y_true, scores, min_rows=200, min_pos=10):
    """Fit isotonic regression of y_true on scores; return a JSON-able table
    dict, or None (with a warning) when the sample is too small or degenerate.
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if y.size < int(min_rows) or n_pos < int(min_pos) or n_pos == y.size:
        logger.warning("calibration skipped: n=%d pos=%d below thresholds "
                       "(min_rows=%d, min_pos=%d) or single-class",
                       y.size, n_pos, min_rows, min_pos)
        return None
    if float(np.nanmax(s)) <= float(np.nanmin(s)):
        logger.warning("calibration skipped: constant scores")
        return None

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, y)
    try:
        xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
        ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
    except AttributeError:  # very old sklearn: rebuild the curve by prediction
        xs = np.unique(s)
        ys = iso.predict(xs)
    # np.interp needs strictly increasing x; collapse duplicates keeping the
    # last (largest) fitted value so monotonicity is preserved (xs is sorted).
    xs_u = np.unique(xs)
    if xs_u.size != xs.size:
        last_idx = np.searchsorted(xs, xs_u, side="right") - 1
        ys = ys[last_idx]
    xs = xs_u

    table = {
        "version": 1,
        "method": "isotonic",
        "fit_split": "valid",
        "n_fit": int(y.size),
        "n_pos": n_pos,
        "base_rate": float(np.mean(y)),
        "score_range": [float(np.min(s)), float(np.max(s))],
        "x": [float(v) for v in xs],
        "y": [float(v) for v in ys],
        "out_of_bounds": "clip",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    logger.info("calibration fit: n=%d pos=%d thresholds=%d score_range=[%.4f, %.4f]",
                y.size, n_pos, xs.size, table["score_range"][0], table["score_range"][1])
    return table


def apply_table(scores, table):
    """Replay the calibration curve — identical formula to deployed predict.py."""
    s = np.asarray(scores, dtype=np.float64)
    return np.interp(s, np.asarray(table["x"], dtype=np.float64),
                     np.asarray(table["y"], dtype=np.float64))


def save_table(path, table):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)


def load_table(path):
    p = Path(path)
    if not p.is_file():
        return None
    with open(str(p), "r", encoding="utf-8") as f:
        return json.load(f)
