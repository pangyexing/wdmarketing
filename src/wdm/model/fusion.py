"""Alpha-weighted fusion of two model scores for funnel ranking.

fused = resp^alpha * qual^(1-alpha), both inputs CALIBRATED probabilities.
alpha = 0.5 is rank-equivalent to the plain product (geometric mean); the grid
fit lets the data say which stage's score deserves more weight for the
end-to-end credit objective. Fit on a window that precedes the OOT evaluation
window so reported lifts stay honest.
"""
import logging

import numpy as np

from wdm.metrics.ranking import lift_at_k, precision_at_k

logger = logging.getLogger(__name__)

DEFAULT_GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00 .. 1.00


def fuse(resp_scores, qual_scores, alpha, eps=1e-12):
    """resp^alpha * qual^(1-alpha) with scores clipped to eps first —
    np.power(0.0, 0.0) == 1.0 would jump a zero score to the top of the
    ranking at the alpha endpoints."""
    a = float(alpha)
    r = np.clip(np.asarray(resp_scores, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(qual_scores, dtype=np.float64), eps, None)
    return np.power(r, a) * np.power(q, 1.0 - a)


def fit_alpha(resp_scores, qual_scores, y_stage, k_pct=0.10, grid=None,
              min_rows=2000, min_pos=20):
    """Grid-search alpha maximizing lift@K of the stage label on the FIT window.

    Returns (best_alpha, alpha_source, results) where results is a list of
    {"alpha", "lift_at_k", "precision_at_k"} for auditability. Falls back to
    alpha=0.5 (plain-product ranking) when the fit sample is too small; ties
    break toward 0.5.
    """
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    y = np.asarray(y_stage, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if y.size < int(min_rows) or n_pos < int(min_pos):
        logger.warning("alpha fit skipped: n=%d pos=%d below thresholds "
                       "(min_rows=%d, min_pos=%d) -> alpha=0.5",
                       y.size, n_pos, min_rows, min_pos)
        return 0.5, "default_fallback", []

    results = []
    for a in grid:
        fused = fuse(resp_scores, qual_scores, a)
        results.append({
            "alpha": float(a),
            "lift_at_k": float(lift_at_k(y, fused, float(k_pct))),
            "precision_at_k": float(precision_at_k(y, fused, float(k_pct))),
        })
    best_lift = max(r["lift_at_k"] for r in results
                    if np.isfinite(r["lift_at_k"]))
    candidates = [r["alpha"] for r in results
                  if np.isfinite(r["lift_at_k"]) and r["lift_at_k"] >= best_lift - 1e-12]
    best_alpha = min(candidates, key=lambda a: abs(a - 0.5))
    logger.info("alpha fit: best alpha=%.2f (lift@%.0f%%=%.4f, %d-way tie -> closest to 0.5)",
                best_alpha, float(k_pct) * 100, best_lift, len(candidates))
    return float(best_alpha), "grid_fit", results
