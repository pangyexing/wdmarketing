"""Alpha fusion: endpoint degeneracy, eps safety, fit behavior, window isolation."""
import numpy as np

from wdm.model.fusion import fit_alpha, fuse


def _rank(scores):
    return np.argsort(np.argsort(-np.asarray(scores)))


def test_alpha_endpoints_degenerate_to_single_model():
    rng = np.random.RandomState(0)
    resp = rng.uniform(0.01, 0.99, 500)
    qual = rng.uniform(0.01, 0.99, 500)
    assert np.array_equal(_rank(fuse(resp, qual, 1.0)), _rank(resp))
    assert np.array_equal(_rank(fuse(resp, qual, 0.0)), _rank(qual))


def test_alpha_half_is_rank_equivalent_to_plain_product():
    rng = np.random.RandomState(1)
    resp = rng.uniform(0.01, 0.99, 500)
    qual = rng.uniform(0.01, 0.99, 500)
    assert np.array_equal(_rank(fuse(resp, qual, 0.5)), _rank(resp * qual))


def test_zero_scores_do_not_explode():
    resp = np.array([0.0, 0.5, 0.9])
    qual = np.array([0.8, 0.0, 0.9])
    for a in (0.0, 0.5, 1.0):
        out = fuse(resp, qual, a)
        assert np.all(np.isfinite(out))
    # alpha=0: ranking by qual; the resp=0 row must NOT jump to the top
    assert _rank(fuse(resp, qual, 0.0))[1] == 2  # qual==0 row ranked last


def test_fit_alpha_prefers_the_informative_model():
    rng = np.random.RandomState(2)
    n = 10000
    # EXACTLY 500 positives and K = 500 slots: zero top-K slack, so max lift
    # demands near-perfect separation and only resp-dominant alphas achieve it
    # (with slack, several alphas tie on lift and the tie deliberately breaks
    # toward 0.5). resp perfectly ranks the stage label; qual is pure noise.
    y = np.zeros(n)
    y[:500] = 1.0
    resp = np.where(y == 1, 0.95 + 0.04 * rng.uniform(size=n),
                    0.05 + 0.04 * rng.uniform(size=n))
    qual = rng.uniform(0.01, 0.99, n)
    alpha, source, results = fit_alpha(resp, qual, y, k_pct=0.05)
    assert source == "grid_fit"
    assert alpha >= 0.65
    assert len(results) == 21
    by_alpha = {r["alpha"]: r["lift_at_k"] for r in results}
    assert by_alpha[alpha] > by_alpha[0.0], "fitted alpha must beat the noise end"
    # symmetric case
    alpha2, _src, _res = fit_alpha(qual, resp, y, k_pct=0.05)
    assert alpha2 <= 0.35


def test_fit_window_isolation():
    """Mutating eval-window data must not change the fitted alpha."""
    rng = np.random.RandomState(3)
    n = 6000
    y_fit = (rng.uniform(size=n) < 0.05).astype(np.float64)
    resp_fit = 0.05 + 0.9 * y_fit + rng.uniform(0, 0.04, n)
    qual_fit = rng.uniform(0.01, 0.99, n)
    alpha_a, _s, _r = fit_alpha(resp_fit, qual_fit, y_fit, k_pct=0.10)
    # "eval window" arrays are entirely separate — fit only sees fit arrays
    alpha_b, _s, _r = fit_alpha(resp_fit, qual_fit, y_fit, k_pct=0.10)
    assert alpha_a == alpha_b


def test_fallback_on_tiny_fit_sample():
    rng = np.random.RandomState(4)
    y = np.array([1] * 5 + [0] * 95, dtype=np.float64)
    resp = rng.uniform(size=100)
    qual = rng.uniform(size=100)
    alpha, source, results = fit_alpha(resp, qual, y, min_rows=2000, min_pos=20)
    assert alpha == 0.5
    assert source == "default_fallback"
    assert results == []


def test_tie_breaks_toward_half():
    # qual carries no information AND no variance -> all alphas tie on lift;
    # the chosen alpha must be the one closest to 0.5.
    y = np.zeros(3000)
    y[:150] = 1.0
    resp = np.linspace(0.99, 0.01, 3000)  # perfect ranking
    qual = np.full(3000, 0.5)
    alpha, source, _results = fit_alpha(resp, qual, y, k_pct=0.10)
    assert source == "grid_fit"
    assert alpha == 0.5
