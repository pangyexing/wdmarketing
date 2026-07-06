"""wdm.metrics.pr_auc — PR-AUC / ROC-AUC wrapper contracts."""
import numpy as np

from wdm.metrics.pr_auc import pr_auc, roc_auc


def test_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pr_auc(y, s) == 1.0
    assert roc_auc(y, s) == 1.0


def test_inverted_ranking():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert roc_auc(y, s) == 0.0
    # AP of a fully inverted ranking is bounded by the base rate ordering,
    # strictly below the perfect score.
    assert pr_auc(y, s) < 1.0


def test_single_class_returns_zero():
    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    assert pr_auc(y, s) == 0.0
    assert roc_auc(y, s) == 0.0
    assert pr_auc(np.ones(10, dtype=int), s) == 0.0


def test_random_scores_pr_auc_near_base_rate():
    rng = np.random.RandomState(0)
    y = (rng.rand(20000) < 0.1).astype(int)
    s = rng.rand(20000)
    ap = pr_auc(y, s)
    assert abs(ap - y.mean()) < 0.02


def test_accepts_lists_and_bool_labels():
    y = [True, False, True, False]
    s = [0.9, 0.1, 0.8, 0.2]
    assert roc_auc(y, s) == 1.0
    assert pr_auc(y, s) == 1.0
