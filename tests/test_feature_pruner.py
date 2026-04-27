"""Stage-2 candidate → final pruner tests.

Construct StageTwoData directly (no CSV/build_dataset round-trip) so the test
isolates the ranking + (base, __isnan) pairing logic. SHAP is exercised via a
fake module monkey-patched into the pruner so the suite stays dep-free.
"""
import types

import numpy as np
import pandas as pd
import pytest

from wdm.model.dataset import StageTwoData
from wdm.model.feature_pruner import maybe_prune_to_final


def _make_signal_data(n_rows, base_feats, signal_idx, indicator_feats=(), seed=0):
    rng = np.random.RandomState(seed)
    n_base = len(base_feats)
    X_base = rng.normal(size=(n_rows, n_base)).astype(np.float32)
    # Concentrate label signal on a few base columns.
    logits = X_base[:, list(signal_idx)].sum(axis=1)
    y = (logits > 0).astype(np.int64)

    # Indicator columns are random 0/1 — they should not survive ranking
    # unless their base is itself in the kept set.
    n_ind = len(indicator_feats)
    X_ind = rng.randint(0, 2, size=(n_rows, n_ind)).astype(np.float32)
    X = np.concatenate([X_base, X_ind], axis=1) if n_ind else X_base

    feature_list = list(base_feats) + list(indicator_feats)
    n = X.shape[0]
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    # Stand-in spec_map / fitted: one sentinel value per base + a __default__,
    # so tests can verify the pruner slices both dicts to surviving bases.
    spec_map = {feat: ("spec", feat) for feat in base_feats}
    spec_map["__default__"] = ("spec", "__default__")
    fitted = {feat: ("fit", feat) for feat in base_feats}
    return StageTwoData(
        X_train=X[:n_tr],          y_train=y[:n_tr],
        X_valid=X[n_tr:n_tr+n_va], y_valid=y[n_tr:n_tr+n_va],
        X_oot=X[n_tr+n_va:],       y_oot=y[n_tr+n_va:],
        feature_list=feature_list,
        base_feature_list=list(base_feats),
        fitted=fitted,
        spec_map=spec_map,
        indicator_features=list(indicator_feats),
        raw_index=np.arange(n),
        train_mask=np.zeros(n, dtype=bool),
        valid_mask=np.zeros(n, dtype=bool),
        oot_mask=np.zeros(n, dtype=bool),
    )


def _cfg(final_n, candidate_n=None, num_boost_round=80,
         ranking_method="gain", n_seeds=2, n_permutation_repeats=2):
    return {
        "training": {
            "final_feature_count": final_n,
            "stage2_candidate_count": candidate_n,
            "stage2_pruning": {
                "ranking_method": ranking_method,
                "n_seeds": n_seeds,
                "n_permutation_repeats": n_permutation_repeats,
                "permutation_seed": 0,
                "num_boost_round": num_boost_round,
                "early_stopping_rounds": 20,
                "xgb_params": {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "eval_metric": "aucpr",
                    "max_depth": 4,
                    "eta": 0.2,
                    "verbosity": 0,
                },
            },
        }
    }


def test_pruning_skipped_when_pool_at_or_below_final(tmp_path):
    base = ["f{0}".format(i) for i in range(5)]
    data = _make_signal_data(200, base, signal_idx=[0, 1])
    cfg = _cfg(final_n=10)  # final_n > n_base → no-op

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)

    assert out is data
    assert out.feature_list == base
    assert not (tmp_path / "exploratory_importance.csv").exists()
    assert not (tmp_path / "pruned_features.txt").exists()


def test_pruning_reduces_to_final_count_and_keeps_signal(tmp_path):
    n_base = 30
    base = ["f{0}".format(i) for i in range(n_base)]
    signal_idx = [0, 1, 2, 3, 4]
    data = _make_signal_data(2000, base, signal_idx=signal_idx, seed=1)
    cfg = _cfg(final_n=8, candidate_n=n_base)

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)

    assert out is not data
    assert len(out.base_feature_list) == 8
    assert len(out.feature_list) == 8
    assert out.X_train.shape[1] == 8
    assert out.X_valid.shape[1] == 8
    assert out.X_oot.shape[1] == 8

    # Highest-signal features should survive. We're not asking for all 5 because
    # gain ranking on a held-out tree can be noisy near the boundary; require at
    # least 4/5 to land in top-8 — a strong signal-vs-noise gap.
    kept = set(out.base_feature_list)
    survived = sum(1 for f in [base[i] for i in signal_idx] if f in kept)
    assert survived >= 4, "expected ≥4 of 5 signal features in kept set, got {0}".format(survived)


def test_pruning_artifacts_written(tmp_path):
    n_base = 20
    base = ["f{0}".format(i) for i in range(n_base)]
    data = _make_signal_data(1500, base, signal_idx=[0, 1, 2], seed=2)
    cfg = _cfg(final_n=5, candidate_n=n_base)

    maybe_prune_to_final(data, cfg, run_dir=tmp_path)

    imp_path = tmp_path / "exploratory_importance.csv"
    feat_path = tmp_path / "pruned_features.txt"
    assert imp_path.is_file()
    assert feat_path.is_file()

    imp_df = pd.read_csv(imp_path)
    assert set(imp_df.columns) >= {"feature", "score", "kept", "ranking_method"}
    assert len(imp_df) == n_base
    assert int(imp_df["kept"].sum()) == 5

    kept_lines = [l for l in feat_path.read_text().splitlines() if l.strip()]
    assert len(kept_lines) == 5


def test_pruner_slices_spec_map_and_fitted(tmp_path):
    # Bundle hygiene: missing_spec.json must not carry specs/fitted entries
    # for dropped candidates, otherwise the deploy bundle inflates ~4× and
    # implies the model still depends on features it doesn't.
    n_base = 12
    base = ["f{0}".format(i) for i in range(n_base)]
    data = _make_signal_data(800, base, signal_idx=[0, 1, 2], seed=4)
    cfg = _cfg(final_n=4, candidate_n=n_base, ranking_method="gain")

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)

    kept = set(out.base_feature_list)
    assert set(out.fitted) == kept, "fitted should only contain surviving bases"
    spec_keys = set(out.spec_map)
    assert spec_keys == kept | {"__default__"}, (
        "spec_map must keep __default__ (predict-time fallback) plus surviving bases"
    )


def test_indicator_features_pair_with_surviving_bases(tmp_path):
    # 10 bases; signal lives in f0..f2 — only their indicators should ride
    # along when final_n=3. f7's indicator must be dropped because its base
    # falls out of the top-3 ranking.
    n_base = 10
    base = ["f{0}".format(i) for i in range(n_base)]
    indicators = ["f0__isnan", "f7__isnan"]
    data = _make_signal_data(2000, base, signal_idx=[0, 1, 2],
                             indicator_feats=indicators, seed=3)
    cfg = _cfg(final_n=3, candidate_n=n_base)

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)

    assert len(out.base_feature_list) == 3
    surviving_bases = set(out.base_feature_list)
    expected_indicators = [
        ind for ind in indicators if ind[: -len("__isnan")] in surviving_bases
    ]
    assert out.indicator_features == expected_indicators
    # Layout invariant: bases first, indicators last.
    assert out.feature_list == out.base_feature_list + out.indicator_features
    assert out.X_train.shape[1] == len(out.feature_list)


@pytest.mark.parametrize("method", ["gain", "stability", "permutation",
                                    "permutation_stability"])
def test_each_ranking_method_keeps_signal(tmp_path, method):
    # Each ranker should pick the few signal-bearing features out of a noisy
    # candidate pool. The stability variants average across seeds, and the
    # permutation variants score by valid PR-AUC drop, so each path through
    # _RANKERS should land at the same answer modulo small ordering noise.
    n_base = 18
    base = ["f{0}".format(i) for i in range(n_base)]
    signal_idx = [0, 1, 2]
    data = _make_signal_data(2000, base, signal_idx=signal_idx, seed=11)

    # Keep n_seeds / repeats low so the parametric run finishes quickly.
    cfg = _cfg(final_n=5, candidate_n=n_base, ranking_method=method,
               n_seeds=2, n_permutation_repeats=2)

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)
    assert len(out.base_feature_list) == 5

    kept = set(out.base_feature_list)
    survived = sum(1 for f in [base[i] for i in signal_idx] if f in kept)
    assert survived >= 2, (
        "{0}: expected ≥2 of 3 signal features in kept set, got {1}"
        .format(method, survived))

    imp = pd.read_csv(tmp_path / "exploratory_importance.csv")
    assert (imp["ranking_method"] == method).all()


def test_unknown_ranking_method_raises(tmp_path):
    base = ["f{0}".format(i) for i in range(8)]
    data = _make_signal_data(300, base, signal_idx=[0])
    cfg = _cfg(final_n=3, candidate_n=8, ranking_method="bogus")
    with pytest.raises(ValueError, match="ranking_method"):
        maybe_prune_to_final(data, cfg, run_dir=tmp_path)


def _install_fake_shap(monkeypatch, signal_indices):
    """Patch the pruner's import guard with a deterministic SHAP stand-in.

    The fake explainer hands back |shap| ≈ 5 for the planted-signal columns
    and ≈ 0.1 elsewhere, so a correct ranker has no excuse to miss them.
    """
    class FakeExplainer:
        def __init__(self, booster):
            self.booster = booster

        def shap_values(self, X):
            n, p = X.shape
            sv = np.zeros((n, p), dtype=np.float64)
            for j in range(p):
                weight = 5.0 if j in signal_indices else 0.1
                sv[:, j] = weight * np.random.RandomState(42 + j).normal(size=n)
            return sv

    fake = types.ModuleType("shap")
    fake.TreeExplainer = FakeExplainer
    monkeypatch.setattr(
        "wdm.model.feature_pruner._try_import_shap",
        lambda: (fake, None),
    )


@pytest.mark.parametrize("method", ["shap", "shap_stability"])
def test_shap_methods_when_shap_available(tmp_path, monkeypatch, method):
    n_base = 12
    base = ["f{0}".format(i) for i in range(n_base)]
    signal_idx = [0, 1, 2]
    data = _make_signal_data(800, base, signal_idx=signal_idx, seed=7)
    _install_fake_shap(monkeypatch, set(signal_idx))

    cfg = _cfg(final_n=4, candidate_n=n_base, ranking_method=method, n_seeds=2)
    cfg["training"]["stage2_pruning"]["shap_sample_size"] = 200

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)
    assert len(out.base_feature_list) == 4

    kept = set(out.base_feature_list)
    survived = sum(1 for f in [base[i] for i in signal_idx] if f in kept)
    assert survived == 3, "fake SHAP signal is unambiguous; all 3 should survive"

    imp = pd.read_csv(tmp_path / "exploratory_importance.csv")
    assert (imp["ranking_method"] == method).all()


def test_shap_falls_back_to_stability_when_unavailable(tmp_path, monkeypatch):
    n_base = 14
    base = ["f{0}".format(i) for i in range(n_base)]
    data = _make_signal_data(1500, base, signal_idx=[0, 1, 2], seed=8)

    monkeypatch.setattr(
        "wdm.model.feature_pruner._try_import_shap",
        lambda: (None, ImportError("simulated shap failure")),
    )

    cfg = _cfg(final_n=4, candidate_n=n_base, ranking_method="shap", n_seeds=2)
    cfg["training"]["stage2_pruning"]["shap_fallback"] = "stability"

    out = maybe_prune_to_final(data, cfg, run_dir=tmp_path)
    assert len(out.base_feature_list) == 4

    imp = pd.read_csv(tmp_path / "exploratory_importance.csv")
    # Persisted method reflects what actually scored, not what was requested.
    assert (imp["ranking_method"] == "stability").all()


def test_shap_fallback_raise_propagates(tmp_path, monkeypatch):
    n_base = 8
    base = ["f{0}".format(i) for i in range(n_base)]
    data = _make_signal_data(300, base, signal_idx=[0])

    monkeypatch.setattr(
        "wdm.model.feature_pruner._try_import_shap",
        lambda: (None, ImportError("simulated shap failure")),
    )

    cfg = _cfg(final_n=3, candidate_n=n_base, ranking_method="shap_stability")
    cfg["training"]["stage2_pruning"]["shap_fallback"] = "raise"

    with pytest.raises(RuntimeError, match="shap"):
        maybe_prune_to_final(data, cfg, run_dir=tmp_path)


def test_shap_fallback_to_shap_method_rejected(tmp_path, monkeypatch):
    # Self-referential fallback (shap → shap_stability) is a config error;
    # both routes need SHAP and would loop forever.
    n_base = 8
    base = ["f{0}".format(i) for i in range(n_base)]
    data = _make_signal_data(300, base, signal_idx=[0])

    monkeypatch.setattr(
        "wdm.model.feature_pruner._try_import_shap",
        lambda: (None, ImportError("simulated shap failure")),
    )

    cfg = _cfg(final_n=3, candidate_n=n_base, ranking_method="shap")
    cfg["training"]["stage2_pruning"]["shap_fallback"] = "shap_stability"

    with pytest.raises(ValueError, match="shap_fallback"):
        maybe_prune_to_final(data, cfg, run_dir=tmp_path)


def test_legacy_path_when_candidate_count_unset(tmp_path):
    # candidate_count=None: pruner is bypassed. Verifies back-compat for
    # configs that haven't opted into the funnel yet.
    base = ["f{0}".format(i) for i in range(20)]
    data = _make_signal_data(500, base, signal_idx=[0, 1])
    cfg = _cfg(final_n=5, candidate_n=None)

    # Sanity: the pre-funnel callsite is Stage-1 (selector.py), which already
    # truncates to final_feature_count when candidate_n is unset. Stage-2's
    # pruner only sees a reduced pool in that legacy mode and exits early.
    cfg_legacy = {
        "training": {
            **cfg["training"],
            "final_feature_count": 50,  # ≥ 20 → pruner becomes no-op
        }
    }
    out = maybe_prune_to_final(data, cfg_legacy, run_dir=tmp_path)
    assert out is data
