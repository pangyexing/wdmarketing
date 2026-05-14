"""ABCD seed populations vs OOT — model score distribution compare.

Loads a trained model bundle from artifacts/<product>/models/<run_id>/ via
the bundle's own self-contained predict.py, scores:

  * OOT slice of data/<product>.csv (same time-split the training run used)
  * Each seed CSV under <seeds_dir>/<name>.csv

then writes score-distribution comparison artifacts answering business
questions like:

  * Do A/B (high-intent populations) actually score higher than C/D (general)?
  * What fraction of each seed lands in the OOT top decile (the operating
    threshold)?
  * Is the score distribution shape consistent (PSI on the score itself)
    between each seed and OOT? A big score-PSI is the strongest signal that
    the model would behave differently on this population than on OOT.

Usage:
    PYTHONPATH=src python3 scripts/compare_seed_scores.py \\
        --product hzz_day --run-id smoke02 --seeds-dir data/seeds

Outputs under artifacts/<product>/score_compare/<run_id>/:
    manifest.json
    scores/<source>.csv             per-row scores (OOT also keeps cd_time)
    score_summary.csv               source × {n, mean, median, %_above_oot_top10}
    score_quantiles.csv             source × quantiles (p1..p99)
    score_psi.csv                   each seed score-PSI vs OOT
    score_deciles.csv               OOT-decile cuts × source × count/pct —
                                    answers "how is each seed redistributed
                                    across the OOT score deciles"
"""
import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.analysis.psi import compute_psi, flag
from wdm.config import load_config
from wdm.utils.binning import equal_freq_edges
from wdm.utils.logging import setup_logging
from wdm.utils.paths import artifacts_root, model_run_dir
from wdm.utils.time_utils import split_by_yyyymmdd

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = ("A", "B", "C", "D")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
N_DECILES = 10


def _load_predictor(bundle_dir):
    """Dynamically import the bundle's predict.py and return Predictor(bundle_dir).

    Each bundle ships its own self-contained predict.py; importing it
    (instead of re-implementing the missing-rule replay here) ensures we
    score with the exact same logic that production deploys use.
    """
    p = Path(bundle_dir) / "predict.py"
    if not p.is_file():
        raise FileNotFoundError("predict.py not found in bundle: {0}".format(p))
    spec = importlib.util.spec_from_file_location("bundle_predict", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Predictor(bundle_dir)


def _score_oot(predictor, baseline_path, cfg):
    """Apply the configured time-split, return (scores, cd_time_values) for OOT."""
    time_col = cfg["data"]["time_column"]
    if not time_col:
        raise ValueError("data.time_column required to slice OOT")
    needed = list(dict.fromkeys(list(predictor.base_features) + [time_col]))
    header = pd.read_csv(baseline_path, nrows=0).columns
    missing = set(predictor.base_features) - set(header)
    if missing:
        raise ValueError(
            "Baseline {0} missing model features: {1}".format(
                baseline_path, sorted(missing)[:5]))
    df = pd.read_csv(baseline_path, usecols=needed)
    _, _, m_oot = split_by_yyyymmdd(df[time_col], cfg["training"]["split"]["ratios"])
    df_oot = df.loc[m_oot, predictor.base_features].copy()
    cd_time_oot = df.loc[m_oot, time_col].values
    logger.info("Scoring OOT: %d rows", len(df_oot))
    scores = predictor.predict_proba(df_oot)
    return np.asarray(scores, dtype=np.float64), cd_time_oot


def _score_seed(predictor, seed_path):
    df = pd.read_csv(seed_path)
    missing = set(predictor.base_features) - set(df.columns)
    if missing:
        raise ValueError(
            "Seed {0} missing model base features ({1} total); first 5: {2}".format(
                seed_path, len(missing), sorted(missing)[:5]))
    logger.info("Scoring seed %s: %d rows", seed_path.name, len(df))
    scores = predictor.predict_proba(df[list(predictor.base_features)])
    return np.asarray(scores, dtype=np.float64)


def _quantile_table(scores_by_source):
    rows = []
    for name, s in scores_by_source.items():
        s = np.asarray(s, dtype=np.float64)
        row = {
            "source": name,
            "n": int(s.size),
            "mean": float(s.mean()) if s.size else float("nan"),
            "std": float(s.std()) if s.size else float("nan"),
            "min": float(s.min()) if s.size else float("nan"),
            "max": float(s.max()) if s.size else float("nan"),
        }
        for q in QUANTILES:
            row["p{0}".format(int(q * 100))] = (
                float(np.quantile(s, q)) if s.size else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(scores_by_source, oot_top10_threshold):
    rows = []
    for name, s in scores_by_source.items():
        s = np.asarray(s, dtype=np.float64)
        rows.append({
            "source": name,
            "n": int(s.size),
            "mean_score": float(s.mean()) if s.size else float("nan"),
            "median_score": float(np.median(s)) if s.size else float("nan"),
            "p90_score": float(np.quantile(s, 0.90)) if s.size else float("nan"),
            "pct_above_oot_top10": (
                float((s >= oot_top10_threshold).mean()) if s.size else float("nan")),
        })
    return pd.DataFrame(rows)


def _psi_on_scores(scores_by_source, oot_name, n_bins):
    oot = np.asarray(scores_by_source[oot_name], dtype=np.float64)
    edges = equal_freq_edges(oot, n_bins=n_bins)
    rows = []
    for name, s in scores_by_source.items():
        if name == oot_name:
            continue
        s = np.asarray(s, dtype=np.float64)
        psi = compute_psi(oot, s, edges=edges, n_bins=n_bins, missing_as_bin=True)
        rows.append({"source": name, "psi_vs_oot": float(psi), "flag": flag(psi)})
    return pd.DataFrame(rows)


def _decile_table(scores_by_source, oot_name):
    """For each source, bucket scores by OOT-derived decile edges.

    The pct column for OOT is approximately 0.10 across all 10 deciles (the
    equal-freq construction is on OOT). Seeds with `pct[10] >> 0.10` are
    enriched in high-score rows (the headline business metric).
    """
    oot = np.asarray(scores_by_source[oot_name], dtype=np.float64)
    edges = np.quantile(oot, np.linspace(0.0, 1.0, N_DECILES + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    inner = edges[1:-1]
    rows = []
    for name, s in scores_by_source.items():
        s = np.asarray(s, dtype=np.float64)
        idx = np.clip(np.digitize(s, inner, right=False), 0, N_DECILES - 1)
        cnt = np.bincount(idx, minlength=N_DECILES)
        pct = cnt / max(1, cnt.sum())
        for d in range(N_DECILES):
            rows.append({
                "source": name,
                "decile": d + 1,
                "decile_lo": float(edges[d]),
                "decile_hi": float(edges[d + 1]),
                "n": int(cnt[d]),
                "pct": float(pct[d]),
            })
    return pd.DataFrame(rows)


def _resolve_bundle_dir(cfg, run_id, bundle_arg, repo_root):
    if bundle_arg:
        p = Path(bundle_arg)
        return p if p.is_absolute() else (repo_root / p)
    if run_id:
        return model_run_dir(cfg, run_id)
    models_root = artifacts_root(cfg) / "models"
    if not models_root.is_dir():
        raise FileNotFoundError("No models dir at {0}".format(models_root))
    candidates = sorted([p for p in models_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No run dirs under {0}".format(models_root))
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser(
        description="Compare ABCD seed score distributions against OOT for "
                    "a trained model bundle.")
    ap.add_argument("--product", required=True)
    ap.add_argument("--run-id", default=None,
                    help="Model run id under artifacts/<product>/models/. "
                    "If omitted (and --bundle also omitted), uses the lexically "
                    "latest run dir.")
    ap.add_argument("--bundle", default=None,
                    help="Explicit bundle directory; overrides --run-id.")
    ap.add_argument("--seeds-dir", default="data/seeds")
    ap.add_argument("--seeds", default=",".join(DEFAULT_SEEDS),
                    help="Comma-separated seed names; expects <seeds_dir>/<name>.csv.")
    ap.add_argument("--baseline-path", default=None,
                    help="Override cfg.data.train_path.")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="PSI bins for score histogram; defaults to "
                    "cfg.analysis.n_bins.")
    ap.add_argument("--out-dir", default=None,
                    help="Override artifacts/<product>/score_compare/<run_id>/.")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)
    repo_root = Path(cfg["_repo_root"])

    bundle_dir = _resolve_bundle_dir(cfg, args.run_id, args.bundle, repo_root)
    if not bundle_dir.is_dir():
        raise FileNotFoundError("Bundle dir not found: {0}".format(bundle_dir))
    logger.info("Using bundle: %s", bundle_dir)

    baseline_path = Path(args.baseline_path) if args.baseline_path else (
        repo_root / cfg["data"]["train_path"])
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    if not baseline_path.is_file():
        raise FileNotFoundError("Baseline CSV not found: {0}".format(baseline_path))

    seeds_dir = Path(args.seeds_dir)
    if not seeds_dir.is_absolute():
        seeds_dir = repo_root / seeds_dir
    seed_names = [s.strip() for s in args.seeds.split(",") if s.strip()]

    predictor = _load_predictor(bundle_dir)
    logger.info("Predictor loaded; base_features=%d indicators=%d",
                len(predictor.base_features), len(predictor.indicator_features))

    oot_scores, oot_cd_time = _score_oot(predictor, baseline_path, cfg)
    seed_scores: Dict[str, np.ndarray] = {}
    for name in seed_names:
        p = seeds_dir / "{0}.csv".format(name)
        if not p.is_file():
            logger.warning("Seed file not found: %s — skipping", p)
            continue
        seed_scores[name] = _score_seed(predictor, p)
    if not seed_scores:
        raise RuntimeError(
            "No seed CSVs found under {0}; expected {1}".format(
                seeds_dir, ["{0}.csv".format(n) for n in seed_names]))

    sources = {"OOT": oot_scores}
    sources.update(seed_scores)

    out_root = Path(args.out_dir) if args.out_dir else (
        artifacts_root(cfg) / "score_compare" / bundle_dir.name)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_dir = out_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cd_time": oot_cd_time, "score": oot_scores}).to_csv(
        scores_dir / "OOT.csv", index=False)
    for name, s in seed_scores.items():
        pd.DataFrame({"score": s}).to_csv(
            scores_dir / "{0}.csv".format(name), index=False)

    n_bins = int(args.n_bins or cfg.get("analysis", {}).get("n_bins", 10))
    oot_top10 = float(np.quantile(oot_scores, 0.90))

    qt = _quantile_table(sources)
    qt.to_csv(out_root / "score_quantiles.csv", index=False)
    summary = _summary(sources, oot_top10)
    summary.to_csv(out_root / "score_summary.csv", index=False)
    psi_tbl = _psi_on_scores(sources, "OOT", n_bins)
    psi_tbl.to_csv(out_root / "score_psi.csv", index=False)
    deciles = _decile_table(sources, "OOT")
    deciles.to_csv(out_root / "score_deciles.csv", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product": args.product,
        "bundle_dir": str(bundle_dir),
        "run_id": bundle_dir.name,
        "baseline_path": str(baseline_path),
        "oot_top10_threshold": oot_top10,
        "n_bins_psi": n_bins,
        "sources": {name: int(np.asarray(s).size) for name, s in sources.items()},
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=" * 60)
    print("Score comparison complete.")
    print("  bundle           :", bundle_dir.name)
    print("  oot_top10_thresh : {0:.4f}".format(oot_top10))
    print("  out dir          :", out_root)
    print()
    print("score_summary:")
    print(summary.to_string(index=False))
    print()
    print("score_psi (vs OOT):")
    print(psi_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
