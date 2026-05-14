"""ABCD seed populations vs hzz_day baseline — dual-baseline distribution compare.

Why dual baseline: ABCD only match the latest single-day partition, which is
later than every cd_time in the modeling CSV. Comparing a single-day seed to
the multi-month baseline conflates time-drift and population-difference. We
therefore report two PSIs per (feature, seed):

  * psi_full : seed vs the entire baseline (time-mixed). Useful as a sanity
    check but contaminated by time drift.
  * psi_oot  : seed vs the last `oot_ratio` of the baseline by cd_time —
    the closest available time-aligned slice. This is the meaningful figure
    for "is this population different from the modeling distribution".

Usage:
    PYTHONPATH=src python3 scripts/compare_seed_distributions.py \\
        --product hzz_day --seeds-dir data/seeds

    # only the v1_auto 200-feature scope
    PYTHONPATH=src python3 scripts/compare_seed_distributions.py \\
        --product hzz_day --scope selected --features-version v1_auto

    # only the full ~970-feature scope
    PYTHONPATH=src python3 scripts/compare_seed_distributions.py \\
        --product hzz_day --scope full

Outputs to artifacts/<product>/seed_compare/:
    manifest.json
    selected_<version>/  (or full/)
        psi_matrix.csv          feature × seed × {psi_full,flag_full,psi_oot,flag_oot}
        stats_matrix.csv        feature × (full,oot,A..D) × {n,missing,mean,...}
        summary_per_seed.csv    per-seed roll-up
        top_shift_<seed>.csv    per-seed top-50 by psi_oot
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdm.analysis.psi import compute_psi, flag
from wdm.config import load_config
from wdm.io.chunked_reader import iter_column_chunks, read_full
from wdm.preprocess.missing import build_missing_spec, get_spec, to_nan_array
from wdm.utils.binning import equal_freq_edges
from wdm.utils.logging import setup_logging
from wdm.utils.paths import (
    artifacts_root,
    inject_cn_column,
    load_column_mapping,
    selected_features_file,
)
from wdm.utils.time_utils import to_yyyymmdd_int

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = ("A", "B", "C", "D")
MISSING_FLAG = "missing"  # for features absent in a seed CSV


# ---------- numeric helpers ----------

def _safe(arr, fn):
    if arr.size == 0:
        return float("nan")
    m = ~np.isnan(arr)
    if not m.any():
        return float("nan")
    return float(fn(arr[m]))


def _nanmean(arr):
    return _safe(arr, np.mean)


def _nanmedian(arr):
    return _safe(arr, np.median)


def _nanq(arr, q):
    return _safe(arr, lambda x: np.quantile(x, q))


# ---------- feature list resolution ----------

def _read_selected_features(cfg, version):
    p = selected_features_file(cfg, version)
    if not p.is_file():
        raise FileNotFoundError(
            "Selected features file not found: {0}".format(p))
    feats = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    if not feats:
        raise ValueError("Empty selected features file: {0}".format(p))
    return feats, p


def _read_full_feature_list(baseline_path, drop_cols):
    header = pd.read_csv(baseline_path, nrows=0)
    return [c for c in header.columns if c not in drop_cols]


# ---------- OOT mask ----------

def _build_oot_mask(yyyymmdd_values, oot_ratio):
    """Return boolean mask over the input rows, selecting the last
    `oot_ratio` fraction by cd_time integer order. Ties at the boundary
    fall into the earlier (non-OOT) split — matches split_by_yyyymmdd
    semantics in wdm.utils.time_utils.
    """
    ser = pd.Series(yyyymmdd_values).reset_index(drop=True)
    ints = to_yyyymmdd_int(ser)  # drops NaN
    ranked = pd.Series(ints, index=ser.dropna().index).reindex(ser.index)
    order = ranked.rank(method="first", na_option="bottom").values
    n = ser.size
    n_oot = max(1, int(round(oot_ratio * n)))
    return (order > (n - n_oot)).astype(bool)


# ---------- core comparison loop ----------

def _empty_seed_row(name):
    return {
        "{0}_psi_full".format(name): float("nan"),
        "{0}_flag_full".format(name): MISSING_FLAG,
        "{0}_psi_oot".format(name): float("nan"),
        "{0}_flag_oot".format(name): MISSING_FLAG,
        "{0}_n".format(name): 0,
        "{0}_missing_rate".format(name): float("nan"),
        "{0}_mean".format(name): float("nan"),
        "{0}_median".format(name): float("nan"),
        "{0}_p10".format(name): float("nan"),
        "{0}_p90".format(name): float("nan"),
    }


def _compare_features(cfg, spec_map, baseline_path, time_col, oot_mask,
                      seeds, features, n_bins):
    """Iterate baseline by column-chunks; for each feature emit one row of
    baseline stats + per-seed PSI/stats. Mirrors compute_psi_table_single_source.
    """
    seed_names = list(seeds.keys())
    rows = []
    for df_chunk, block in iter_column_chunks(
            baseline_path, features, always=[time_col],
            chunk_size=50, desc="baseline"):
        for feat in block:
            spec = get_spec(spec_map, feat)
            arr_full, m_full = to_nan_array(df_chunk[feat], spec, analysis=True)
            arr_oot = arr_full[oot_mask]
            m_oot = m_full[oot_mask]

            edges_full = equal_freq_edges(arr_full, n_bins=n_bins)
            edges_oot = equal_freq_edges(arr_oot, n_bins=n_bins)

            row = {
                "feature": feat,
                "full_n": int(arr_full.size),
                "full_missing_rate": float(m_full.mean()),
                "full_mean": _nanmean(arr_full),
                "full_median": _nanmedian(arr_full),
                "full_p10": _nanq(arr_full, 0.10),
                "full_p90": _nanq(arr_full, 0.90),
                "oot_n": int(arr_oot.size),
                "oot_missing_rate": float(m_oot.mean()),
                "oot_mean": _nanmean(arr_oot),
                "oot_median": _nanmedian(arr_oot),
                "oot_p10": _nanq(arr_oot, 0.10),
                "oot_p90": _nanq(arr_oot, 0.90),
            }
            for name in seed_names:
                sdf = seeds[name]
                if feat not in sdf.columns:
                    row.update(_empty_seed_row(name))
                    continue
                arr_s, m_s = to_nan_array(sdf[feat], spec, analysis=True)
                psi_full = compute_psi(arr_full, arr_s, edges=edges_full,
                                       n_bins=n_bins, missing_as_bin=True)
                psi_oot = compute_psi(arr_oot, arr_s, edges=edges_oot,
                                      n_bins=n_bins, missing_as_bin=True)
                row["{0}_psi_full".format(name)] = float(psi_full)
                row["{0}_flag_full".format(name)] = flag(psi_full)
                row["{0}_psi_oot".format(name)] = float(psi_oot)
                row["{0}_flag_oot".format(name)] = flag(psi_oot)
                row["{0}_n".format(name)] = int(arr_s.size)
                row["{0}_missing_rate".format(name)] = float(m_s.mean())
                row["{0}_mean".format(name)] = _nanmean(arr_s)
                row["{0}_median".format(name)] = _nanmedian(arr_s)
                row["{0}_p10".format(name)] = _nanq(arr_s, 0.10)
                row["{0}_p90".format(name)] = _nanq(arr_s, 0.90)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------- output writers ----------

def _build_psi_matrix(df, seed_names, cn_map):
    cols = ["feature"]
    for name in seed_names:
        cols.extend([
            "{0}_psi_full".format(name),
            "{0}_flag_full".format(name),
            "{0}_psi_oot".format(name),
            "{0}_flag_oot".format(name),
        ])
    out = df[cols].copy()
    if cn_map:
        out = inject_cn_column(out, cn_map)
    psi_oot_cols = ["{0}_psi_oot".format(n) for n in seed_names]
    out["__sort_key"] = out[psi_oot_cols].max(axis=1)
    out = out.sort_values("__sort_key", ascending=False, na_position="last")
    return out.drop(columns=["__sort_key"]).reset_index(drop=True)


def _build_stats_matrix(df, seed_names, cn_map):
    cols = ["feature"]
    for prefix in ["full", "oot"] + list(seed_names):
        cols.extend([
            "{0}_n".format(prefix),
            "{0}_missing_rate".format(prefix),
            "{0}_mean".format(prefix),
            "{0}_median".format(prefix),
            "{0}_p10".format(prefix),
            "{0}_p90".format(prefix),
        ])
    out = df[cols].copy()
    if cn_map:
        out = inject_cn_column(out, cn_map)
    return out


def _build_summary(df, seed_names, missing_per_seed):
    rows = []
    for name in seed_names:
        psi_full = df["{0}_psi_full".format(name)]
        psi_oot = df["{0}_psi_oot".format(name)]
        flag_full = df["{0}_flag_full".format(name)]
        flag_oot = df["{0}_flag_oot".format(name)]
        valid_full = flag_full != MISSING_FLAG
        valid_oot = flag_oot != MISSING_FLAG
        n_rows_col = "{0}_n".format(name)
        n_rows = int(df[n_rows_col].dropna().max()) if df[n_rows_col].notna().any() else 0
        top5 = (df[["feature", "{0}_psi_oot".format(name)]]
                .dropna()
                .sort_values("{0}_psi_oot".format(name), ascending=False)
                .head(5)["feature"].astype(str).tolist())
        rows.append({
            "seed": name,
            "n_rows": n_rows,
            "n_features_compared": int(valid_oot.sum()),
            "n_missing_features": len(missing_per_seed.get(name, [])),
            "mean_psi_full": float(psi_full[valid_full].mean()) if valid_full.any() else float("nan"),
            "median_psi_full": float(psi_full[valid_full].median()) if valid_full.any() else float("nan"),
            "pct_stable_full": float((flag_full == "stable").sum() / max(1, int(valid_full.sum()))),
            "pct_shift_full": float((flag_full == "shift").sum() / max(1, int(valid_full.sum()))),
            "pct_broken_full": float((flag_full == "broken").sum() / max(1, int(valid_full.sum()))),
            "mean_psi_oot": float(psi_oot[valid_oot].mean()) if valid_oot.any() else float("nan"),
            "median_psi_oot": float(psi_oot[valid_oot].median()) if valid_oot.any() else float("nan"),
            "pct_stable_oot": float((flag_oot == "stable").sum() / max(1, int(valid_oot.sum()))),
            "pct_shift_oot": float((flag_oot == "shift").sum() / max(1, int(valid_oot.sum()))),
            "pct_broken_oot": float((flag_oot == "broken").sum() / max(1, int(valid_oot.sum()))),
            "mean_missing_rate": float(df["{0}_missing_rate".format(name)].dropna().mean()),
            "top5_shift_features": ";".join(top5),
        })
    return pd.DataFrame(rows)


def _build_top_shift(df, name, cn_map, top=50):
    cols = ["feature",
            "{0}_psi_full".format(name), "{0}_flag_full".format(name),
            "{0}_psi_oot".format(name), "{0}_flag_oot".format(name),
            "{0}_n".format(name), "{0}_missing_rate".format(name),
            "{0}_mean".format(name), "{0}_median".format(name),
            "oot_missing_rate", "oot_mean", "oot_median"]
    out = df[cols].dropna(subset=["{0}_psi_oot".format(name)]).copy()
    if cn_map:
        out = inject_cn_column(out, cn_map)
    return out.sort_values(
        "{0}_psi_oot".format(name), ascending=False).head(top).reset_index(drop=True)


# ---------- per-scope driver ----------

def _scope_dir_name(scope, features_version):
    if scope == "selected":
        return "selected_{0}".format(features_version)
    return scope


def _resolve_drop_cols(cfg):
    cols = {cfg["data"].get("label_column"),
            cfg["data"].get("time_column"),
            cfg["data"].get("treatment_column")}
    cols.update(cfg["data"].get("id_columns") or [])
    cols.discard(None)
    cols.discard("")
    return cols


def _run_one_scope(cfg, spec_map, scope, features_version, baseline_path,
                   time_col, oot_mask, seeds_dir, seed_names, n_bins,
                   out_root, cn_map):
    if scope == "selected":
        features, fpath = _read_selected_features(cfg, features_version)
        logger.info("Scope=selected: %d features from %s", len(features), fpath)
    else:
        features = _read_full_feature_list(baseline_path, _resolve_drop_cols(cfg))
        logger.info("Scope=full: %d features (CSV header minus label/time/id)",
                    len(features))

    seeds: Dict[str, pd.DataFrame] = {}
    missing_per_seed: Dict[str, List[str]] = {}
    for name in seed_names:
        p = Path(seeds_dir) / "{0}.csv".format(name)
        if not p.is_file():
            logger.warning("Seed file not found: %s — skipping seed %s", p, name)
            continue
        seed_header = pd.read_csv(p, nrows=0)
        available = [f for f in features if f in seed_header.columns]
        missing_per_seed[name] = sorted(set(features) - set(available))
        logger.info("Seed %s: %d/%d features present (%d missing)",
                    name, len(available), len(features),
                    len(missing_per_seed[name]))
        if not available:
            logger.warning("Seed %s shares 0 features with %s scope — "
                           "all per-feature stats will be NaN", name, scope)
            seeds[name] = pd.DataFrame()
        else:
            seeds[name] = read_full(p, columns=available)

    if not seeds:
        raise RuntimeError(
            "No seed CSVs found under {0}; expected files {1}".format(
                seeds_dir, ["{0}.csv".format(n) for n in seed_names]))

    df = _compare_features(cfg, spec_map, baseline_path, time_col, oot_mask,
                           seeds, features, n_bins)

    final_seed_names = list(seeds.keys())
    psi_matrix = _build_psi_matrix(df, final_seed_names, cn_map)
    stats_matrix = _build_stats_matrix(df, final_seed_names, cn_map)
    summary = _build_summary(df, final_seed_names, missing_per_seed)

    out_dir = out_root / _scope_dir_name(scope, features_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_matrix.to_csv(out_dir / "psi_matrix.csv", index=False)
    stats_matrix.to_csv(out_dir / "stats_matrix.csv", index=False)
    summary.to_csv(out_dir / "summary_per_seed.csv", index=False)
    for name in final_seed_names:
        _build_top_shift(df, name, cn_map).to_csv(
            out_dir / "top_shift_{0}.csv".format(name), index=False)

    logger.info("Scope=%s outputs written to %s", scope, out_dir)
    return {
        "scope": scope,
        "out_dir": str(out_dir),
        "n_features": len(features),
        "seeds": {name: int(len(seeds[name])) for name in final_seed_names},
        "missing_features_per_seed": {
            n: len(v) for n, v in missing_per_seed.items()},
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare ABCD seed populations' feature distributions "
                    "against the hzz_day baseline (dual: full + OOT).")
    ap.add_argument("--product", required=True,
                    help="Product name; configs/products/<name>.yaml.")
    ap.add_argument("--seeds-dir", default="data/seeds",
                    help="Directory holding <seed>.csv files. Relative paths "
                    "resolve against the repo root.")
    ap.add_argument("--seeds", default=",".join(DEFAULT_SEEDS),
                    help="Comma-separated seed names; expects "
                    "<seeds_dir>/<name>.csv for each.")
    ap.add_argument("--features-version", default=None,
                    help="Selected-features version (e.g. v1_auto). Defaults "
                    "to cfg.selected_features.active_version. Only used when "
                    "--scope includes 'selected'.")
    ap.add_argument("--scope", choices=["selected", "full", "both"],
                    default="both",
                    help="Which feature scope(s) to compare.")
    ap.add_argument("--oot-ratio", type=float, default=None,
                    help="Last fraction of baseline by cd_time as the OOT "
                    "(time-aligned) basis. Defaults to "
                    "cfg.training.split.ratios[-1].")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="PSI equal-frequency bin count. Defaults to "
                    "cfg.analysis.n_bins.")
    ap.add_argument("--baseline-path", default=None,
                    help="Override cfg.data.train_path.")
    ap.add_argument("--out-dir", default=None,
                    help="Override artifacts/<product>/seed_compare/.")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.product)

    repo_root = Path(cfg["_repo_root"])
    baseline_path = Path(args.baseline_path) if args.baseline_path else (
        repo_root / cfg["data"]["train_path"])
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    if not baseline_path.is_file():
        raise FileNotFoundError(
            "Baseline CSV not found: {0}".format(baseline_path))

    seeds_dir = Path(args.seeds_dir)
    if not seeds_dir.is_absolute():
        seeds_dir = repo_root / seeds_dir

    out_root = Path(args.out_dir) if args.out_dir else (
        artifacts_root(cfg) / "seed_compare")
    out_root.mkdir(parents=True, exist_ok=True)

    features_version = (args.features_version
                        or cfg["selected_features"]["active_version"])
    oot_ratio = (args.oot_ratio if args.oot_ratio is not None
                 else float(cfg["training"]["split"]["ratios"][-1]))
    n_bins = int(args.n_bins or cfg.get("analysis", {}).get("n_bins", 10))
    seed_names = [s.strip() for s in args.seeds.split(",") if s.strip()]
    scopes = ["selected", "full"] if args.scope == "both" else [args.scope]

    time_col = cfg["data"]["time_column"]
    if not time_col:
        raise ValueError(
            "data.time_column required to compute the OOT slice")
    logger.info("Reading %s for OOT mask construction...", time_col)
    times_df = pd.read_csv(baseline_path, usecols=[time_col])
    oot_mask = _build_oot_mask(times_df[time_col].values, oot_ratio)
    if oot_mask.any():
        cd_time_cutoff = int(
            times_df.loc[oot_mask, time_col].astype(np.int64).min())
    else:
        cd_time_cutoff = None
        logger.warning("OOT mask is empty — psi_oot will fall back to psi_full")
    logger.info("OOT slice: %d/%d rows (cd_time >= %s, ratio=%.3f)",
                int(oot_mask.sum()), oot_mask.size, cd_time_cutoff, oot_ratio)

    spec_map = build_missing_spec(cfg)
    cn_map = load_column_mapping(cfg)

    scope_results = []
    for sc in scopes:
        scope_results.append(_run_one_scope(
            cfg, spec_map, sc, features_version, baseline_path,
            time_col, oot_mask, seeds_dir, seed_names, n_bins,
            out_root, cn_map))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product": args.product,
        "baseline_path": str(baseline_path),
        "baseline_rows_full": int(oot_mask.size),
        "baseline_rows_oot": int(oot_mask.sum()),
        "cd_time_cutoff": cd_time_cutoff,
        "oot_ratio": oot_ratio,
        "n_bins": n_bins,
        "features_version": features_version,
        "seeds_dir": str(seeds_dir),
        "seeds_requested": seed_names,
        "scopes": scope_results,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("=" * 60)
    print("Seed distribution comparison complete.")
    print("  baseline rows full :", manifest["baseline_rows_full"])
    print("  baseline rows oot  :", manifest["baseline_rows_oot"])
    print("  oot cd_time cutoff :", manifest["cd_time_cutoff"])
    print("  features version   :", features_version)
    print("  out dir            :", out_root)
    for s in scope_results:
        seed_summary = ", ".join("{0}={1}".format(k, v)
                                 for k, v in s["seeds"].items())
        print("  scope={0:<8} -> {1}  seeds: {2}".format(
            s["scope"], s["out_dir"], seed_summary))


if __name__ == "__main__":
    main()
