"""Build a synthetic hzz_day training table for smoke-testing the wdm pipeline.

Generates ~1M rows × (1427 + 3) columns (cd_time, label_register, cust_id,
plus every feature in data/sample_feat_names.csv) with the high-dimensional,
sparse shape that hzz_day expects: lots of NaN, lots of legitimate zeros,
and mixed numeric distributions chosen by feature-name heuristic.

Output: data/hzz_day.csv (matches configs/products/hzz_day.yaml:train_path).

Usage:
    PYTHONPATH=src python3 scripts/build_hzz_day_synthetic.py
    PYTHONPATH=src python3 scripts/build_hzz_day_synthetic.py --rows 10000  # smoke
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from wdm.utils.logging import setup_logging
except Exception:  # pragma: no cover — script-only fallback
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

logger = logging.getLogger(__name__)


# ---------- feature-name heuristic ---------------------------------------- #

def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("_radio", "_ratio", "_rate", "_per", "_pct")):
        return "ratio"
    if n.startswith("is_") or n.endswith("_flag") or n.endswith("_indicator"):
        return "flag"
    if any(k in n for k in ("_std", "_cv", "_var")):
        return "std"
    if any(k in n for k in (
            "_amt", "_money", "_fee", "_balance",
            "_credit", "_loan", "_prin", "_cost", "_quota")):
        return "amount"
    if any(k in n for k in (
            "_days", "_day_diff", "_month", "_year",
            "_from_now", "_diff")):
        return "tdiff"
    if any(k in n for k in (
            "_num", "_cnt", "_count", "_allnum", "_orgnum",
            "_daynum", "_orgtypenum", "_times")):
        return "count"
    return "numeric"


def feature_params(rng, names):
    """Per-feature (type, miss_p, zero_p), drawn deterministically from rng."""
    types = [classify(n) for n in names]
    miss = (rng.beta(2.0, 4.0, len(names)) * 0.85 + 0.02).astype(np.float32)
    zero = (rng.beta(2.0, 3.0, len(names)) * 0.60).astype(np.float32)
    for i, t in enumerate(types):
        if t == "flag":
            miss[i] = rng.uniform(0.02, 0.20)
            zero[i] = 0.0  # flags are already binary
        elif t == "ratio":
            zero[i] = max(zero[i], 0.05)
        elif t == "amount":
            zero[i] = max(zero[i], 0.20)  # amounts often have legit zeros
    return types, miss, zero


def gen_column(rng, n, ftype, miss_p, zero_p):
    if ftype == "ratio":
        v = rng.beta(2.0, 5.0, n)
    elif ftype == "flag":
        v = (rng.random(n) < 0.20).astype(np.float32)
    elif ftype == "std":
        v = rng.lognormal(1.5, 1.0, n)
    elif ftype == "amount":
        v = rng.lognormal(6.5, 1.6, n)
    elif ftype == "tdiff":
        v = rng.integers(0, 720, n).astype(np.float32)
    elif ftype == "count":
        v = rng.negative_binomial(2, 0.30, n).astype(np.float32)
    else:  # numeric
        v = rng.lognormal(3.0, 1.5, n)
    v = v.astype(np.float32)
    if zero_p > 0:
        v[rng.random(n) < zero_p] = 0.0
    if miss_p > 0:
        v[rng.random(n) < miss_p] = np.nan
    return v


# ---------- label injection ----------------------------------------------- #

def pick_signal_cols(rng, names, types, k=10):
    pool = [i for i, t in enumerate(types)
            if t in ("amount", "numeric", "count", "ratio")]
    idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
    weights = rng.normal(0.0, 0.6, size=len(idx)).astype(np.float32)
    return [(names[i], float(w)) for i, w in zip(idx, weights)]


def build_label(df, rng, base_rate, signal_cols, col_stats):
    """Sigmoid(intercept + Σ wᵢ·zᵢ). col_stats={col: (mean, std)} from this chunk."""
    n = len(df)
    logit = np.zeros(n, dtype=np.float32)
    for c, w in signal_cols:
        if c not in df.columns:
            continue
        v = df[c].values.astype(np.float32)
        m = ~np.isnan(v)
        if not m.any():
            continue
        mu, sd = col_stats[c]
        z = np.zeros(n, dtype=np.float32)
        z[m] = (v[m] - mu) / (sd + 1e-6)
        logit += z * w
    intercept = float(np.log(base_rate / (1.0 - base_rate)))
    p = 1.0 / (1.0 + np.exp(-(logit + intercept)))
    return (rng.random(n) < p).astype(np.int8)


# ---------- main --------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--out", default="data/hzz_day.csv")
    ap.add_argument("--feat-names", default="data/sample_feat_names.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2024-12-31")
    ap.add_argument("--pos-rate", type=float, default=0.03)
    args = ap.parse_args()

    setup_logging()
    repo = Path(__file__).resolve().parents[1]
    feat_path = repo / args.feat_names
    out_path = repo / args.out

    feat_df = pd.read_csv(feat_path)
    name_col = "name" if "name" in feat_df.columns else feat_df.columns[0]
    feat_names = feat_df[name_col].astype(str).tolist()
    logger.info("Loaded %d feature names from %s", len(feat_names), feat_path)

    rng = np.random.default_rng(args.seed)
    types, miss_ps, zero_ps = feature_params(rng, feat_names)
    logger.info("Type counts: %s", {
        t: types.count(t) for t in sorted(set(types))})

    signal_cols = pick_signal_cols(rng, feat_names, types, k=10)
    logger.info("Signal columns (col, weight): %s",
                [(c, round(w, 3)) for c, w in signal_cols])

    # Stable per-signal-column stats: estimate from the typed distribution
    # rather than the noisy first chunk, so logit weights apply consistently
    # across chunks.
    col_stats = {}
    for c, _ in signal_cols:
        i = feat_names.index(c)
        sample = gen_column(np.random.default_rng(args.seed + 1 + i),
                            20_000, types[i], 0.0, 0.0)
        sample = sample[~np.isnan(sample)]
        col_stats[c] = (float(np.mean(sample)), float(np.std(sample)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["cd_time", "label_register", "cust_id"] + feat_names
    with out_path.open("w", newline="") as fh:
        fh.write(",".join(header) + "\n")

    base = pd.Timestamp(args.start_date)
    span = (pd.Timestamp(args.end_date) - base).days + 1
    cust_id_offset = 10_000_000
    label_total = 0
    written = 0
    for start in range(0, args.rows, args.chunk):
        size = min(args.chunk, args.rows - start)

        day_off = rng.integers(0, span, size)
        ts = base + pd.to_timedelta(day_off.astype(np.int64), unit="D")
        cd_time = (ts.year * 10000 + ts.month * 100 + ts.day).astype(np.int64)

        cust_id = (np.arange(start, start + size, dtype=np.int64)
                   + cust_id_offset)

        feat_arrays = {
            n_: gen_column(rng, size, t_, m_, z_)
            for n_, t_, m_, z_ in zip(feat_names, types, miss_ps, zero_ps)
        }
        df = pd.DataFrame(feat_arrays)

        lbl = build_label(df, rng, args.pos_rate, signal_cols, col_stats)
        df.insert(0, "cust_id", cust_id)
        df.insert(0, "label_register", lbl)
        df.insert(0, "cd_time", cd_time)

        df.to_csv(out_path, mode="a", header=False, index=False,
                  na_rep="", float_format="%.4g")
        written += size
        label_total += int(lbl.sum())
        logger.info("Wrote %d / %d rows (cumulative pos rate: %.4f)",
                    written, args.rows, label_total / written)

    print("Done. Output: {0}".format(out_path))
    print("Shape: {0} rows × {1} columns".format(args.rows, len(header)))
    print("Pos rate: {0:.4f}".format(label_total / args.rows))


if __name__ == "__main__":
    main()
