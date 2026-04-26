"""predict.py template — single-file deployment script.

The exporter copies this file verbatim into the model run directory alongside
the bundle artifacts (booster.json, feature_list.txt, missing_spec.json,
run_manifest.json, validation_samples.csv). predict.py resolves every bundle
file relative to its own location, so no post-copy substitution happens.

Keep this file self-contained — only use xgboost/numpy/pandas/json/argparse.

Contract:
  * Input CSV = raw business data. Missing values, sentinels, negatives, empty
    fields are allowed — they are handled internally via missing_spec.json.
  * Output CSV = <id_col>, score
  * `python predict.py --validate` reproduces y_pred_expected from
    validation_samples.csv to 1e-6, exercising the full pipeline end-to-end.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# Highest missing_spec.json schema version this predict.py can read. Keep in
# sync with MISSING_SPEC_SCHEMA_VERSION in wdm/preprocess/missing.py.
_SUPPORTED_SPEC_SCHEMA = 2


def _apply_missing_rules(df, spec_map, fitted):
    """Replay the training-time missing rules on a raw DataFrame.

    Replicates preprocess.missing behavior without depending on the wdm package,
    so the deploy bundle has no runtime coupling to the training codebase.
    """
    out = {}
    for feat in df.columns:
        spec = spec_map.get(feat) or spec_map.get("__default__") or {}
        fs = fitted.get(feat)
        raw = df[feat]

        # 1. coerce to numeric, treating empty / None / NULL as NaN
        s = raw
        if not pd.api.types.is_numeric_dtype(s):
            if spec.get("treat_empty_as_missing", True):
                s = s.replace({"": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan})
            s = pd.to_numeric(s, errors="coerce")
        arr = s.astype(np.float64).values

        # 2. sentinels
        sentinels = spec.get("sentinels", []) or []
        for v in sentinels:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            arr = np.where(arr == vf, np.nan, arr)

        # 3. negatives
        if spec.get("treat_negative_as_missing", True):
            arr = np.where(arr < 0, np.nan, arr)

        # 4. zeros (deploy-time replays Stage-2's training flag only;
        # the analysis flag is irrelevant at predict time)
        if spec.get("training_treat_zero_as_missing", False):
            arr = np.where(arr == 0.0, np.nan, arr)

        # 5. fill
        if fs is None:
            strategy = spec.get("fill_strategy", "constant")
            fill = float(spec.get("fill_constant", -999.0))
        else:
            strategy = fs.get("fill_strategy", "constant")
            fill = fs.get("fill_value")
        if strategy != "keep_nan" and fill is not None and not (
                isinstance(fill, float) and math.isnan(fill)):
            arr = np.where(np.isnan(arr), float(fill), arr)
        out[feat] = arr
    return pd.DataFrame(out, index=df.index)


class Predictor(object):
    def __init__(self, bundle_dir):
        self.bundle = Path(bundle_dir)
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.bundle / "booster.json"))
        with open(self.bundle / "feature_list.txt", "r", encoding="utf-8") as f:
            # Skip comment header lines; keep order
            self.feature_list = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        with open(self.bundle / "missing_spec.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        spec_version = int(payload.get("schema_version", 0))
        if spec_version > _SUPPORTED_SPEC_SCHEMA:
            raise ValueError(
                "missing_spec.json schema_version={0} is newer than this "
                "predict.py supports (max {1}). Re-export the bundle with a "
                "predict.py from the same training run.".format(
                    spec_version, _SUPPORTED_SPEC_SCHEMA))
        self.spec_map = payload["specs"]
        self.fitted = payload["fitted"]
        # Precompute indicator features (names ending with __isnan)
        self.indicator_features = [f for f in self.feature_list if f.endswith("__isnan")]
        self.base_features = [f for f in self.feature_list if not f.endswith("__isnan")]

        # best_iteration: prefer the manifest value (exporter stamps it from
        # the live booster) over booster.best_iteration, since the attribute
        # may or may not survive save/load across xgboost versions. Falling
        # back to the attribute keeps this predict.py compatible with older
        # bundles that predate the manifest field.
        self.best_iteration = None
        manifest_path = self.bundle / "run_manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    mf = json.load(f)
                bi = mf.get("best_iteration")
                if bi is not None:
                    self.best_iteration = int(bi)
            except (ValueError, OSError, TypeError):
                pass
        if self.best_iteration is None:
            bi_attr = getattr(self.booster, "best_iteration", None)
            if bi_attr is not None:
                try:
                    self.best_iteration = int(bi_attr)
                except (TypeError, ValueError):
                    pass

        # Verify feature_list.txt is consistent with the booster. A silent
        # mismatch would produce plausible-looking but wrong scores because
        # XGBoost's positional DMatrix has no way to flag reordered columns.
        try:
            booster_n = int(self.booster.num_features())
        except Exception:
            booster_n = None
        if booster_n is not None and booster_n != len(self.feature_list):
            raise ValueError(
                "feature_list.txt has {0} entries but booster.json expects {1} "
                "features — bundle is inconsistent. Re-export from training.".format(
                    len(self.feature_list), booster_n))

    def _apply(self, df_raw):
        needed = set(self.base_features)
        missing = needed - set(df_raw.columns)
        if missing:
            raise ValueError("Input is missing required columns: {0}".format(sorted(missing)))
        extras = set(df_raw.columns) - needed
        if extras:
            # Permit extra columns; log to stderr only so scoring isn't interrupted
            sys.stderr.write("[predict] ignored extra columns: {0}\n".format(sorted(extras)))
        base_df = df_raw[self.base_features]
        applied = _apply_missing_rules(base_df, self.spec_map, self.fitted)

        # All-NA rows (every base feature missing under the training rules)
        # produce a DMatrix row of fill values / NaN. The booster's default
        # direction still returns a score, but it's uninformative — warn so
        # operators can triage upstream data-delivery issues instead of
        # trusting the number silently.
        all_na = np.ones(len(df_raw), dtype=bool)
        for feat in self.base_features:
            spec = self.spec_map.get(feat) or self.spec_map.get("__default__") or {}
            mask_df = _apply_missing_rules(
                pd.DataFrame({feat: df_raw[feat]}),
                {feat: {**spec, "fill_strategy": "keep_nan"},
                 "__default__": self.spec_map.get("__default__", {})},
                {feat: {"fill_strategy": "keep_nan", "fill_value": float("nan")}})
            all_na &= np.isnan(mask_df[feat].values)
            if not all_na.any():
                break
        n_all_na = int(all_na.sum())
        if n_all_na:
            rate = 100.0 * n_all_na / max(1, len(df_raw))
            sys.stderr.write(
                "[predict] WARNING: {0}/{1} rows ({2:.2f}%) have every base "
                "feature missing — scores for these rows reflect the booster's "
                "default-direction prior, not real signal.\n".format(
                    n_all_na, len(df_raw), rate))

        # Recompute __isnan indicators from raw values (not from post-fill)
        frames = [applied]
        for ind in self.indicator_features:
            base = ind[:-len("__isnan")]
            spec = self.spec_map.get(base) or self.spec_map.get("__default__") or {}
            mask_df = pd.DataFrame({base: df_raw[base]})
            # Run missing rules but record mask instead of fill
            nan_df = _apply_missing_rules(
                mask_df,
                {base: {**spec, "fill_strategy": "keep_nan"},
                 "__default__": self.spec_map.get("__default__", {})},
                {base: {"fill_strategy": "keep_nan", "fill_value": float("nan")}})
            frames.append(pd.DataFrame({ind: np.isnan(nan_df[base].values).astype(np.int8)},
                                       index=df_raw.index))
        full = pd.concat(frames, axis=1)
        return full[self.feature_list].values.astype(np.float32)

    def predict_proba(self, df_raw):
        X = self._apply(df_raw)
        dmat = xgb.DMatrix(X)
        if self.best_iteration is not None:
            try:
                return self.booster.predict(
                    dmat, iteration_range=(0, self.best_iteration + 1))
            except Exception:
                # Older xgboost variants don't accept iteration_range; fall
                # through to the all-trees prediction below.
                pass
        return self.booster.predict(dmat)


def cmd_predict(args):
    pr = Predictor(args.bundle or Path(__file__).parent)
    df = pd.read_csv(args.input)
    scores = pr.predict_proba(df)
    out = pd.DataFrame({"row_index": np.arange(len(df)), "score": scores})
    out.to_csv(args.output, index=False)
    print("Wrote {0} rows to {1}".format(len(df), args.output))


def cmd_validate(args):
    pr = Predictor(args.bundle or Path(__file__).parent)
    path = Path(args.bundle or Path(__file__).parent) / "validation_samples.csv"
    df = pd.read_csv(path)
    if "y_pred_expected" not in df.columns:
        raise SystemExit("validation_samples.csv is missing y_pred_expected")
    expected = df["y_pred_expected"].values.astype(np.float64)
    feature_cols = [c for c in df.columns if c not in ("y_true", "y_pred_expected")]
    scores = pr.predict_proba(df[feature_cols])
    diff = np.abs(scores - expected)
    worst = float(diff.max()) if diff.size else 0.0
    tol = float(args.tol)
    if worst > tol:
        print("FAIL: max |pred - expected| = {0:.3e} > tol {1:.0e}".format(worst, tol))
        print(pd.DataFrame({"expected": expected, "actual": scores, "diff": diff})
              .sort_values("diff", ascending=False).head(5).to_string(index=False))
        raise SystemExit(1)
    print("OK: all {0} samples within tolerance (max |diff| = {1:.3e})".format(
        len(df), worst))


def main():
    ap = argparse.ArgumentParser(description="Apply the XGBoost model to raw CSV.")
    sub = ap.add_subparsers(dest="mode")
    p_pred = sub.add_parser("predict", help="Score a CSV.")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.add_argument("--bundle", default=None,
                        help="Path to the model bundle directory (defaults to predict.py's own dir).")
    p_val = sub.add_parser("validate", help="Run validation_samples.csv sanity check.")
    p_val.add_argument("--bundle", default=None)
    p_val.add_argument("--tol", default=1e-6)

    # backwards-compat: allow --input/--output/--validate without subcommand
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--tol", default=1e-6)

    args = ap.parse_args()
    if args.mode == "predict":
        cmd_predict(args)
    elif args.mode == "validate":
        cmd_validate(args)
    elif args.validate:
        args.mode = "validate"
        cmd_validate(args)
    elif args.input and args.output:
        args.mode = "predict"
        cmd_predict(args)
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
