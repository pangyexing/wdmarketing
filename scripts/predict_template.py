"""predict.py template — single-file deployment script.

The exporter string-substitutes the constants at the top of this file (model
bundle filenames and feature list) and writes the rendered copy to the
model run dir. Keep this file self-contained — only use xgboost/numpy/pandas/json/argparse.

Contract:
  * Input CSV = raw business data. Missing values, sentinels, negatives, empty
    fields are allowed — they are handled internally via missing_spec.json.
  * Output CSV = <id_col>, score (raw model probability). When the bundle
    contains calibration.json an extra score_calibrated column is emitted
    (isotonic curve replayed via np.interp) — use THAT column when fusing
    scores across models; `score` alone keeps legacy single-model behavior.
  * `python predict.py --validate` reproduces y_pred_expected from
    validation_samples.csv to 1e-6, exercising the full pipeline end-to-end;
    when calibration is present it also checks y_pred_calibrated_expected.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


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

        # 4. fill
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
        # Auto-detect the serialized model: prefer native binary, fall back to
        # json/ubj. xgboost identifies the format from file contents.
        model_path = None
        for cand in ("booster.bin", "booster.json", "booster.ubj"):
            p = self.bundle / cand
            if p.is_file():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(
                "No booster model found in {0} (looked for booster.bin/.json/.ubj)"
                .format(self.bundle))
        self.booster.load_model(str(model_path))
        with open(self.bundle / "feature_list.txt", "r", encoding="utf-8") as f:
            # Skip comment header lines; keep order
            self.feature_list = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        with open(self.bundle / "missing_spec.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.spec_map = payload["specs"]
        self.fitted = payload["fitted"]
        # Precompute indicator features (names ending with __isnan)
        self.indicator_features = [f for f in self.feature_list if f.endswith("__isnan")]
        self.base_features = [f for f in self.feature_list if not f.endswith("__isnan")]
        # Optional isotonic calibration lookup table (raw score -> probability).
        self.calib_x = None
        self.calib_y = None
        calib_path = self.bundle / "calibration.json"
        if calib_path.is_file():
            with open(str(calib_path), "r", encoding="utf-8") as f:
                table = json.load(f)
            self.calib_x = np.asarray(table["x"], dtype=np.float64)
            self.calib_y = np.asarray(table["y"], dtype=np.float64)

    @property
    def has_calibration(self):
        return self.calib_x is not None

    def calibrate(self, raw_scores):
        """Replay the isotonic curve (constant extrapolation at the ends —
        identical to the training-time fit). No-op without calibration.json."""
        s = np.asarray(raw_scores, dtype=np.float64)
        if self.calib_x is None:
            return s
        return np.interp(s, self.calib_x, self.calib_y)

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
        try:
            best_iter = getattr(self.booster, "best_iteration", None)
            if best_iter is not None:
                return self.booster.predict(dmat, iteration_range=(0, best_iter + 1))
        except Exception:
            pass
        return self.booster.predict(dmat)


def cmd_predict(args):
    pr = Predictor(args.bundle or Path(__file__).parent)
    df = pd.read_csv(args.input)
    scores = pr.predict_proba(df)
    out = pd.DataFrame({"row_index": np.arange(len(df)), "score": scores})
    if pr.has_calibration:
        out["score_calibrated"] = pr.calibrate(scores)
    out.to_csv(args.output, index=False)
    print("Wrote {0} rows to {1}".format(len(df), args.output))


_NON_FEATURE_COLS = ("y_true", "y_pred_expected", "y_pred_calibrated_expected")


def cmd_validate(args):
    pr = Predictor(args.bundle or Path(__file__).parent)
    path = Path(args.bundle or Path(__file__).parent) / "validation_samples.csv"
    df = pd.read_csv(path)
    if "y_pred_expected" not in df.columns:
        raise SystemExit("validation_samples.csv is missing y_pred_expected")
    expected = df["y_pred_expected"].values.astype(np.float64)
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    scores = pr.predict_proba(df[feature_cols])
    tol = float(args.tol)
    failed = False

    diff = np.abs(scores - expected)
    worst = float(diff.max()) if diff.size else 0.0
    if worst > tol:
        failed = True
        print("FAIL (raw): max |pred - expected| = {0:.3e} > tol {1:.0e}".format(worst, tol))
        print(pd.DataFrame({"expected": expected, "actual": scores, "diff": diff})
              .sort_values("diff", ascending=False).head(5).to_string(index=False))
    else:
        print("OK (raw): all {0} samples within tolerance (max |diff| = {1:.3e})".format(
            len(df), worst))

    if "y_pred_calibrated_expected" in df.columns and pr.has_calibration:
        expected_cal = df["y_pred_calibrated_expected"].values.astype(np.float64)
        diff_cal = np.abs(pr.calibrate(scores) - expected_cal)
        worst_cal = float(diff_cal.max()) if diff_cal.size else 0.0
        if worst_cal > tol:
            failed = True
            print("FAIL (calibrated): max |diff| = {0:.3e} > tol {1:.0e}".format(
                worst_cal, tol))
        else:
            print("OK (calibrated): all {0} samples within tolerance "
                  "(max |diff| = {1:.3e})".format(len(df), worst_cal))
    elif "y_pred_calibrated_expected" in df.columns:
        failed = True
        print("FAIL: validation_samples.csv has y_pred_calibrated_expected but the "
              "bundle has no calibration.json")

    if failed:
        raise SystemExit(1)


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
