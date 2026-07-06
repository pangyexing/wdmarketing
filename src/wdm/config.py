"""Config loading: global.yaml + product YAML merged via *_overrides.

Public entrypoint: load_config(product_name) -> dict.

Search order for configs root:
1. $WDM_CONFIG_DIR if set
2. <repo>/configs (discovered by walking upward from this file)
3. cwd/configs

The returned dict has overrides already applied; callers treat it as flat config.
"""
import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_OVERRIDE_KEYS = ("analysis_overrides", "training_overrides", "io_overrides",
                  "export_overrides", "plots_overrides")


def _find_configs_dir(hint: Optional[Path] = None) -> Path:
    env = os.environ.get("WDM_CONFIG_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    here = hint or Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "configs"
        if candidate.is_dir() and (candidate / "global.yaml").is_file():
            return candidate
    cwd = Path.cwd() / "configs"
    if cwd.is_dir():
        return cwd
    raise FileNotFoundError(
        "Could not locate configs/ directory. Set $WDM_CONFIG_DIR or "
        "run from the repo root.")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base; override wins on conflicts.
    Lists are replaced wholesale (not merged) — predictable and simple.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_overrides(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Apply analysis_overrides / training_overrides / ... on top of their base sections."""
    out = copy.deepcopy(merged)
    for key in _OVERRIDE_KEYS:
        overrides = out.pop(key, None)
        if not overrides:
            continue
        base_key = key.replace("_overrides", "")
        out[base_key] = _deep_merge(out.get(base_key, {}) or {}, overrides)
    return out


def _validate(cfg: Dict[str, Any]) -> None:
    """Lightweight sanity checks — hard-fail on misconfigurations that would corrupt runs."""
    required_top = ["name", "data", "missing", "analysis", "training", "io", "plots",
                    "export", "feature_groups", "selected_features"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError("Config missing required top-level keys: {0}".format(missing))

    data = cfg["data"]
    if not data.get("train_path"):
        raise ValueError("data.train_path must be set")
    if not data.get("label_column"):
        raise ValueError("data.label_column must be set")

    exclude_rows = data.get("exclude_rows")
    if exclude_rows is not None:
        if not isinstance(exclude_rows, list):
            raise ValueError("data.exclude_rows must be a list of {column, values} dicts")
        for rule in exclude_rows:
            if not isinstance(rule, dict) or not rule.get("column") \
                    or not isinstance(rule.get("values"), list) or not rule["values"]:
                raise ValueError(
                    "data.exclude_rows entries must be {{column: <str>, values: [..]}}; got {0}"
                    .format(rule))

    tuner_objective = cfg["training"].get("tuner_objective", "aucpr")
    if tuner_objective not in ("aucpr", "precision_at_k"):
        raise ValueError("training.tuner_objective must be 'aucpr' or 'precision_at_k'")

    cv_strategy = cfg["training"].get("cv_strategy", "stratified")
    if cv_strategy not in ("stratified", "time_forward"):
        raise ValueError("training.cv_strategy must be 'stratified' or 'time_forward'")
    if cv_strategy == "time_forward" and not data.get("time_column"):
        raise ValueError("training.cv_strategy='time_forward' requires data.time_column")

    sw = cfg["training"].get("sample_weight")
    if sw is not None:
        if not isinstance(sw, dict) or not sw.get("column") \
                or not isinstance(sw.get("mapping"), dict) or not sw["mapping"]:
            raise ValueError(
                "training.sample_weight must be {column: <str>, mapping: {value: weight}, "
                "default: <num, optional>}")
        bad_w = [v for v in list(sw["mapping"].values()) + [sw.get("default", 1.0)]
                 if not isinstance(v, (int, float)) or float(v) < 0]
        if bad_w:
            raise ValueError("training.sample_weight weights must be non-negative "
                             "numbers; got {0}".format(bad_w))

    calib = cfg["export"].get("calibration")
    if calib is not None and calib.get("enabled", False):
        if calib.get("method", "isotonic") != "isotonic":
            raise ValueError("export.calibration.method only supports 'isotonic'")

    split = cfg["training"].get("split", {})
    strategy = split.get("strategy", "stratified")
    if strategy not in ("stratified", "time"):
        raise ValueError("training.split.strategy must be 'stratified' or 'time'")
    ratios = split.get("ratios", [])
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("training.split.ratios must be 3 numbers summing to 1.0")
    if strategy == "time" and not data.get("time_column"):
        raise ValueError("split.strategy='time' requires data.time_column")

    fill_strategy = cfg["missing"]["global"].get("fill_strategy")
    valid = {"constant", "median", "mean", "zero", "special", "keep_nan"}
    if fill_strategy not in valid:
        raise ValueError("missing.global.fill_strategy must be one of {0}".format(valid))

    pf = cfg["analysis"].get("psi_flag_thresholds") or {}
    if pf:
        shift_t = float(pf.get("shift", 0.10))
        broken_t = float(pf.get("broken", 0.25))
        if not (0 <= shift_t <= broken_t):
            raise ValueError("analysis.psi_flag_thresholds requires 0 <= shift <= broken")

    s1n = cfg["analysis"].get("stage1_top_n")
    if s1n is not None and (not isinstance(s1n, int) or isinstance(s1n, bool) or s1n <= 0):
        raise ValueError("analysis.stage1_top_n must be a positive integer or null")

    ni = cfg["analysis"].get("null_importance") or {}
    if ni:
        if not isinstance(ni.get("enabled", False), bool):
            raise ValueError("analysis.null_importance.enabled must be a boolean")
        for k in ("n_actual_runs", "n_null_runs", "n_boost_rounds"):
            v = ni.get(k)
            if v is not None and (not isinstance(v, int) or v <= 0):
                raise ValueError(
                    "analysis.null_importance.{0} must be a positive integer".format(k))
        kp = ni.get("keep_percentile")
        if kp is not None and not (0 < float(kp) < 100):
            raise ValueError("analysis.null_importance.keep_percentile must be in (0, 100)")

    sc = cfg["io"].get("scan_cache") or {}
    if sc and not isinstance(sc.get("enabled", True), bool):
        raise ValueError("io.scan_cache.enabled must be a boolean")

    fmts = cfg["export"].get("model_format", ["json"])
    if isinstance(fmts, str):
        fmts = [fmts]
    valid_fmts = {"json", "bin", "binary", "ubj"}
    bad = [f for f in fmts if str(f).strip().lower() not in valid_fmts]
    if bad:
        raise ValueError("export.model_format has invalid entries {0}; "
                         "valid: {1}".format(bad, sorted(valid_fmts)))

    _validate_feature_groups(cfg)


def _validate_feature_groups(cfg: Dict[str, Any]) -> None:
    """Validate feature_groups.window_patterns / window_pattern up-front so
    regex / preset mistakes fail at config load, not mid-Stage-1.
    """
    fg = cfg.get("feature_groups") or {}
    if not fg.get("enable_window_family", True):
        return
    patterns = fg.get("window_patterns")
    single = fg.get("window_pattern")
    if not patterns and not single:
        raise ValueError("feature_groups must define either 'window_patterns' "
                         "(list) or 'window_pattern' (single regex)")

    from wdm.analysis.family import _resolve_patterns  # lazy to avoid cycles
    try:
        _resolve_patterns(cfg)
    except (re.error, ValueError) as exc:
        raise ValueError("feature_groups pattern config invalid: {0}".format(exc))


def load_config(product_name: str,
                configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load and merge global.yaml with configs/products/<product_name>.yaml.

    Returns a fully-resolved dict (overrides applied, validated).
    """
    cfg_dir = configs_dir or _find_configs_dir()
    global_path = cfg_dir / "global.yaml"
    product_path = cfg_dir / "products" / "{0}.yaml".format(product_name)
    if not global_path.is_file():
        raise FileNotFoundError("Missing {0}".format(global_path))
    if not product_path.is_file():
        raise FileNotFoundError("Missing {0}".format(product_path))

    with open(global_path, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f) or {}
    with open(product_path, "r", encoding="utf-8") as f:
        product_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(global_cfg, product_cfg)
    resolved = _apply_overrides(merged)

    resolved["_configs_dir"] = str(cfg_dir)
    resolved["_repo_root"] = str(cfg_dir.parent)
    resolved.setdefault("name", product_name)

    _validate(resolved)
    logger.info("Loaded config for product %s", product_name)
    return resolved


def repo_path(cfg: Dict[str, Any], rel: str) -> Path:
    """Resolve a relative path against the repo root (configs/..)."""
    return Path(cfg["_repo_root"]) / rel
