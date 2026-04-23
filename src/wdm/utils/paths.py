"""Artifact path resolver + column-mapping loader.

Artifact layout under artifacts/<product>/:
  analysis/
    column_index.json
    report/{summary,psi,iv_woe,lift,missing,correlation_edges,families,semantic_groups}.csv
    report/index.html
    per_feature/<feat>/{dist,woe,psi,missing,lift}.png
    per_family/<family_base>/{iv_by_window,woe_overlay,psi_by_window}.png
    sample.parquet
  selected_features/v*.txt
  models/<run_id>/...
"""
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def artifacts_root(cfg):
    return Path(cfg["_repo_root"]) / "artifacts" / cfg["name"]


def analysis_dir(cfg):
    return artifacts_root(cfg) / "analysis"


def report_dir(cfg):
    return analysis_dir(cfg) / "report"


def per_feature_dir(cfg, feature):
    return analysis_dir(cfg) / "per_feature" / feature


def per_family_dir(cfg, family_base):
    return analysis_dir(cfg) / "per_family" / family_base


def selected_features_dir(cfg):
    configured = cfg.get("selected_features", {}).get("versions_dir")
    if configured:
        return Path(cfg["_repo_root"]) / configured
    return artifacts_root(cfg) / "selected_features"


def selected_features_file(cfg, version=None):
    v = version or cfg["selected_features"]["active_version"]
    return selected_features_dir(cfg) / "{0}.txt".format(v)


def model_run_dir(cfg, run_id):
    return artifacts_root(cfg) / "models" / run_id


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_column_mapping(cfg):
    """Return {english_name: chinese_name}. Missing keys fall back to English."""
    path_rel = cfg["data"].get("column_mapping")
    if not path_rel:
        return {}
    path = Path(cfg["_repo_root"]) / path_rel
    if not path.is_file():
        logger.warning("column_mapping file not found: %s", path)
        return {}
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8")
    if "feature" not in df.columns or "feature_cn" not in df.columns:
        raise ValueError("column_mapping CSV must have columns: feature, feature_cn")
    return dict(zip(df["feature"].astype(str), df["feature_cn"].astype(str)))


def cn(mapping, feature):
    """mapping.get(feature, feature) — guaranteed fallback."""
    if not mapping:
        return feature
    return mapping.get(feature, feature)


def inject_cn_column(df, mapping, feature_col="feature", cn_col="feature_cn"):
    """Insert `feature_cn` as the 2nd column of df (right after `feature`)."""
    if feature_col not in df.columns:
        return df
    df = df.copy()
    cn_series = df[feature_col].map(lambda x: cn(mapping, x))
    if cn_col in df.columns:
        df[cn_col] = cn_series
    else:
        pos = list(df.columns).index(feature_col) + 1
        df.insert(pos, cn_col, cn_series)
    return df
