"""One-shot column inventory: read only the header + a small sample for dtype probing.

Produces a JSON index at artifacts/<product>/analysis/column_index.json:
{
  "n_columns": 14,
  "columns": ["age", "job", ...],
  "dtypes": {"age": "int64", "job": "int64", ...},
  "n_sample_rows": 10000,
  "label_column": "y",
  "time_column": null,
  "id_columns": []
}
"""
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from wdm.utils.paths import analysis_dir, ensure_dirs

logger = logging.getLogger(__name__)


def scan_columns(cfg, sample_rows=10000):
    """Read header + sample_rows, probe dtypes, write column_index.json."""
    path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    if not path.is_file():
        raise FileNotFoundError("data.train_path not found: {0}".format(path))

    header_df = pd.read_csv(path, nrows=0)
    columns = list(header_df.columns)

    sample_df = pd.read_csv(path, nrows=sample_rows)
    dtypes = {c: str(sample_df[c].dtype) for c in sample_df.columns}

    label_col = cfg["data"]["label_column"]
    time_col = cfg["data"].get("time_column")
    id_cols = cfg["data"].get("id_columns") or []

    if label_col not in columns:
        raise ValueError("label_column '{0}' not in data columns: {1}".format(
            label_col, columns))
    if time_col and time_col not in columns:
        raise ValueError("time_column '{0}' not in data columns".format(time_col))
    for c in id_cols:
        if c not in columns:
            raise ValueError("id column '{0}' not in data columns".format(c))

    # Feature columns = everything except label, time, id, treatment
    excluded = {label_col}
    if time_col:
        excluded.add(time_col)
    for c in id_cols:
        excluded.add(c)
    treatment = cfg["data"].get("treatment_column")
    if treatment:
        excluded.add(treatment)
    features = [c for c in columns if c not in excluded]

    out = {
        "n_columns": len(columns),
        "columns": columns,
        "features": features,
        "dtypes": dtypes,
        "n_sample_rows": int(min(sample_rows, len(sample_df))),
        "label_column": label_col,
        "time_column": time_col,
        "id_columns": id_cols,
        "treatment_column": treatment,
        "data_path": str(path),
    }

    out_dir = analysis_dir(cfg)
    ensure_dirs(out_dir)
    out_path = out_dir / "column_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Wrote column index: %s (%d features)", out_path, len(features))
    return out
