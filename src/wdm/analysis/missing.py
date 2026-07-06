"""Per-feature missing-rate / n_unique / dtype stats.

Uses the same NaN-aware contract as the other analyses — sentinels/negatives/
empties already treated as NaN via preprocess.missing.to_nan_array.
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def missing_row_from_array(feat, raw_dtype_str, arr, mask, n_total):
    """Per-feature kernel shared by compute_missing_stats and the single-pass
    scan. `raw_dtype_str` is the pre-conversion pandas column dtype."""
    non_nan = arr[~np.isnan(arr)]
    return {
        "feature": feat,
        "missing_rate": float(mask.mean()) if n_total else 0.0,
        "n_unique": int(np.unique(non_nan).size) if non_nan.size else 0,
        "dtype": raw_dtype_str,
        "n_total": int(n_total),
    }


def compute_missing_stats(chunk_iter, spec_map, get_spec_fn):
    """Return DataFrame[feature, missing_rate, n_unique, dtype, n_total]."""
    from wdm.preprocess.missing import to_nan_array

    rows = []
    for df_chunk, block in chunk_iter:
        n_total = len(df_chunk)
        for feat in block:
            spec = get_spec_fn(spec_map, feat)
            raw = df_chunk[feat]
            arr, mask = to_nan_array(raw, spec)
            rows.append(missing_row_from_array(feat, str(raw.dtype), arr,
                                               mask, n_total))
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)
