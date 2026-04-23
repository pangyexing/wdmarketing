"""IV / WOE computation, NaN-aware.

Input to all public functions is the output of preprocess.missing.to_nan_array —
NaN represents missing. Never accept post-fill values (e.g., -999) here.

When `missing_as_bin=True`, NaN forms its own bin; its WOE is meaningful and
contributes to IV. This matters because "is_missing" often carries real signal
for marketing response (e.g., pdays=-1 means "never contacted before").
"""
import dataclasses
import logging
import math
from typing import List, Optional

import numpy as np
import pandas as pd

from wdm.utils.binning import (
    bin_counts, digitize_with_missing, equal_freq_edges, tree_edges,
)

logger = logging.getLogger(__name__)

_EPS = 1e-6


@dataclasses.dataclass
class BinSpec:
    feature: str
    edges: List[float]
    bin_counts: List[int]          # length = n_bins
    pos_counts: List[int]          # length = n_bins
    woe_values: List[float]        # length = n_bins
    missing_n: int                 # samples in the missing bin (if missing_as_bin)
    missing_pos: int
    missing_woe: Optional[float]   # None when no missing or missing_as_bin=False
    iv: float
    monotonic: bool

    def to_dict(self):
        return dataclasses.asdict(self)


def _woe_for_bin(pos_in_bin, neg_in_bin, total_pos, total_neg):
    p = (pos_in_bin + _EPS) / (total_pos + _EPS)
    n = (neg_in_bin + _EPS) / (total_neg + _EPS)
    return math.log(p / n)


def _is_monotonic(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return True
    diffs = np.diff(arr)
    return bool(np.all(diffs >= -_EPS) or np.all(diffs <= _EPS))


def woe_bin(values_nan, y, feature,
            n_bins=10, strategy="equal_freq", missing_as_bin=True):
    """Compute a BinSpec from NaN-aware values + labels.

    values_nan: float array with NaN for missing.
    y: 0/1 label array aligned with values_nan.
    """
    values = np.asarray(values_nan, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    if values.shape[0] != y.shape[0]:
        raise ValueError("values and y length mismatch")

    if strategy == "equal_freq":
        edges = equal_freq_edges(values, n_bins=n_bins)
    elif strategy == "tree":
        edges = tree_edges(values, y, n_bins=n_bins)
    else:
        raise ValueError("Unknown strategy: {0}".format(strategy))

    n_real_bins = max(1, edges.size - 1)
    bins = digitize_with_missing(values, edges)
    bc = bin_counts(bins, n_bins=n_real_bins, y=y)

    total_pos = int(y.sum())
    total_neg = int(y.size - total_pos)

    woes = []
    for i in range(n_real_bins):
        woes.append(_woe_for_bin(int(bc["pos"][i]),
                                 int(bc["n"][i] - bc["pos"][i]),
                                 total_pos, total_neg))

    # Missing bin
    miss_woe = None
    if missing_as_bin and bc["n_missing"] > 0:
        miss_woe = _woe_for_bin(int(bc["pos_missing"]),
                                int(bc["n_missing"] - bc["pos_missing"]),
                                total_pos, total_neg)

    # IV = Σ (p_i − n_i) · WOE_i
    iv = 0.0
    for i in range(n_real_bins):
        pi = (int(bc["pos"][i]) + _EPS) / (total_pos + _EPS)
        ni = (int(bc["n"][i] - bc["pos"][i]) + _EPS) / (total_neg + _EPS)
        iv += (pi - ni) * woes[i]
    if miss_woe is not None:
        pi = (int(bc["pos_missing"]) + _EPS) / (total_pos + _EPS)
        ni = (int(bc["n_missing"] - bc["pos_missing"]) + _EPS) / (total_neg + _EPS)
        iv += (pi - ni) * miss_woe

    return BinSpec(
        feature=feature,
        edges=edges.astype(float).tolist(),
        bin_counts=[int(x) for x in bc["n"]],
        pos_counts=[int(x) for x in bc["pos"]],
        woe_values=[float(x) for x in woes],
        missing_n=int(bc["n_missing"]),
        missing_pos=int(bc["pos_missing"]),
        missing_woe=float(miss_woe) if miss_woe is not None else None,
        iv=float(iv),
        monotonic=_is_monotonic(woes),
    )


def compute_iv_table(chunk_iter, spec_map, y_series, feature_names,
                     cfg, get_spec_fn):
    """Iterate chunks, compute BinSpec for each feature, return:
       - DataFrame[feature, iv, n_bins, monotonic, missing_n, missing_woe]
       - Dict[feature, BinSpec] for plotting / WOE encoding later.

    chunk_iter yields (df_chunk, block_features) from io.chunked_reader.
    y_series is the full-data label column as pandas Series aligned to the CSV.
    """
    y = y_series.values
    n_bins_cfg = int(cfg["analysis"].get("n_bins", 10))
    strategy = str(cfg["analysis"].get("binning", "equal_freq"))

    bin_specs = {}
    rows = []
    seen = set()
    for df_chunk, block in chunk_iter:
        if len(df_chunk) != y.shape[0]:
            raise ValueError(
                "chunk rows ({0}) != y rows ({1}); the CSV must be a single "
                "consistent file for chunked analysis.".format(len(df_chunk), y.shape[0]))
        for feat in block:
            if feat in seen:
                continue
            seen.add(feat)
            spec = get_spec_fn(spec_map, feat)
            from wdm.preprocess.missing import to_nan_array
            arr, _ = to_nan_array(df_chunk[feat], spec)
            bs = woe_bin(arr, y, feat,
                         n_bins=n_bins_cfg, strategy=strategy,
                         missing_as_bin=bool(spec.treat_as_missing_in_woe) or True)
            bin_specs[feat] = bs
            rows.append({
                "feature": feat,
                "iv": bs.iv,
                "n_bins": len(bs.bin_counts),
                "monotonic": bool(bs.monotonic),
                "missing_n": bs.missing_n,
                "missing_woe": bs.missing_woe,
            })
    df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return df, bin_specs
