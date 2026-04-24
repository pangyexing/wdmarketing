"""Column-chunked CSV reader for memory-safe Stage-1 analysis.

Yields DataFrames of (chunk_size features) + always_cols (label/time/treatment).
Each yield loads the full N rows for that column subset — O(rows × chunk_size) memory.

Usage:
    for chunk in iter_column_chunks(path, features, always=['y'], chunk_size=50):
        # analyze chunk[feature_block + always]
        ...

Why column-chunk not row-chunk: every per-feature statistic we compute
(PSI/IV/WOE/Lift/missing_rate/correlation) needs the full row-count for a feature.
Column-chunking keeps each feature's full distribution in memory for one pass.
"""
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def iter_column_chunks(path,
                       features,
                       always=None,
                       chunk_size=50,
                       dtype=None,
                       desc=None):
    """Yield DataFrames containing always-cols + chunk_size feature cols.

    Args:
        path: CSV path.
        features: list of feature column names to rotate through.
        always: list of columns always read (label, time, treatment, ids).
        chunk_size: how many feature cols per chunk.
        dtype: optional dtype dict passed to pd.read_csv.
        desc: optional label; when provided, per-chunk progress is emitted
            at INFO so a long Stage-1 pass shows a heartbeat instead of
            going silent between start/end.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("data file not found: {0}".format(path))
    always = list(always or [])
    features = list(features)
    n_chunks = (len(features) + chunk_size - 1) // chunk_size
    log_fn = logger.info if desc else logger.debug
    prefix = "{0} ".format(desc) if desc else ""
    for i in range(n_chunks):
        block = features[i * chunk_size:(i + 1) * chunk_size]
        cols = always + block
        # dedupe preserving order in case of overlap
        seen = set()
        cols = [c for c in cols if not (c in seen or seen.add(c))]
        df = pd.read_csv(path, usecols=cols, dtype=dtype)
        # reorder so always comes first then block (pd.read_csv preserves CSV order)
        df = df[cols]
        log_fn("%schunk %d/%d: %d features, %d rows", prefix, i + 1, n_chunks,
               len(block), len(df))
        yield df, block


def read_full(path, columns=None, dtype=None):
    """Read the full table, optionally limited to the given columns.

    Stage 2 uses this after feature selection reduces to <=200 columns.
    """
    return pd.read_csv(path, usecols=columns, dtype=dtype)


def read_raw_rows(path, row_indices, columns=None):
    """Fetch specific row positions from CSV as raw strings-as-typed, for the
    exporter's validation_samples (preserves original feature values).
    """
    import numpy as np
    df = pd.read_csv(path, usecols=columns)
    return df.iloc[np.asarray(row_indices)].copy()
