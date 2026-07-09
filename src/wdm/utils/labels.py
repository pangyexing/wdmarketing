"""Label sanity checks shared by Stage-1 analysis and Stage-2 training."""
import pandas as pd


def validate_binary_label(series, label_col):
    """Fail fast unless every value is 0 or 1 (no NaN, no other values).

    Without this a stray 2 / -1 / NaN is silently coerced downstream and
    counted as a negative in scale_pos_weight and every y==1 metric —
    producing a quietly wrong model. Rows that should not train must be
    dropped via data.exclude_rows (this check runs on the surviving rows).
    """
    num = pd.to_numeric(pd.Series(series), errors="coerce")
    counts = num.value_counts(dropna=False)
    bad = counts[~counts.index.isin([0.0, 1.0])]
    if len(bad):
        pairs = ", ".join(
            "{0}: {1}".format("NaN" if pd.isna(k) else k, int(v))
            for k, v in bad.items())
        raise ValueError(
            "label column {0!r} must be binary 0/1 with no NaN; offending "
            "values (value: rows): {1}. Drop such rows via data.exclude_rows "
            "or derive a binary label upstream.".format(label_col, pairs))
