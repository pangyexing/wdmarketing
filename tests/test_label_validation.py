"""validate_binary_label — a stray 2 / -1 / NaN in the label must abort the
run instead of being silently coerced into a negative."""
import numpy as np
import pandas as pd
import pytest

from wdm.utils.labels import validate_binary_label


def test_clean_binary_passes():
    validate_binary_label(pd.Series([0, 1, 1, 0]), "y")
    validate_binary_label(pd.Series([0.0, 1.0]), "y")
    validate_binary_label(pd.Series([True, False]), "y")


def test_third_class_rejected():
    with pytest.raises(ValueError) as ei:
        validate_binary_label(pd.Series([0, 1, 2, 2, 1]), "y")
    assert "2" in str(ei.value) and "'y'" in str(ei.value)


def test_negative_class_rejected():
    with pytest.raises(ValueError):
        validate_binary_label(pd.Series([0, 1, -1]), "credit_1v1")


def test_nan_rejected():
    with pytest.raises(ValueError) as ei:
        validate_binary_label(pd.Series([0, 1, np.nan]), "y")
    assert "NaN" in str(ei.value)


def test_non_numeric_rejected():
    with pytest.raises(ValueError):
        validate_binary_label(pd.Series(["yes", "no"]), "y")
