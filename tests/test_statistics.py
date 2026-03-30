"""Tests for the weighted statistics module."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistics import (
    weighted_mean,
    weighted_std,
    weighted_median,
    weighted_frequency,
    weighted_crosstab,
    descriptive_stats,
)


def test_weighted_mean_basic():
    s = pd.Series([1.0, 2.0, 3.0])
    w = pd.Series([1.0, 1.0, 1.0])
    assert weighted_mean(s, w) == pytest.approx(2.0)


def test_weighted_mean_with_weights():
    s = pd.Series([10.0, 20.0])
    w = pd.Series([3.0, 1.0])
    assert weighted_mean(s, w) == pytest.approx(12.5)


def test_weighted_mean_with_nan():
    s = pd.Series([1.0, np.nan, 3.0])
    w = pd.Series([1.0, 1.0, 1.0])
    assert weighted_mean(s, w) == pytest.approx(2.0)


def test_weighted_std():
    s = pd.Series([2.0, 2.0, 2.0])
    w = pd.Series([1.0, 1.0, 1.0])
    assert weighted_std(s, w) == pytest.approx(0.0)


def test_weighted_std_nonzero():
    s = pd.Series([1.0, 3.0])
    w = pd.Series([1.0, 1.0])
    assert weighted_std(s, w) == pytest.approx(1.0)


def test_weighted_median():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    w = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    result = weighted_median(s, w)
    assert result == pytest.approx(3.0)


def test_weighted_frequency_normalized():
    s = pd.Series([1, 1, 2, 2, 2])
    w = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    freq = weighted_frequency(s, w, normalize=True)
    assert freq[1] == pytest.approx(0.4)
    assert freq[2] == pytest.approx(0.6)


def test_weighted_frequency_raw():
    s = pd.Series([1, 2, 2])
    w = pd.Series([2.0, 1.0, 3.0])
    freq = weighted_frequency(s, w, normalize=False)
    assert freq[1] == pytest.approx(2.0)
    assert freq[2] == pytest.approx(4.0)


def test_weighted_crosstab():
    row = pd.Series(["A", "A", "B", "B"])
    col = pd.Series(["X", "Y", "X", "Y"])
    w = pd.Series([1.0, 2.0, 3.0, 4.0])
    ct = weighted_crosstab(row, col, w)
    assert ct.loc["A", "X"] == pytest.approx(1.0)
    assert ct.loc["B", "Y"] == pytest.approx(4.0)


def test_weighted_crosstab_normalized():
    row = pd.Series(["A", "A", "B", "B"])
    col = pd.Series(["X", "Y", "X", "Y"])
    w = pd.Series([1.0, 1.0, 1.0, 1.0])
    ct = weighted_crosstab(row, col, w, normalize="index")
    assert ct.loc["A", "X"] == pytest.approx(0.5)
    assert ct.loc["A", "Y"] == pytest.approx(0.5)


def test_descriptive_stats_numeric():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]})
    result = descriptive_stats(df, ["val"], "w")
    assert result["val"]["type"] == "numeric"
    assert result["val"]["mean"] == pytest.approx(2.0)
    assert result["val"]["n_valid"] == 3
