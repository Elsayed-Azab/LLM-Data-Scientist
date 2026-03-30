"""Weighted survey statistics — ground-truth implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Compute the weighted mean, dropping NaN pairs."""
    mask = series.notna() & weights.notna()
    return np.average(series[mask], weights=weights[mask])


def weighted_std(series: pd.Series, weights: pd.Series) -> float:
    """Compute weighted standard deviation (reliability weights)."""
    mask = series.notna() & weights.notna()
    s, w = series[mask].values, weights[mask].values
    avg = np.average(s, weights=w)
    variance = np.average((s - avg) ** 2, weights=w)
    return float(np.sqrt(variance))


def weighted_median(series: pd.Series, weights: pd.Series) -> float:
    """Compute the weighted median."""
    mask = series.notna() & weights.notna()
    s, w = series[mask].values, weights[mask].values
    order = np.argsort(s)
    s, w = s[order], w[order]
    cumw = np.cumsum(w)
    cutoff = cumw[-1] / 2.0
    return float(s[np.searchsorted(cumw, cutoff)])


def weighted_frequency(
    series: pd.Series,
    weights: pd.Series,
    normalize: bool = True,
) -> pd.Series:
    """Weighted frequency table for a categorical/discrete variable."""
    mask = series.notna() & weights.notna()
    df = pd.DataFrame({"val": series[mask], "w": weights[mask]})
    freq = df.groupby("val")["w"].sum().sort_index()
    if normalize:
        freq = freq / freq.sum()
    return freq


def weighted_crosstab(
    row_var: pd.Series,
    col_var: pd.Series,
    weights: pd.Series,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Weighted cross-tabulation.

    Args:
        normalize: None for raw counts, 'index' for row-pct, 'columns' for col-pct, 'all' for total-pct.
    """
    mask = row_var.notna() & col_var.notna() & weights.notna()
    df = pd.DataFrame({"row": row_var[mask], "col": col_var[mask], "w": weights[mask]})
    ct = df.pivot_table(values="w", index="row", columns="col", aggfunc="sum", fill_value=0)

    if normalize == "index":
        ct = ct.div(ct.sum(axis=1), axis=0)
    elif normalize == "columns":
        ct = ct.div(ct.sum(axis=0), axis=1)
    elif normalize == "all":
        ct = ct / ct.values.sum()

    return ct


def descriptive_stats(
    df: pd.DataFrame,
    columns: list[str],
    weight_column: str,
) -> dict[str, dict]:
    """Compute weighted descriptive statistics for multiple columns."""
    weights = df[weight_column]
    results = {}
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            results[col] = {
                "type": "categorical",
                "frequencies": weighted_frequency(df[col], weights).to_dict(),
            }
        else:
            results[col] = {
                "type": "numeric",
                "mean": weighted_mean(df[col], weights),
                "std": weighted_std(df[col], weights),
                "median": weighted_median(df[col], weights),
                "n_valid": int(df[col].notna().sum()),
                "n_missing": int(df[col].isna().sum()),
            }
    return results
