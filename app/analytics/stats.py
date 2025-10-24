"""
Statistical Analysis Utilities Module

This module provides statistical functions for time series analysis, particularly for
mean-reversion and spread analysis in pairs trading.

Functions:
- compute_zscore: Calculate rolling z-score for a time series
- estimate_half_life: Estimate mean-reversion half-life using AR(1) regression
"""

import pandas as pd
import numpy as np
from statsmodels.api import OLS
from typing import Union


def compute_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score for a time series.

    Manual: Z-score measures how many standard deviations a value is from the mean.
    In pairs trading, z-score > 2 indicates the spread is significantly above its mean
    (potential short opportunity), while z-score < -2 indicates significantly below mean
    (potential long opportunity). The rolling window ensures the mean and std adapt to recent data.

    Parameters:
    - series: Time series to compute z-score for
    - window: Rolling window size for mean/std calculation (default: 60)

    Returns:
    - Series of z-score values
    """
    if series.empty or len(series) < window:
        return pd.Series(dtype=float)

    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (series - rolling_mean) / rolling_std

    return zscore.fillna(0)  # Fill NaN with 0 for early periods


def estimate_half_life(series: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion for a time series using AR(1) regression.

    Manual: Half-life indicates how quickly a series reverts to its mean. For pairs trading spreads,
    shorter half-life means faster mean reversion (better for trading). Calculated by fitting
    an AR(1) model: ΔS_t = φ * S_{t-1} + ε, then half_life = -ln(2) / ln(|φ|).

    Parameters:
    - series: Time series (e.g., spread) to analyze

    Returns:
    - Half-life in periods (float). Returns inf if no mean reversion detected.
    """
    if series.empty or len(series) < 10:
        return np.inf

    try:
        # Create lagged series for AR(1) regression
        # ΔS_t = φ * S_{t-1} + ε
        spread_lag = series.shift(1).dropna()
        spread_diff = series.diff().dropna()

        # Align the series
        common_index = spread_lag.index.intersection(spread_diff.index)
        if len(common_index) < 5:
            return np.inf

        y = spread_diff.loc[common_index]  # ΔS_t
        X = spread_lag.loc[common_index]   # S_{t-1}

        # Fit AR(1) model
        model = OLS(y, X).fit()
        phi = model.params.iloc[0]

        # Check for stationarity (|φ| < 1 for mean reversion)
        if abs(phi) >= 1:
            return np.inf  # No mean reversion

        # Half-life = -ln(2) / ln(|φ|)
        half_life = -np.log(2) / np.log(abs(phi))

        # Ensure positive half-life
        return max(0.0, half_life)

    except Exception:
        # Return infinity if estimation fails
        return np.inf


def compute_rolling_stats(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling statistics for a time series.

    Parameters:
    - series: Time series to analyze
    - window: Rolling window size

    Returns:
    - DataFrame with rolling mean, std, min, max, and z-score
    """
    if series.empty or len(series) < window:
        return pd.DataFrame()

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()

    # Z-score
    zscore = (series - rolling_mean) / rolling_std

    return pd.DataFrame({
        'mean': rolling_mean,
        'std': rolling_std,
        'min': rolling_min,
        'max': rolling_max,
        'zscore': zscore
    })
