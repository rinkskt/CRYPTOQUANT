import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Optional
from .cointegration import test_cointegration

def calculate_spread(series1: pd.Series, series2: pd.Series, beta: float) -> pd.Series:
    """
    Compute the spread between two cointegrated series.

    Args:
        series1: First price series.
        series2: Second price series.
        beta: Cointegration coefficient (from Engle-Granger test).

    Returns:
        pd.Series: Spread series (residuals).
    """
    if len(series1) != len(series2) or beta is None:
        return pd.Series(dtype=float)

    # Handle beta being a Series or scalar
    if isinstance(beta, pd.Series):
        if beta.empty or beta.isna().all():
            return pd.Series(dtype=float)
        beta_value = beta.iloc[0] if len(beta) > 0 else np.nan
    else:
        beta_value = beta

    if pd.isna(beta_value):
        return pd.Series(dtype=float)

    # Align series
    combined = pd.concat([series1, series2], axis=1).dropna()
    if len(combined) < 2:
        return pd.Series(dtype=float)

    s1 = combined.iloc[:, 0]
    s2 = combined.iloc[:, 1]

    # Compute spread: s1 - beta * s2
    spread = s1 - beta * s2

    return spread

def compute_spread(series1: pd.Series, series2: pd.Series, beta: float) -> pd.Series:
    """
    Compute the spread between two cointegrated series.

    Args:
        series1: First price series.
        series2: Second price series.
        beta: Cointegration coefficient (from Engle-Granger test).

    Returns:
        pd.Series: Spread series (residuals).
    """
    return calculate_spread(series1, series2, beta)

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute z-score (standardized score) for a time series.

    Args:
        series: Time series data.
        window: Rolling window size for mean and std calculation.

    Returns:
        pd.Series: Z-score series.
    """
    if series.empty or len(series) < window:
        return pd.Series(dtype=float)

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    zscore_series = (series - rolling_mean) / rolling_std

    return zscore_series

def estimate_half_life(spread_series: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion for a spread series using AR(1) model.

    Args:
        spread_series: Spread time series.

    Returns:
        float: Half-life in periods. Returns np.nan if estimation fails.
    """
    if spread_series.empty or len(spread_series) < 30:
        return np.nan

    # Remove NaN values
    spread_clean = spread_series.dropna()
    if len(spread_clean) < 30:
        return np.nan

    # Compute lagged spread: spread_{t-1}
    spread_lag = spread_clean.shift(1).dropna()
    spread_clean = spread_clean.iloc[1:]  # Align with lagged series

    if len(spread_clean) < 10:
        return np.nan

    try:
        # Fit AR(1) model: spread_t = phi * spread_{t-1} + error
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_clean, X).fit()
        phi = model.params.iloc[1]  # AR(1) coefficient

        # Half-life formula: ln(0.5) / ln(phi)
        if abs(phi) >= 1:
            return np.nan  # Not mean-reverting

        half_life = -np.log(0.5) / np.log(abs(phi))
        return half_life

    except Exception as e:
        print(f"Error estimating half-life: {e}")
        return np.nan

def compute_spread_statistics(spread_series: pd.Series, zscore_window: int = 20) -> dict:
    """
    Compute comprehensive statistics for a spread series.

    Args:
        spread_series: Spread time series.
        zscore_window: Window for z-score calculation.

    Returns:
        dict: Dictionary with spread statistics.
    """
    if spread_series.empty:
        return {}

    spread_clean = spread_series.dropna()

    stats = {
        'mean': spread_clean.mean(),
        'std': spread_clean.std(),
        'min': spread_clean.min(),
        'max': spread_clean.max(),
        'current_value': spread_clean.iloc[-1] if len(spread_clean) > 0 else np.nan,
        'half_life': estimate_half_life(spread_clean),
        'adf_t_stat': np.nan,
        'adf_p_value': np.nan
    }

    # Compute z-score
    zscore_series = zscore(spread_clean, window=zscore_window)
    if not zscore_series.empty:
        stats['current_zscore'] = zscore_series.iloc[-1]
        stats['zscore_mean'] = zscore_series.mean()
        stats['zscore_std'] = zscore_series.std()

    # ADF test for stationarity
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(spread_clean, maxlag=None, autolag='AIC')
        stats['adf_t_stat'] = adf_result[0]
        stats['adf_p_value'] = adf_result[1]
    except:
        pass

    return stats

def find_entry_exit_signals(zscore_series: pd.Series,
                          entry_threshold: float = 2.0,
                          exit_threshold: float = 0.5) -> pd.DataFrame:
    """
    Find entry and exit signals based on z-score thresholds.

    Args:
        zscore_series: Z-score time series.
        entry_threshold: Z-score threshold for entry signals.
        exit_threshold: Z-score threshold for exit signals.

    Returns:
        pd.DataFrame: Signals with columns ['signal', 'zscore']
    """
    if zscore_series.empty:
        return pd.DataFrame()

    signals = []

    for i in range(len(zscore_series)):
        zscore_val = zscore_series.iloc[i]

        if abs(zscore_val) >= entry_threshold:
            signal = 'SHORT' if zscore_val > 0 else 'LONG'
            signals.append({
                'timestamp': zscore_series.index[i],
                'signal': signal,
                'zscore': zscore_val
            })
        elif abs(zscore_val) <= exit_threshold:
            signals.append({
                'timestamp': zscore_series.index[i],
                'signal': 'EXIT',
                'zscore': zscore_val
            })

    return pd.DataFrame(signals)
