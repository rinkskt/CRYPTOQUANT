import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional, Dict
from .preprocess import align_series

# Local imports to avoid circular dependencies
def compute_spread(series1: pd.Series, series2: pd.Series, beta: float) -> pd.Series:
    """Local copy to avoid circular import"""
    if len(series1) != len(series2) or beta is None or pd.isna(beta):
        return pd.Series(dtype=float)
    combined = pd.concat([series1, series2], axis=1).dropna()
    if len(combined) < 2:
        return pd.Series(dtype=float)
    s1 = combined.iloc[:, 0]
    s2 = combined.iloc[:, 1]
    spread = s1 - beta * s2
    return spread

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Local copy to avoid circular import"""
    if series.empty or len(series) < window:
        return pd.Series(dtype=float)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore_series = (series - rolling_mean) / rolling_std
    return zscore_series

def estimate_half_life(spread_series: pd.Series) -> float:
    """Local copy to avoid circular import"""
    if spread_series.empty or len(spread_series) < 30:
        return np.nan
    spread_clean = spread_series.dropna()
    if len(spread_clean) < 30:
        return np.nan
    spread_lag = spread_clean.shift(1).dropna()
    spread_clean = spread_clean.iloc[1:]
    if len(spread_clean) < 10:
        return np.nan
    try:
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_clean, X).fit()
        phi = model.params.iloc[1]
        if abs(phi) >= 1:
            return np.nan
        half_life = -np.log(0.5) / np.log(abs(phi))
        return half_life
    except Exception as e:
        return np.nan

def test_cointegration(series1: pd.Series, series2: pd.Series,
                      method: str = 'engle-granger') -> Dict:
    """
    Test cointegration between two price series using Engle-Granger method.

    Args:
        series1: First price series.
        series2: Second price series.
        method: Cointegration test method ('engle-granger' or 'johansen').

    Returns:
        Dict: Results with keys 'cointegrated', 'p_value', 'beta', 'spread', 'zscore', 'half_life'
    """
    if len(series1) != len(series2) or len(series1) < 30:
        return {
            'cointegrated': False,
            'p_value': np.nan,
            'beta': np.nan,
            'spread': pd.Series(dtype=float),
            'zscore': pd.Series(dtype=float),
            'half_life': np.nan
        }

    # Remove NaN values
    combined = pd.concat([series1, series2], axis=1).dropna()
    if len(combined) < 30:
        return {
            'cointegrated': False,
            'p_value': np.nan,
            'beta': np.nan,
            'spread': pd.Series(dtype=float),
            'zscore': pd.Series(dtype=float),
            'half_life': np.nan
        }

    s1 = combined.iloc[:, 0]
    s2 = combined.iloc[:, 1]

    if method == 'engle-granger':
        return _engle_granger_test(s1, s2)
    elif method == 'johansen':
        return _johansen_test(s1, s2)
    else:
        raise ValueError("Method must be 'engle-granger' or 'johansen'")

def _engle_granger_test(s1: pd.Series, s2: pd.Series) -> Dict:
    """
    Perform Engle-Granger cointegration test.

    Returns:
        Dict: Test results
    """
    # Step 1: Estimate beta using OLS: s1 = alpha + beta * s2 + error
    X = sm.add_constant(s2)
    model = sm.OLS(s1, X).fit()
    beta = model.params.iloc[1]  # coefficient for s2
    alpha = model.params.iloc[0]  # intercept

    # Step 2: Compute spread (residuals)
    spread = s1 - (alpha + beta * s2)

    # Step 3: Test if spread is stationary using ADF test
    try:
        adf_result = adfuller(spread, maxlag=None, autolag='AIC')
        t_stat = adf_result[0]
        p_value = adf_result[1]
        cointegrated = p_value < 0.05  # 5% significance level
    except:
        t_stat = np.nan
        p_value = np.nan
        cointegrated = False

    # Step 4: Compute z-score and half-life
    try:
        zscore_series = zscore(spread, window=30)
        half_life = estimate_half_life(spread)
    except:
        zscore_series = pd.Series(dtype=float)
        half_life = np.nan

    return {
        'cointegrated': cointegrated,
        'p_value': p_value,
        'beta': beta,
        'spread': spread,
        'zscore': zscore_series,
        'half_life': half_life,
        'latest_zscore': zscore_series.iloc[-1] if not zscore_series.empty else np.nan
    }

def _johansen_test(s1: pd.Series, s2: pd.Series) -> Dict:
    """
    Perform Johansen cointegration test (simplified for 2 variables).

    Returns:
        Dict: Test results
    """
    # For simplicity, fall back to Engle-Granger for now
    # Full Johansen implementation would require more complex eigenvalue analysis
    return _engle_granger_test(s1, s2)

def test_pair_cointegration(series1: pd.Series, series2: pd.Series,
                          method: str = 'engle-granger') -> Tuple[float, float, float, pd.Series]:
    """
    Legacy function for backward compatibility.
    """
    result = test_cointegration(series1, series2, method)
    return result['p_value'], result['beta'], result['spread'], result['zscore']

def is_cointegrated(t_stat: float, p_value: float, significance_level: float = 0.05) -> bool:
    """
    Determine if series are cointegrated based on test results.

    Args:
        t_stat: ADF test statistic.
        p_value: p-value from ADF test.
        significance_level: Significance level for hypothesis testing.

    Returns:
        bool: True if cointegrated, False otherwise.
    """
    if pd.isna(t_stat) or pd.isna(p_value):
        return False

    # For ADF test, we reject null hypothesis (non-stationary) if t_stat < critical_value
    # or p_value < significance_level
    return p_value < significance_level

def compute_cointegration_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cointegration test results for all pairs in the dataset.

    Args:
        prices_df: DataFrame with timestamps as index and assets as columns.

    Returns:
        pd.DataFrame: Results with columns ['asset1', 'asset2', 't_stat', 'p_value', 'beta', 'cointegrated']
    """
    if prices_df.empty or len(prices_df.columns) < 2:
        return pd.DataFrame()

    assets = prices_df.columns
    results = []

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:  # Only test upper triangle
                series1 = prices_df[asset1].dropna()
                series2 = prices_df[asset2].dropna()

                if len(series1) == 0 or len(series2) == 0:
                    continue

                # Align series to common dates
                aligned = align_series({asset1: pd.DataFrame({'timestamp': series1.index, 'close': series1.values}),
                                       asset2: pd.DataFrame({'timestamp': series2.index, 'close': series2.values})})

                if aligned.empty or len(aligned.columns) < 2:
                    continue

                s1_aligned = aligned.iloc[:, 0]
                s2_aligned = aligned.iloc[:, 1]

                coint_result = test_cointegration(s1_aligned, s2_aligned)

                results.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'cointegrated': coint_result['cointegrated'],
                    'p_value': coint_result['p_value'],
                    'beta': coint_result['beta'],
                    'half_life': coint_result['half_life'],
                    'latest_zscore': coint_result['latest_zscore']
                })

    return pd.DataFrame(results)
