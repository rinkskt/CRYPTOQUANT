"""
Pairwise Cointegration Analysis Module

This module provides functions for analyzing cointegration relationships between pairs of crypto assets.
Cointegration helps identify pairs that tend to move together over time, useful for statistical arbitrage.

Functions:
- score_all_pairs: Analyze all pairs for cointegration relationships
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
import itertools
from typing import List, Dict, Any


def score_all_pairs(prices_df: pd.DataFrame, top_n: int = 50, min_length: int = 100) -> pd.DataFrame:
    """
    Analyze all asset pairs for cointegration relationships.

    Manual: This function tests every possible pair of crypto assets for cointegration using the Engle-Granger
    two-step method. Cointegrated pairs have a long-term relationship, meaning their price difference tends
    to revert to a mean. This is fundamental for pairs trading strategies. The function returns the top_n
    pairs by correlation, ranked by cointegration test p-value (lower is better).

    Parameters:
    - prices_df: DataFrame with assets as columns, timestamps as index
    - top_n: Number of top correlated pairs to test for cointegration (default: 50)
    - min_length: Minimum data points required for testing (default: 100)

    Returns:
    - DataFrame with columns: asset_x, asset_y, correlation, beta, t_stat, p_value, half_life, latest_zscore
    """
    if prices_df.empty or len(prices_df.columns) < 2:
        return pd.DataFrame()

    # Ensure we have enough data
    if len(prices_df) < min_length:
        return pd.DataFrame()

    assets = list(prices_df.columns)
    results = []

    # Calculate pairwise correlations
    corr_matrix = prices_df.corr()

    # Get top_n pairs by absolute correlation
    pairs = []
    for i, j in itertools.combinations(range(len(assets)), 2):
        asset_x, asset_y = assets[i], assets[j]
        correlation = corr_matrix.loc[asset_x, asset_y]
        pairs.append((asset_x, asset_y, abs(correlation), correlation))

    # Sort by absolute correlation (highest first) and take top_n
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:top_n]

    for asset_x, asset_y, abs_corr, correlation in top_pairs:
        try:
            # Get price series
            x_prices = prices_df[asset_x].dropna()
            y_prices = prices_df[asset_y].dropna()

            # Align the series
            common_index = x_prices.index.intersection(y_prices.index)
            if len(common_index) < min_length:
                continue

            x_aligned = x_prices.loc[common_index]
            y_aligned = y_prices.loc[common_index]

            # Run cointegration test
            coint_result = coint(x_aligned, y_aligned)
            t_stat, p_value, crit_values = coint_result

            # Estimate beta using OLS (Y = beta * X + error)
            model = OLS(y_aligned, x_aligned).fit()
            beta = model.params.iloc[0]

            # Calculate spread and z-score
            spread = y_aligned - beta * x_aligned
            spread_mean = spread.mean()
            spread_std = spread.std()

            if spread_std > 0:
                zscore = (spread - spread_mean) / spread_std
                latest_zscore = zscore.iloc[-1]

                # Estimate half-life of mean reversion
                half_life = _estimate_half_life(spread)
            else:
                latest_zscore = 0
                half_life = np.inf

            results.append({
                'asset_x': asset_x,
                'asset_y': asset_y,
                'correlation': correlation,
                'beta': beta,
                't_stat': t_stat,
                'p_value': p_value,
                'half_life': half_life,
                'latest_zscore': latest_zscore,
                'cointegrated': p_value < 0.05  # 5% significance level
            })

        except Exception as e:
            # Skip pairs that fail testing
            continue

    # Convert to DataFrame and sort by p-value
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('p_value')

    return results_df


def _estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion for a spread series.

    Manual: Half-life measures how quickly a spread reverts to its mean. Lower half-life means faster
    mean reversion, which is preferable for trading. Calculated using AR(1) regression on the spread.

    Parameters:
    - spread: Series of spread values

    Returns:
    - Half-life in periods (float)
    """
    try:
        # AR(1) regression: spread_t = phi * spread_{t-1} + error
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align the series
        common_index = spread_lag.index.intersection(spread_diff.index)
        if len(common_index) < 10:  # Need minimum data
            return np.inf

        y = spread_diff.loc[common_index]
        X = spread_lag.loc[common_index]

        model = OLS(y, X).fit()
        phi = model.params.iloc[0]

        # Half-life = -ln(2) / ln(phi) for |phi| < 1
        if abs(phi) >= 1:
            return np.inf

        half_life = -np.log(2) / np.log(abs(phi))
        return max(0, half_life)  # Ensure non-negative

    except:
        return np.inf
