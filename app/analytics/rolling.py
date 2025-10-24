"""
Rolling Correlation Analytics Module

This module provides functions for computing rolling correlations between crypto assets.
Rolling correlations help identify changing relationships between assets over time.

Functions:
- rolling_correlation: Compute rolling correlation matrix over time windows
- latest_rolling_corr_matrix: Get the most recent correlation matrix
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def rolling_correlation(prices_df: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
    """
    Compute rolling correlation matrix for all asset pairs over time.

    Manual: Rolling correlations show how the relationships between crypto assets change over time.
    A 30-day window means each correlation coefficient represents the relationship over the past 30 days.
    This helps identify periods of high/low correlation, useful for portfolio diversification and
    pair trading strategies. Low correlations suggest diversification opportunities.

    Parameters:
    - prices_df: DataFrame with assets as columns, timestamps as index
    - window: Rolling window size in days (default: 30)

    Returns:
    - Dict containing:
      - 'correlations': Dict of DataFrames (one per timestamp)
      - 'timestamps': List of timestamps
      - 'assets': List of asset names
    """
    if prices_df.empty:
        return {'correlations': {}, 'timestamps': [], 'assets': []}

    # Ensure we have enough data
    if len(prices_df) < window:
        return {'correlations': {}, 'timestamps': [], 'assets': list(prices_df.columns)}

    # Calculate percentage returns for correlation
    returns_df = prices_df.pct_change().dropna()

    if len(returns_df) < window:
        return {'correlations': {}, 'timestamps': [], 'assets': list(prices_df.columns)}

    correlations = {}
    timestamps = []

    # Roll through the data
    for i in range(window, len(returns_df) + 1):
        window_data = returns_df.iloc[i-window:i]
        corr_matrix = window_data.corr()
        timestamp = returns_df.index[i-1]  # Use the last date in the window

        correlations[str(timestamp)] = corr_matrix
        timestamps.append(timestamp)

    return {
        'correlations': correlations,
        'timestamps': timestamps,
        'assets': list(prices_df.columns)
    }


def latest_rolling_corr_matrix(prices_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Get the most recent rolling correlation matrix.

    Manual: This function returns the latest correlation matrix computed over the most recent
    time window. Use this to see current relationships between assets for decision making.

    Parameters:
    - prices_df: DataFrame with assets as columns, timestamps as index
    - window: Rolling window size in days (default: 30)

    Returns:
    - DataFrame: Latest correlation matrix
    """
    result = rolling_correlation(prices_df, window)

    if not result['correlations']:
        # Return empty correlation matrix if no data
        assets = list(prices_df.columns) if not prices_df.empty else []
        return pd.DataFrame(index=assets, columns=assets)

    # Get the latest correlation matrix
    latest_timestamp = max(result['correlations'].keys())
    return result['correlations'][latest_timestamp]
