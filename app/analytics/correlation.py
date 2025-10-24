import pandas as pd
import numpy as np
from typing import Optional, Tuple
from .preprocess import align_series, compute_log_returns

def correlation_matrix(prices_df: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
    """
    Compute correlation matrix for price series.

    Args:
        prices_df: DataFrame with timestamps as index and assets as columns.
        window: Rolling window size for correlation. If None, compute static correlation.

    Returns:
        pd.DataFrame: Correlation matrix. If window is specified, returns a DataFrame
                     with MultiIndex (timestamp, asset1, asset2).
    """
    if prices_df.empty or len(prices_df.columns) < 2:
        return pd.DataFrame()

    if window is None:
        # Static correlation matrix
        returns_df = compute_log_returns(prices_df)
        if returns_df.empty:
            return pd.DataFrame()

        corr_matrix = returns_df.corr()
        return corr_matrix
    else:
        # Rolling correlation
        returns_df = compute_log_returns(prices_df)
        if returns_df.empty:
            return pd.DataFrame()

        # Compute rolling correlations for all pairs
        assets = returns_df.columns
        rolling_corrs = []

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:  # Only compute upper triangle to avoid duplicates
                    rolling_corr = returns_df[asset1].rolling(window=window).corr(returns_df[asset2])
                    rolling_corr = rolling_corr.dropna()

                    for timestamp, corr_value in rolling_corr.items():
                        rolling_corrs.append({
                            'timestamp': timestamp,
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': corr_value
                        })

        if not rolling_corrs:
            return pd.DataFrame()

        result_df = pd.DataFrame(rolling_corrs)
        result_df = result_df.set_index(['timestamp', 'asset1', 'asset2'])
        return result_df

def compute_correlation_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute static correlation matrix for price series.

    Args:
        prices_df: DataFrame with timestamps as index and assets as columns.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return correlation_matrix(prices_df, window=None)

def get_top_correlated_pairs(corr_matrix: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Get top N most correlated pairs from correlation matrix.

    Args:
        corr_matrix: Correlation matrix from correlation_matrix function.
        top_n: Number of top pairs to return.

    Returns:
        pd.DataFrame: DataFrame with columns ['asset1', 'asset2', 'correlation']
    """
    if corr_matrix.empty:
        return pd.DataFrame()

    # Get upper triangle of correlation matrix
    corr_pairs = []
    assets = corr_matrix.columns

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:  # Upper triangle
                corr_value = corr_matrix.loc[asset1, asset2]
                if not pd.isna(corr_value):
                    corr_pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': corr_value
                    })

    if not corr_pairs:
        return pd.DataFrame()

    pairs_df = pd.DataFrame(corr_pairs)
    pairs_df = pairs_df.sort_values('correlation', ascending=False)
    return pairs_df.head(top_n)

def correlation_heatmap_data(corr_matrix: pd.DataFrame) -> dict:
    """
    Prepare correlation matrix data for heatmap visualization.

    Args:
        corr_matrix: Correlation matrix.

    Returns:
        dict: Dictionary with 'matrix', 'assets', and 'values' for plotting.
    """
    if corr_matrix.empty:
        return {'matrix': [], 'assets': [], 'values': []}

    assets = list(corr_matrix.columns)
    matrix = corr_matrix.values.tolist()

    return {
        'matrix': matrix,
        'assets': assets,
        'values': corr_matrix.values.flatten().tolist()
    }
