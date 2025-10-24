import pandas as pd
import numpy as np
from typing import Dict, List

def align_series(df_dict: Dict[str, pd.DataFrame], freq: str = '1d') -> pd.DataFrame:
    """
    Align multiple time series DataFrames to a common frequency and index.

    Args:
        df_dict: Dictionary with asset names as keys and DataFrames as values.
                 Each DataFrame should have 'timestamp' and 'close' columns.
        freq: Frequency for resampling (e.g., '1d', '1h', '1w')

    Returns:
        pd.DataFrame: Aligned DataFrame with timestamps as index and assets as columns.
    """
    aligned_series = {}

    for asset, df in df_dict.items():
        # Ensure timestamp is datetime and set as index
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Resample to desired frequency, forward fill missing values
        series = df['close'].resample(freq).last().ffill()
        aligned_series[asset] = series

    # Combine all series into a single DataFrame
    result = pd.DataFrame(aligned_series)

    # Remove rows with all NaN values
    result = result.dropna(how='all')

    return result

def compute_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns for a DataFrame of price series.

    Args:
        prices_df: DataFrame with timestamps as index and assets as columns.

    Returns:
        pd.DataFrame: Log returns DataFrame.
    """
    if prices_df.empty:
        return pd.DataFrame()

    # Compute log returns: ln(P_t / P_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))

    # Remove the first row which will be NaN
    log_returns = log_returns.iloc[1:]

    return log_returns

def compute_simple_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns for a DataFrame of price series.

    Args:
        prices_df: DataFrame with timestamps as index and assets as columns.

    Returns:
        pd.DataFrame: Simple returns DataFrame.
    """
    if prices_df.empty:
        return pd.DataFrame()

    # Compute simple returns: (P_t - P_{t-1}) / P_{t-1}
    simple_returns = prices_df.pct_change()

    # Remove the first row which will be NaN
    simple_returns = simple_returns.iloc[1:]

    return simple_returns

def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing values in a DataFrame.

    Args:
        df: DataFrame with missing values.
        method: Method to use ('ffill', 'bfill', 'interpolate', etc.)

    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    if method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate()
    else:
        return df.fillna(df.mean())
