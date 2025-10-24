import logging
import pandas as pd
from datetime import datetime, timedelta
from app.db.engine import engine
from app.analytics.preprocess import align_series
from app.analytics.correlation import correlation_matrix, get_top_correlated_pairs
from app.analytics.cointegration import compute_cointegration_matrix
from app.analytics.spread import compute_spread, compute_spread_statistics
from app.analytics.persist import (save_correlation_matrix, save_cointegration_results,
                                  save_spread_metrics, get_asset_id_map)
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_price_data(days: int = 365) -> pd.DataFrame:
    """
    Load recent price data from database.

    Args:
        days: Number of days of data to load.

    Returns:
        pd.DataFrame: Price data with timestamps as index and assets as columns.
    """
    cutoff_date = datetime.now() - timedelta(days=days)

    query = """
    SELECT a.symbol, o.timestamp, o.close
    FROM ohlcv o
    JOIN assets a ON o.asset_id = a.id
    WHERE a.active = 1 AND o.timestamp >= :cutoff_date
    ORDER BY o.timestamp
    """

    with engine.connect() as conn:
        result = conn.execute(text(query), {'cutoff_date': cutoff_date})
        rows = result.fetchall()

    if not rows:
        logger.warning("No price data found")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['symbol', 'timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    # Remove duplicates based on symbol and timestamp, keep last
    df = df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

    # Pivot to get assets as columns
    price_df = df.pivot(index='timestamp', columns='symbol', values='close')
    price_df = price_df.dropna(how='all')  # Remove rows with all NaN

    return price_df

def run_correlation_analysis(price_df: pd.DataFrame, asset_id_map: dict,
                           timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Run correlation analysis and save results.

    Args:
        price_df: Price data DataFrame.
        asset_id_map: Asset symbol to ID mapping.
        timestamp: Timestamp for results.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    logger.info("Computing correlation matrix...")
    corr_matrix = correlation_matrix(price_df)

    if not corr_matrix.empty:
        save_correlation_matrix(corr_matrix, asset_id_map, timestamp)
        logger.info(f"Saved correlation matrix with {len(corr_matrix)} assets")

    return corr_matrix

def run_cointegration_analysis(price_df: pd.DataFrame, asset_id_map: dict,
                             timestamp: pd.Timestamp, top_n_pairs: int = 10) -> pd.DataFrame:
    """
    Run cointegration analysis on top correlated pairs.

    Args:
        price_df: Price data DataFrame.
        asset_id_map: Asset symbol to ID mapping.
        timestamp: Timestamp for results.
        top_n_pairs: Number of top correlated pairs to test.

    Returns:
        pd.DataFrame: Cointegration results.
    """
    logger.info("Running cointegration analysis...")

    # First get correlation matrix
    corr_matrix = correlation_matrix(price_df)
    if corr_matrix.empty:
        logger.warning("No correlation data available")
        return pd.DataFrame()

    # Get top correlated pairs
    top_pairs = get_top_correlated_pairs(corr_matrix, top_n=top_n_pairs)
    if top_pairs.empty:
        logger.warning("No correlated pairs found")
        return pd.DataFrame()

    logger.info(f"Testing cointegration for {len(top_pairs)} top pairs...")

    # Test cointegration for top pairs
    coint_results = []

    for _, pair in top_pairs.iterrows():
        asset1 = pair['asset1']
        asset2 = pair['asset2']

        if asset1 not in price_df.columns or asset2 not in price_df.columns:
            continue

        series1 = price_df[asset1].dropna()
        series2 = price_df[asset2].dropna()

        if len(series1) < 30 or len(series2) < 30:
            continue

        # Align series
        aligned = align_series({
            asset1: pd.DataFrame({'timestamp': series1.index, 'close': series1.values}),
            asset2: pd.DataFrame({'timestamp': series2.index, 'close': series2.values})
        })

        if aligned.empty or len(aligned.columns) < 2:
            continue

        s1_aligned = aligned.iloc[:, 0]
        s2_aligned = aligned.iloc[:, 1]

        # Test cointegration
        from app.analytics.cointegration import test_pair_cointegration, is_cointegrated
        t_stat, p_value, beta, spread = test_pair_cointegration(s1_aligned, s2_aligned)
        cointegrated = is_cointegrated(t_stat, p_value)

        coint_results.append({
            'asset1': asset1,
            'asset2': asset2,
            'correlation': pair['correlation'],
            't_stat': t_stat,
            'p_value': p_value,
            'beta': beta,
            'cointegrated': cointegrated
        })

    results_df = pd.DataFrame(coint_results)
    if not results_df.empty:
        save_cointegration_results(results_df, asset_id_map, timestamp)
        logger.info(f"Saved cointegration results for {len(results_df)} pairs")

    return results_df

def run_spread_analysis(price_df: pd.DataFrame, coint_results: pd.DataFrame,
                       asset_id_map: dict, timestamp: pd.Timestamp):
    """
    Run spread analysis for cointegrated pairs.

    Args:
        price_df: Price data DataFrame.
        coint_results: Cointegration test results.
        asset_id_map: Asset symbol to ID mapping.
        timestamp: Timestamp for results.
    """
    logger.info("Running spread analysis for cointegrated pairs...")

    cointegrated_pairs = coint_results[coint_results['cointegrated'] == True]

    for _, pair in cointegrated_pairs.iterrows():
        asset1 = pair['asset1']
        asset2 = pair['asset2']
        beta = pair['beta']

        if asset1 not in price_df.columns or asset2 not in price_df.columns:
            continue

        series1 = price_df[asset1].dropna()
        series2 = price_df[asset2].dropna()

        # Align series
        aligned = align_series({
            asset1: pd.DataFrame({'timestamp': series1.index, 'close': series1.values}),
            asset2: pd.DataFrame({'timestamp': series2.index, 'close': series2.values})
        })

        if aligned.empty or len(aligned.columns) < 2:
            continue

        s1_aligned = aligned.iloc[:, 0]
        s2_aligned = aligned.iloc[:, 1]

        # Compute spread
        spread = compute_spread(s1_aligned, s2_aligned, beta)
        if spread.empty:
            continue

        # Compute spread statistics
        spread_stats = compute_spread_statistics(spread)

        # Save spread metrics
        pair_key = f"{asset1}_{asset2}"
        asset1_id = asset_id_map.get(asset1)
        asset2_id = asset_id_map.get(asset2)

        if asset1_id is not None and asset2_id is not None:
            save_spread_metrics(asset1_id, asset2_id, timestamp, spread_stats, pair_key)
            logger.info(f"Saved spread metrics for {pair_key}")

def run_all(days: int = 365, top_n_pairs: int = 10):
    """
    Run complete analytics pipeline.

    Args:
        days: Number of days of data to analyze.
        top_n_pairs: Number of top correlated pairs to test for cointegration.
    """
    logger.info("Starting analytics pipeline...")

    try:
        # Get asset mapping
        asset_id_map = get_asset_id_map()
        if not asset_id_map:
            logger.error("No active assets found")
            return

        logger.info(f"Found {len(asset_id_map)} active assets")

        # Load price data
        price_df = load_price_data(days=days)
        if price_df.empty:
            logger.error("No price data available")
            return

        logger.info(f"Loaded price data with shape: {price_df.shape}")

        # Use latest timestamp for results
        timestamp = price_df.index[-1] if len(price_df) > 0 else pd.Timestamp.now()

        # Run correlation analysis
        corr_matrix = run_correlation_analysis(price_df, asset_id_map, timestamp)

        # Run cointegration analysis
        coint_results = run_cointegration_analysis(price_df, asset_id_map, timestamp, top_n_pairs)

        # Run spread analysis for cointegrated pairs
        if not coint_results.empty:
            run_spread_analysis(price_df, coint_results, asset_id_map, timestamp)

        logger.info("Analytics pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in analytics pipeline: {e}")
        raise

if __name__ == '__main__':
    run_all()
