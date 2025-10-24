import pandas as pd
from sqlalchemy import text
from app.db.engine import engine
from typing import Dict, List, Optional

def save_analytics_metric(asset_id: int, ts: pd.Timestamp, metric: str, value: float):
    """
    Save a single analytics metric to the database.

    Args:
        asset_id: Asset ID.
        ts: Timestamp.
        metric: Metric name (e.g., 'correlation', 'zscore', 'half_life').
        value: Metric value.
    """
    if pd.isna(value):
        return

    with engine.connect() as conn:
        # For SQLite, use INSERT OR REPLACE
        conn.execute(text("""
            INSERT OR REPLACE INTO analytics (asset_id, ts, metric, value)
            VALUES (:asset_id, :ts, :metric, :value)
        """), {
            'asset_id': asset_id,
            'ts': str(ts.date()) if isinstance(ts, pd.Timestamp) else str(ts),
            'metric': metric,
            'value': value
        })
        conn.commit()

def save_pair_metric(asset1_id: int, asset2_id: int, ts: pd.Timestamp,
                    metric: str, value: float, pair_key: Optional[str] = None):
    """
    Save a pair-wise analytics metric.

    Args:
        asset1_id: First asset ID.
        asset2_id: Second asset ID.
        ts: Timestamp.
        metric: Metric name.
        value: Metric value.
        pair_key: Optional pair identifier (e.g., 'BTC_ETH').
    """
    if pd.isna(value):
        return

    # Use asset1_id as the primary key, store pair info in metric name
    metric_name = f"{metric}_{pair_key}" if pair_key else metric

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO analytics (asset_id, ts, metric, value)
            VALUES (:asset_id, :ts, :metric, :value)
        """), {
            'asset_id': asset1_id,
            'ts': str(ts.date()) if isinstance(ts, pd.Timestamp) else str(ts),
            'metric': metric_name,
            'value': value
        })
        conn.commit()

def save_correlation_matrix(corr_matrix: pd.DataFrame, asset_id_map: Dict[str, int],
                           timestamp: pd.Timestamp):
    """
    Save correlation matrix to analytics table.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        asset_id_map: Mapping from asset symbols to IDs.
        timestamp: Timestamp for the correlation data.
    """
    if corr_matrix.empty:
        return

    assets = corr_matrix.columns

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:  # Upper triangle only
                asset1_id = asset_id_map.get(asset1)
                asset2_id = asset_id_map.get(asset2)

                if asset1_id is None or asset2_id is None:
                    continue

                corr_value = corr_matrix.loc[asset1, asset2]
                if not pd.isna(corr_value):
                    pair_key = f"{asset1}_{asset2}"
                    save_pair_metric(asset1_id, asset2_id, timestamp,
                                   'correlation', corr_value, pair_key)

def save_cointegration_results(results_df: pd.DataFrame, asset_id_map: Dict[str, int],
                             timestamp: pd.Timestamp):
    """
    Save cointegration test results to analytics table.

    Args:
        results_df: DataFrame with cointegration results.
        asset_id_map: Mapping from asset symbols to IDs.
        timestamp: Timestamp for the results.
    """
    if results_df.empty:
        return

    for _, row in results_df.iterrows():
        asset1_id = asset_id_map.get(row['asset1'])
        asset2_id = asset_id_map.get(row['asset2'])

        if asset1_id is None or asset2_id is None:
            continue

        pair_key = f"{row['asset1']}_{row['asset2']}"

        # Save t-statistic
        if not pd.isna(row['t_stat']):
            save_pair_metric(asset1_id, asset2_id, timestamp,
                           'coint_t_stat', row['t_stat'], pair_key)

        # Save p-value
        if not pd.isna(row['p_value']):
            save_pair_metric(asset1_id, asset2_id, timestamp,
                           'coint_p_value', row['p_value'], pair_key)

        # Save beta
        beta_value = row['beta']
        if isinstance(beta_value, pd.Series):
            if not beta_value.empty and not beta_value.isna().all():
                save_pair_metric(asset1_id, asset2_id, timestamp,
                               'coint_beta', float(beta_value.iloc[0]), pair_key)
        else:
            if not pd.isna(beta_value):
                save_pair_metric(asset1_id, asset2_id, timestamp,
                               'coint_beta', float(beta_value), pair_key)

        # Save cointegration flag
        save_pair_metric(asset1_id, asset2_id, timestamp,
                        'cointegrated', 1.0 if row['cointegrated'] else 0.0, pair_key)

def save_spread_metrics(asset1_id: int, asset2_id: int, timestamp: pd.Timestamp,
                       spread_stats: dict, pair_key: str):
    """
    Save spread-related metrics.

    Args:
        asset1_id: First asset ID.
        asset2_id: Second asset ID.
        timestamp: Timestamp.
        spread_stats: Dictionary with spread statistics.
        pair_key: Pair identifier.
    """
    metrics_to_save = [
        'half_life', 'current_zscore', 'adf_t_stat', 'adf_p_value'
    ]

    for metric in metrics_to_save:
        if metric in spread_stats and not pd.isna(spread_stats[metric]):
            save_pair_metric(asset1_id, asset2_id, timestamp,
                           f"spread_{metric}", spread_stats[metric], pair_key)

def get_asset_id_map() -> Dict[str, int]:
    """
    Get mapping from asset symbols to IDs.

    Returns:
        Dict[str, int]: Symbol to ID mapping.
    """
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, symbol FROM assets WHERE active = 1"))
        return {row[1]: row[0] for row in result.fetchall()}

def save_analytics_data(data: List[Dict]):
    """
    Save multiple analytics data points to the database.

    Args:
        data: List of dictionaries with keys 'asset_id', 'ts', 'metric', 'value'
    """
    if not data:
        return

    with engine.connect() as conn:
        for record in data:
            if pd.isna(record.get('value')):
                continue

            conn.execute(text("""
                INSERT OR REPLACE INTO analytics (asset_id, ts, metric, value)
                VALUES (:asset_id, :ts, :metric, :value)
            """), {
                'asset_id': record['asset_id'],
                'ts': str(record['ts']),
                'metric': record['metric'],
                'value': record['value']
            })
        conn.commit()

def get_analytics_data(asset_id: Optional[int] = None, metric: Optional[str] = None,
                        limit: int = 100) -> List[Dict]:
    """
    Retrieve latest analytics data from database.

    Args:
        asset_id: Filter by asset ID.
        metric: Filter by metric name.
        limit: Maximum number of records to return.

    Returns:
        List[Dict]: Analytics data as list of dictionaries.
    """
    query = "SELECT asset_id, ts, metric, value FROM analytics WHERE 1=1"
    params = {}

    if asset_id is not None:
        query += " AND asset_id = :asset_id"
        params['asset_id'] = asset_id

    if metric is not None:
        query += " AND metric LIKE :metric"
        params['metric'] = f"%{metric}%"

    query += " ORDER BY ts DESC LIMIT :limit"
    params['limit'] = limit

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()

    if not rows:
        return []

    return [
        {
            'asset_id': row[0],
            'ts': str(row[1]),
            'metric': row[2],
            'value': row[3]
        }
        for row in rows
    ]

def get_latest_analytics(asset_id: Optional[int] = None, metric: Optional[str] = None,
                        limit: int = 100) -> pd.DataFrame:
    """
    Retrieve latest analytics data from database.

    Args:
        asset_id: Filter by asset ID.
        metric: Filter by metric name.
        limit: Maximum number of records to return.

    Returns:
        pd.DataFrame: Analytics data.
    """
    query = "SELECT asset_id, ts, metric, value FROM analytics WHERE 1=1"
    params = {}

    if asset_id is not None:
        query += " AND asset_id = :asset_id"
        params['asset_id'] = asset_id

    if metric is not None:
        query += " AND metric LIKE :metric"
        params['metric'] = f"%{metric}%"

    query += " ORDER BY ts DESC LIMIT :limit"
    params['limit'] = limit

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['asset_id', 'ts', 'metric', 'value'])
    df['ts'] = pd.to_datetime(df['ts'])
    return df
