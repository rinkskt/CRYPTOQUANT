"""
API Client for Dashboard

This module provides functions to interact with the crypto quant API endpoints.
Handles data fetching, error handling, and data transformation for the dashboard.
"""

import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st


# API base URL - configure this based on your setup
import os

# Default to localhost for development
API_BASE_URL = "http://localhost:8000/api/v1"

# Override for production environments
if os.getenv('API_BASE_URL'):
    API_BASE_URL = os.getenv('API_BASE_URL')
elif os.getenv('STREAMLIT_SERVER_HEADLESS', '').lower() == 'true':
    # Running on Streamlit Cloud - use Render API
    API_BASE_URL = "https://cryptoquant.onrender.com/api/v1"


def get_assets() -> List[Dict[str, Any]]:
    """
    Fetch list of available assets from API with fallback to direct Binance data.

    Returns:
        List of asset dictionaries with id, symbol, name, exchange, active
    """
    try:
        # Try API first
        response = requests.get(f"{API_BASE_URL}/assets", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        # Fallback to direct Binance data
        try:
            from app.analytics.portfolio.data_loader import get_portfolio_data
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            data = get_portfolio_data(symbols, '1d', 1)
            return [{'symbol': sym, 'price': data[sym]['close'].iloc[-1]} for sym in symbols if sym in data]
        except Exception as e:
            st.error(f"Erro ao buscar ativos: {e}")
            return []


def get_ohlcv_data(symbol: str, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV data for a specific asset.

    Args:
        symbol: Asset symbol (e.g., 'BTC/USDT')
        start_date: Start date for data
        end_date: End date for data
        limit: Maximum number of records

    Returns:
        List of OHLCV records
    """
    try:
        params = {"limit": limit}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        # Garantir que o símbolo tenha o par USDT
        trading_symbol = symbol if "/USDT" in symbol else f"{symbol}/USDT"
        response = requests.get(f"{API_BASE_URL}/ohlcv", params={"symbol": trading_symbol, **params}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar dados OHLCV para {symbol}: {e}")
        return []


def get_analytics_data(asset_id: Optional[int] = None, symbol: Optional[str] = None,
                      metric: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch analytics data with optional filtering.

    Args:
        asset_id: Asset ID to filter by
        symbol: Asset symbol to filter by
        metric: Metric name to filter by
        limit: Maximum number of records

    Returns:
        List of analytics records
    """
    try:
        params = {"limit": limit}
        if asset_id:
            params["asset_id"] = asset_id
        if symbol:
            params["symbol"] = symbol
        if metric:
            params["metric"] = metric

        response = requests.get(f"{API_BASE_URL}/analytics", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar dados analíticos: {e}")
        return []


def get_cointegrated_pairs(top_n: int = 50, format: str = "json") -> Any:
    """
    Fetch cointegrated pairs analysis.

    Args:
        top_n: Number of top pairs to analyze
        format: Response format ('json' or 'csv')

    Returns:
        Cointegrated pairs data
    """
    try:
        params = {"top_n": top_n, "format": format}
        response = requests.get(f"{API_BASE_URL}/analytics/cointegrated_pairs", params=params, timeout=30)
        response.raise_for_status()

        if format == "csv":
            return response.text
        else:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar pares cointegrados: {e}")
        return [] if format == "json" else ""


def get_correlation_matrix(limit: int = 100) -> Dict[str, Any]:
    """
    Fetch correlation matrix for all assets.

    Args:
        limit: Number of recent records to use for correlation calculation

    Returns:
        Correlation matrix data
    """
    try:
        params = {"limit": limit}
        response = requests.get(f"{API_BASE_URL}/analytics/correlation/matrix", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar matriz de correlação: {e}")
        return {}


def get_rolling_correlation(window: int = 30, format: str = "json") -> Any:
    """
    Fetch rolling correlation matrix.

    Args:
        window: Rolling window size in days
        format: Response format ('json' or 'csv')

    Returns:
        Rolling correlation data
    """
    try:
        params = {"window": window, "format": format}
        response = requests.get(f"{API_BASE_URL}/analytics/rolling_correlation", params=params, timeout=30)
        response.raise_for_status()

        if format == "csv":
            return response.text
        else:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar correlação rolling: {e}")
        return {} if format == "json" else ""


def trigger_etl_run() -> Dict[str, Any]:
    """
    Trigger ETL pipeline run.

    Returns:
        Response from ETL trigger
    """
    try:
        response = requests.post(f"{API_BASE_URL}/ingest/run", timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao executar ETL: {e}")
        return {"error": str(e)}


def trigger_analytics_run() -> Dict[str, Any]:
    """
    Trigger analytics pipeline run.

    Returns:
        Response from analytics trigger
    """
    try:
        response = requests.post(f"{API_BASE_URL}/analytics/run", timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao executar analytics: {e}")
        return {"error": str(e)}


def get_health_status() -> Dict[str, Any]:
    """
    Get API health status.

    Returns:
        Health status information
    """
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


def convert_ohlcv_to_dataframe(ohlcv_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert OHLCV API response to pandas DataFrame.

    Args:
        ohlcv_data: OHLCV data from API

    Returns:
        DataFrame with OHLCV data
    """
    if not ohlcv_data:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv_data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    return df


def convert_analytics_to_dataframe(analytics_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert analytics API response to pandas DataFrame.

    Args:
        analytics_data: Analytics data from API

    Returns:
        DataFrame with analytics data
    """
    if not analytics_data:
        return pd.DataFrame()

    df = pd.DataFrame(analytics_data)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')

    return df
