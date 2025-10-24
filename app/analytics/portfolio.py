"""
Portfolio Analytics Module

This module provides functions for portfolio-level analysis including value calculation,
returns computation, beta estimation, and cointegration testing.

Functions:
- compute_portfolio_value: Calculate portfolio value over time
- compute_portfolio_returns: Calculate portfolio returns
- compute_portfolio_beta: Calculate portfolio beta vs benchmark
- compute_asset_betas: Calculate individual asset betas
- compute_portfolio_cointegration: Test portfolio cointegration vs benchmark
- compute_asset_zscores_relative_portfolio: Calculate z-scores relative to portfolio
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
from typing import Dict, Any, Optional


def compute_portfolio_value(prices_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Compute portfolio value over time given asset prices and weights.

    Manual: This function calculates the total value of a portfolio over time by multiplying
    each asset's price by its weight and summing. Weights should sum to 1.0 for a fully
    invested portfolio. Use this to track portfolio performance over time.

    Parameters:
    - prices_df: DataFrame with assets as columns, timestamps as index
    - weights: Dict mapping asset names to portfolio weights

    Returns:
    - Series of portfolio values over time
    """
    if prices_df.empty or not weights:
        return pd.Series(dtype=float)

    # Filter to assets in weights
    available_assets = [asset for asset in weights.keys() if asset in prices_df.columns]
    if not available_assets:
        return pd.Series(dtype=float)

    # Normalize weights for available assets
    total_weight = sum(weights[asset] for asset in available_assets)
    if total_weight == 0:
        return pd.Series(dtype=float)

    normalized_weights = {asset: weights[asset] / total_weight for asset in available_assets}

    # Calculate weighted prices
    portfolio_value = pd.Series(0.0, index=prices_df.index, dtype=float)
    for asset in available_assets:
        portfolio_value += prices_df[asset] * normalized_weights[asset]

    return portfolio_value


def compute_portfolio_returns(portfolio_value_series: pd.Series) -> pd.Series:
    """
    Compute portfolio returns from portfolio value series.

    Manual: Returns are calculated as the percentage change in portfolio value.
    This uses log returns for better statistical properties (additivity over time).
    Positive returns indicate portfolio growth, negative indicate decline.

    Parameters:
    - portfolio_value_series: Series of portfolio values over time

    Returns:
    - Series of portfolio returns
    """
    if portfolio_value_series.empty or len(portfolio_value_series) < 2:
        return pd.Series(dtype=float)

    # Calculate log returns: ln(P_t / P_{t-1})
    returns = np.log(portfolio_value_series / portfolio_value_series.shift(1))
    return returns.dropna()


def compute_portfolio_beta(returns_df: pd.DataFrame, benchmark_symbol: str, weights: Dict[str, float]) -> float:
    """
    Compute portfolio beta relative to a benchmark.

    Manual: Beta measures portfolio sensitivity to market movements. Beta = 1 means
    portfolio moves with the market. Beta > 1 means more volatile, beta < 1 means less volatile.
    Used for risk assessment and portfolio optimization.

    Parameters:
    - returns_df: DataFrame with asset returns as columns
    - benchmark_symbol: Column name of benchmark asset
    - weights: Dict mapping asset names to portfolio weights

    Returns:
    - Portfolio beta (float)
    """
    if returns_df.empty or benchmark_symbol not in returns_df.columns:
        return np.nan

    # Filter to assets in weights
    available_assets = [asset for asset in weights.keys() if asset in returns_df.columns]
    if not available_assets or benchmark_symbol not in available_assets:
        return np.nan

    # Get benchmark returns
    benchmark_returns = returns_df[benchmark_symbol].dropna()

    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=returns_df.index, dtype=float)
    total_weight = sum(weights[asset] for asset in available_assets)

    for asset in available_assets:
        weight = weights[asset] / total_weight
        portfolio_returns += returns_df[asset] * weight

    # Align series
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_index) < 10:  # Need minimum data
        return np.nan

    port_returns_aligned = portfolio_returns.loc[common_index]
    bench_returns_aligned = benchmark_returns.loc[common_index]

    # Calculate beta: Cov(R_p, R_b) / Var(R_b)
    try:
        covariance = np.cov(port_returns_aligned, bench_returns_aligned)[0, 1]
        variance = np.var(bench_returns_aligned)

        if variance > 0:
            beta = covariance / variance
            return beta
        else:
            return np.nan
    except:
        return np.nan


def compute_asset_betas(returns_df: pd.DataFrame, benchmark_symbol: str) -> Dict[str, float]:
    """
    Compute beta for each asset relative to benchmark.

    Manual: Individual asset betas show how each asset moves relative to the market.
    Assets with beta > 1 are more volatile than the market, useful for risk management.

    Parameters:
    - returns_df: DataFrame with asset returns as columns
    - benchmark_symbol: Column name of benchmark asset

    Returns:
    - Dict mapping asset names to their betas
    """
    if returns_df.empty or benchmark_symbol not in returns_df.columns:
        return {}

    benchmark_returns = returns_df[benchmark_symbol]
    betas = {}

    for asset in returns_df.columns:
        if asset == benchmark_symbol:
            continue

        try:
            # Align series
            common_index = returns_df[asset].index.intersection(benchmark_returns.index)
            if len(common_index) < 10:
                continue

            asset_returns = returns_df[asset].loc[common_index]
            bench_returns = benchmark_returns.loc[common_index]

            # Calculate beta
            covariance = np.cov(asset_returns, bench_returns)[0, 1]
            variance = np.var(bench_returns)

            if variance > 0:
                beta = covariance / variance
                betas[asset] = beta

        except:
            continue

    return betas


def compute_portfolio_cointegration(portfolio_series: pd.Series, benchmark_series: pd.Series) -> Dict[str, Any]:
    """
    Test cointegration between portfolio and benchmark.

    Manual: Cointegration test checks if portfolio and benchmark have a long-term relationship.
    Significant cointegration (low p-value) suggests the portfolio tracks the benchmark well.
    Useful for assessing portfolio-market alignment.

    Parameters:
    - portfolio_series: Portfolio value series
    - benchmark_series: Benchmark value series

    Returns:
    - Dict with cointegration test results
    """
    if portfolio_series.empty or benchmark_series.empty:
        return {'cointegrated': False, 'p_value': 1.0, 't_stat': 0.0}

    # Align series
    common_index = portfolio_series.index.intersection(benchmark_series.index)
    if len(common_index) < 30:  # Need sufficient data
        return {'cointegrated': False, 'p_value': 1.0, 't_stat': 0.0}

    port_aligned = portfolio_series.loc[common_index]
    bench_aligned = benchmark_series.loc[common_index]

    try:
        # Run cointegration test
        coint_result = coint(port_aligned, bench_aligned)
        t_stat, p_value, crit_values = coint_result

        return {
            'cointegrated': p_value < 0.05,
            'p_value': p_value,
            't_stat': t_stat,
            'critical_values': crit_values
        }
    except:
        return {'cointegrated': False, 'p_value': 1.0, 't_stat': 0.0}


def compute_asset_zscores_relative_portfolio(prices_df: pd.DataFrame, weights: Dict[str, float], window_z: int = 60) -> pd.DataFrame:
    """
    Compute z-scores for each asset relative to the portfolio.

    Manual: This calculates how each asset deviates from its expected value based on portfolio weights.
    High absolute z-scores indicate assets that are significantly over/under-valued relative to
    the portfolio, useful for rebalancing decisions.

    Parameters:
    - prices_df: DataFrame with asset prices
    - weights: Portfolio weights dict
    - window_z: Rolling window for z-score calculation

    Returns:
    - DataFrame with z-scores for each asset
    """
    if prices_df.empty or not weights:
        return pd.DataFrame()

    # Compute portfolio value
    portfolio_value = compute_portfolio_value(prices_df, weights)
    if portfolio_value.empty:
        return pd.DataFrame()

    zscores = {}

    for asset in prices_df.columns:
        if asset not in weights or weights[asset] == 0:
            continue

        try:
            # Calculate expected price based on portfolio weight
            expected_price = portfolio_value * weights[asset]

            # Create spread: actual - expected
            spread = prices_df[asset] - expected_price

            # Compute rolling z-score
            rolling_mean = spread.rolling(window=window_z).mean()
            rolling_std = spread.rolling(window=window_z).std()

            zscore = (spread - rolling_mean) / rolling_std
            zscores[asset] = zscore

        except:
            continue

    if zscores:
        return pd.DataFrame(zscores)
    else:
        return pd.DataFrame()
