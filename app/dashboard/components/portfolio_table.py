"""
Portfolio Table Component

This module provides reusable portfolio table components.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def create_portfolio_table(portfolio_data, weights, current_prices):
    """
    Create an interactive portfolio table.

    Args:
        portfolio_data: Dictionary with portfolio information
        weights: Dictionary of asset weights
        current_prices: Dictionary of current prices

    Returns:
        pandas DataFrame for display
    """
    if not portfolio_data or not weights:
        return pd.DataFrame()

    table_data = []

    for asset, weight in weights.items():
        if asset in current_prices:
            price = current_prices[asset]
            quantity = portfolio_data.get('quantities', {}).get(asset, 0)
            value = quantity * price

            # Calculate additional metrics (mock data for now)
            beta = portfolio_data.get('betas', {}).get(asset, 1.0)
            zscore = portfolio_data.get('zscores', {}).get(asset, 0.0)
            volatility = portfolio_data.get('volatilities', {}).get(asset, 0.0)

            table_data.append({
                'Ativo': asset,
                'Quantidade': quantity,
                'Preço Atual': price,
                'Valor': value,
                'Peso (%)': weight * 100,
                'Beta': beta,
                'Z-Score': zscore,
                'Volatilidade': volatility
            })

    df = pd.DataFrame(table_data)

    # Format columns
    if not df.empty:
        df['Preço Atual'] = df['Preço Atual'].apply(lambda x: f"${x:.4f}")
        df['Valor'] = df['Valor'].apply(lambda x: f"${x:.2f}")
        df['Peso (%)'] = df['Peso (%)'].apply(lambda x: f"{x:.2f}")
        df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}")
        df['Z-Score'] = df['Z-Score'].apply(lambda x: f"{x:.2f}")
        df['Volatilidade'] = df['Volatilidade'].apply(lambda x: f"{x:.2f}")

    return df


def create_portfolio_rebalance_table(current_weights, target_weights, current_prices):
    """
    Create a rebalancing recommendation table.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        current_prices: Current asset prices

    Returns:
        pandas DataFrame with rebalancing suggestions
    """
    if not current_weights or not target_weights:
        return pd.DataFrame()

    rebalance_data = []

    for asset in set(current_weights.keys()) | set(target_weights.keys()):
        current_weight = current_weights.get(asset, 0)
        target_weight = target_weights.get(asset, 0)
        difference = target_weight - current_weight

        # Calculate trade amounts (simplified)
        portfolio_value = 100000  # Mock portfolio value
        current_value = portfolio_value * current_weight
        target_value = portfolio_value * target_weight
        trade_value = target_value - current_value

        if asset in current_prices:
            price = current_prices[asset]
            trade_quantity = trade_value / price if price > 0 else 0
        else:
            trade_quantity = 0

        rebalance_data.append({
            'Ativo': asset,
            'Peso Atual (%)': current_weight * 100,
            'Peso Alvo (%)': target_weight * 100,
            'Diferença (%)': difference * 100,
            'Valor Atual ($)': current_value,
            'Valor Alvo ($)': target_value,
            'Valor Trade ($)': trade_value,
            'Quantidade Trade': trade_quantity
        })

    df = pd.DataFrame(rebalance_data)

    # Format columns
    if not df.empty:
        df['Peso Atual (%)'] = df['Peso Atual (%)'].apply(lambda x: f"{x:.2f}")
        df['Peso Alvo (%)'] = df['Peso Alvo (%)'].apply(lambda x: f"{x:.2f}")
        df['Diferença (%)'] = df['Diferença (%)'].apply(lambda x: f"{x:.2f}")
        df['Valor Atual ($)'] = df['Valor Atual ($)'].apply(lambda x: f"${x:.2f}")
        df['Valor Alvo ($)'] = df['Valor Alvo ($)'].apply(lambda x: f"${x:.2f}")
        df['Valor Trade ($)'] = df['Valor Trade ($)'].apply(lambda x: f"${x:.2f}")
        df['Quantidade Trade'] = df['Quantidade Trade'].apply(lambda x: f"{x:.4f}")

    return df


def create_portfolio_risk_table(risk_metrics, assets):
    """
    Create a risk contribution table.

    Args:
        risk_metrics: Dictionary with risk metrics per asset
        assets: List of asset symbols

    Returns:
        pandas DataFrame with risk metrics
    """
    if not risk_metrics:
        return pd.DataFrame()

    risk_data = []

    for asset in assets:
        metrics = risk_metrics.get(asset, {})

        risk_data.append({
            'Ativo': asset,
            'Contribuição VaR (%)': metrics.get('var_contribution', 0) * 100,
            'Volatilidade (%)': metrics.get('volatility', 0) * 100,
            'Beta': metrics.get('beta', 1.0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100
        })

    df = pd.DataFrame(risk_data)

    # Format columns
    if not df.empty:
        df['Contribuição VaR (%)'] = df['Contribuição VaR (%)'].apply(lambda x: f"{x:.2f}")
        df['Volatilidade (%)'] = df['Volatilidade (%)'].apply(lambda x: f"{x:.2f}")
        df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}")
        df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        df['Max Drawdown (%)'] = df['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}")

    return df


def create_portfolio_performance_table(performance_data, periods):
    """
    Create a performance table for different time periods.

    Args:
        performance_data: Dictionary with performance metrics
        periods: List of time periods

    Returns:
        pandas DataFrame with performance by period
    """
    if not performance_data:
        return pd.DataFrame()

    perf_data = []

    for period in periods:
        metrics = performance_data.get(period, {})

        perf_data.append({
            'Período': period,
            'Retorno Total (%)': metrics.get('total_return', 0) * 100,
            'Retorno Anualizado (%)': metrics.get('annualized_return', 0) * 100,
            'Volatilidade (%)': metrics.get('volatility', 0) * 100,
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
            'Beta': metrics.get('beta', 1.0)
        })

    df = pd.DataFrame(perf_data)

    # Format columns
    if not df.empty:
        df['Retorno Total (%)'] = df['Retorno Total (%)'].apply(lambda x: f"{x:.2f}")
        df['Retorno Anualizado (%)'] = df['Retorno Anualizado (%)'].apply(lambda x: f"{x:.2f}")
        df['Volatilidade (%)'] = df['Volatilidade (%)'].apply(lambda x: f"{x:.2f}")
        df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        df['Max Drawdown (%)'] = df['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}")
        df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}")

    return df


def create_asset_allocation_table(weights, asset_info):
    """
    Create an asset allocation summary table.

    Args:
        weights: Dictionary of asset weights
        asset_info: Dictionary with additional asset information

    Returns:
        pandas DataFrame with allocation details
    """
    if not weights:
        return pd.DataFrame()

    allocation_data = []

    for asset, weight in weights.items():
        info = asset_info.get(asset, {})

        allocation_data.append({
            'Ativo': asset,
            'Peso (%)': weight * 100,
            'Setor': info.get('sector', 'N/A'),
            'País': info.get('country', 'N/A'),
            'Tipo': info.get('asset_type', 'Crypto'),
            'Rating': info.get('rating', 'N/A')
        })

    df = pd.DataFrame(allocation_data)

    # Sort by weight descending
    df = df.sort_values('Peso (%)', ascending=False)

    # Format columns
    if not df.empty:
        df['Peso (%)'] = df['Peso (%)'].apply(lambda x: f"{x:.2f}")

    return df


def create_portfolio_optimization_table(optimization_results):
    """
    Create a table showing optimization results.

    Args:
        optimization_results: Dictionary with optimization results

    Returns:
        pandas DataFrame with optimization scenarios
    """
    if not optimization_results:
        return pd.DataFrame()

    opt_data = []

    for scenario, metrics in optimization_results.items():
        opt_data.append({
            'Cenário': scenario,
            'Retorno Esperado (%)': metrics.get('expected_return', 0) * 100,
            'Volatilidade (%)': metrics.get('volatility', 0) * 100,
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
            'Beta': metrics.get('beta', 1.0)
        })

    df = pd.DataFrame(opt_data)

    # Format columns
    if not df.empty:
        df['Retorno Esperado (%)'] = df['Retorno Esperado (%)'].apply(lambda x: f"{x:.2f}")
        df['Volatilidade (%)'] = df['Volatilidade (%)'].apply(lambda x: f"{x:.2f}")
        df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        df['Max Drawdown (%)'] = df['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}")
        df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}")

    return df
