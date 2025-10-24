"""
Portfolio Metrics Module

Este módulo consolida métricas agregadas e funções de resumo do portfólio.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .performance import compute_portfolio_value, compute_portfolio_returns
from .risk import calculate_volatility, calculate_var, calculate_drawdown


def summarize_portfolio(prices_df: pd.DataFrame,
                       weights: Dict[str, float],
                       benchmark_symbol: Optional[str] = None) -> Dict[str, any]:
    """
    Gera um resumo completo das métricas do portfólio.

    Args:
        prices_df: DataFrame com preços dos ativos
        weights: Dict com pesos do portfólio
        benchmark_symbol: Símbolo do benchmark (opcional)

    Returns:
        Dict com todas as métricas consolidadas
    """
    if prices_df.empty or not weights:
        return {}

    # Calcular valor e retornos do portfólio
    portfolio_value = compute_portfolio_value(prices_df, weights)
    portfolio_returns = compute_portfolio_returns(portfolio_value)

    if portfolio_returns.empty:
        return {}

    # Métricas básicas
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    annualized_return = portfolio_returns.mean() * 252 * 100
    volatility = portfolio_returns.std() * np.sqrt(252) * 100

    # Sharpe Ratio (assumindo taxa livre de risco de 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_returns.mean() - risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252)

    # VaR 95%
    var_95 = calculate_var(portfolio_returns, confidence=0.95)['var'] * 100

    # Drawdown máximo
    drawdown_df = calculate_drawdown(portfolio_value)
    max_drawdown = drawdown_df['drawdown'].min() * 100

    # Beta vs benchmark (se fornecido)
    beta = None
    if benchmark_symbol and benchmark_symbol in prices_df.columns:
        benchmark_returns = compute_portfolio_returns(pd.Series(prices_df[benchmark_symbol]))
        if not benchmark_returns.empty:
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 10:
                port_ret = portfolio_returns.loc[common_idx]
                bench_ret = benchmark_returns.loc[common_idx]
                cov = np.cov(port_ret, bench_ret)[0, 1]
                var_bench = np.var(bench_ret)
                if var_bench > 0:
                    beta = cov / var_bench

    summary = {
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'volatility_pct': volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95_pct': var_95,
        'max_drawdown_pct': max_drawdown,
        'beta': beta,
        'assets_count': len([w for w in weights.values() if w > 0]),
        'last_value': portfolio_value.iloc[-1],
        'start_value': portfolio_value.iloc[0],
        'period_days': len(portfolio_value)
    }

    return summary


def compute_portfolio_metrics_over_time(prices_df: pd.DataFrame,
                                       weights: Dict[str, float],
                                       window_days: int = 252) -> pd.DataFrame:
    """
    Calcula métricas do portfólio em janelas móveis.

    Args:
        prices_df: DataFrame com preços
        weights: Pesos do portfólio
        window_days: Tamanho da janela móvel

    Returns:
        DataFrame com métricas ao longo do tempo
    """
    if prices_df.empty or not weights:
        return pd.DataFrame()

    portfolio_value = compute_portfolio_value(prices_df, weights)
    portfolio_returns = compute_portfolio_returns(portfolio_value)

    if portfolio_returns.empty:
        return pd.DataFrame()

    # Calcular métricas móveis
    rolling_returns = portfolio_returns.rolling(window=window_days)
    rolling_volatility = portfolio_returns.rolling(window=window_days).std() * np.sqrt(252)

    # Sharpe Ratio móvel
    risk_free_rate = 0.02
    rolling_sharpe = ((rolling_returns.mean() - risk_free_rate/252) / rolling_volatility) * np.sqrt(252)

    # Drawdown móvel
    rolling_max = portfolio_value.rolling(window=window_days).max()
    rolling_drawdown = (portfolio_value - rolling_max) / rolling_max

    metrics_df = pd.DataFrame({
        'rolling_return': rolling_returns.mean() * 252,
        'rolling_volatility': rolling_volatility,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown
    })

    return metrics_df.dropna()


def compare_portfolios(portfolio_data: Dict[str, Tuple[pd.DataFrame, Dict[str, float]]],
                      benchmark_symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Compara múltiplos portfólios lado a lado.

    Args:
        portfolio_data: Dict com nome -> (prices_df, weights)
        benchmark_symbol: Símbolo do benchmark

    Returns:
        DataFrame com métricas comparativas
    """
    results = {}

    for name, (prices_df, weights) in portfolio_data.items():
        summary = summarize_portfolio(prices_df, weights, benchmark_symbol)
        if summary:
            results[name] = summary

    if not results:
        return pd.DataFrame()

    return pd.DataFrame.from_dict(results, orient='index')


__all__ = [
    'summarize_portfolio',
    'compute_portfolio_metrics_over_time',
    'compare_portfolios'
]
