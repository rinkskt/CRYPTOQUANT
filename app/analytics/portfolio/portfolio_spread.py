"""
Portfolio Spread Analysis Module

Análise de spread entre portfólio e ativo de referência.
Inclui métricas de correlação, cointegration beta e Z-Score.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Tuple, Optional


def compute_portfolio_spread(portfolio_df: pd.Series, benchmark_df: pd.Series) -> pd.Series:
    """
    Calcula o spread entre o portfólio agregado e o ativo de referência.

    Normaliza ambos para comparação percentual.
    """
    portfolio_norm = portfolio_df / portfolio_df.iloc[0]
    benchmark_norm = benchmark_df / benchmark_df.iloc[0]
    spread = portfolio_norm - benchmark_norm
    return spread


def compute_correlation_metrics(portfolio_df: pd.Series,
                              benchmark_df: pd.Series,
                              window: int = 30) -> Tuple[float, pd.Series]:
    """
    Calcula correlação de Pearson e rolling correlation (janelada).
    """
    corr = portfolio_df.corr(benchmark_df)
    rolling_corr = portfolio_df.rolling(window).corr(benchmark_df)
    return corr, rolling_corr


def compute_cointegration_beta(portfolio_df: pd.Series, benchmark_df: pd.Series) -> float:
    """
    Usa regressão OLS para estimar o beta da cointegração.
    """
    # Alinha os dados por índice
    common_index = portfolio_df.index.intersection(benchmark_df.index)
    portfolio_aligned = portfolio_df.loc[common_index]
    benchmark_aligned = benchmark_df.loc[common_index]

    # Regressão OLS
    X = sm.add_constant(benchmark_aligned)
    model = sm.OLS(portfolio_aligned, X).fit()
    beta = model.params.iloc[1]  # beta é o coeficiente da variável independente
    return beta


def compute_zscore(spread: pd.Series) -> pd.Series:
    """
    Normaliza o spread (para medir desvios padrão do equilíbrio).
    """
    mean = spread.mean()
    std = spread.std()
    zscore = (spread - mean) / std
    return zscore


def analyze_spread_full(portfolio_df: pd.Series,
                       benchmark_df: pd.Series,
                       window: int = 30) -> Dict[str, any]:
    """
    Função principal que integra todos os cálculos de spread.

    Retorna dicionário com todos os resultados.
    """
    spread = compute_portfolio_spread(portfolio_df, benchmark_df)
    corr, rolling_corr = compute_correlation_metrics(portfolio_df, benchmark_df, window)
    beta = compute_cointegration_beta(portfolio_df, benchmark_df)
    zscore = compute_zscore(spread)

    return {
        "spread": spread,
        "beta": beta,
        "zscore": zscore,
        "corr": corr,
        "rolling_corr": rolling_corr
    }
