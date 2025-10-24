"""
Portfolio Risk Analysis Module

Implementa cálculos e métricas de risco para análise de portfólio:
- Volatilidade e correlações
- Value at Risk (VaR)
- Expected Shortfall (CVaR)
- Drawdown e Beta
- Contribuição ao risco
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime


def compute_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcula Value at Risk (VaR) usando método histórico.

    Args:
        returns: Série de retornos
        alpha: Nível de confiança (0.05 = 95%)

    Returns:
        VaR como percentual negativo
    """
    if returns.empty:
        return 0.0

    return -np.percentile(returns.dropna(), 100 * alpha)


def compute_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcula Conditional Value at Risk (CVaR/Expected Shortfall).

    Args:
        returns: Série de retornos
        alpha: Nível de confiança

    Returns:
        CVaR como percentual negativo
    """
    if returns.empty:
        return 0.0

    var = compute_var(returns, alpha)
    tail_losses = returns[returns <= -var]

    if len(tail_losses) == 0:
        return var

    return -tail_losses.mean()


def calculate_volatility(returns: pd.DataFrame,
                        weights: Dict[str, float],
                        annualize: bool = True) -> Tuple[float, pd.DataFrame]:
    """
    Calcula volatilidade do portfólio e matriz de covariância.

    σ_p = √(w^T Σ w)

    Args:
        returns: DataFrame de retornos
        weights: Dict com pesos
        annualize: Se deve anualizar (365 dias)

    Returns:
        (volatilidade_portfolio, matriz_covariancia)
    """
    # Matriz de covariância
    cov_matrix = returns.cov()
    if annualize:
        cov_matrix = cov_matrix * 365

    # Array de pesos
    w = np.array([weights.get(col, 0) for col in returns.columns])

    # Volatilidade do portfólio
    portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    return portfolio_vol, cov_matrix


def calculate_correlations(returns: pd.DataFrame,
                         method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula matriz de correlação entre ativos.

    ρ_ij = Cov(r_i,r_j)/(σ_i σ_j)

    Args:
        returns: DataFrame de retornos
        method: Método de correlação ('pearson', 'spearman', 'kendall')

    Returns:
        DataFrame com correlações
    """
    return returns.corr(method=method)


def calculate_var(returns: pd.Series,
                 confidence: float = 0.95,
                 method: str = 'historical') -> Dict[str, float]:
    """
    Calcula Value at Risk (VaR) do portfólio.

    VaR_α = Z_α × σ_p - μ_p

    Args:
        returns: Série de retornos
        confidence: Nível de confiança (0.95 = 95%)
        method: 'historical', 'parametric' ou 'monte_carlo'

    Returns:
        Dict com VaR e parâmetros
    """
    if method == 'parametric':
        # Assume distribuição normal
        z_score = stats.norm.ppf(1 - confidence)
        var = -(returns.mean() + z_score * returns.std())

    elif method == 'monte_carlo':
        # Simulação Monte Carlo
        n_sims = 10000
        mu = returns.mean()
        sigma = returns.std()
        sims = np.random.normal(mu, sigma, n_sims)
        var = -np.percentile(sims, (1 - confidence) * 100)

    else:  # method == 'historical'
        # VaR histórico
        var = -returns.quantile(1 - confidence)

    return {
        'var': var,
        'confidence': confidence,
        'method': method
    }


def calculate_expected_shortfall(returns: pd.Series,
                               confidence: float = 0.95) -> float:
    """
    Calcula Expected Shortfall (CVaR) do portfólio.
    Média das perdas além do VaR.

    Args:
        returns: Série de retornos
        confidence: Nível de confiança

    Returns:
        Expected Shortfall
    """
    var = calculate_var(returns, confidence)['var']
    return -returns[returns < -var].mean()


def calculate_drawdown(prices: pd.Series) -> pd.DataFrame:
    """
    Calcula série de drawdowns.

    DD_t = (P_t - max(P_1..t)) / max(P_1..t)

    Args:
        prices: Série de preços

    Returns:
        DataFrame com drawdown e drawdown duration
    """
    # Máximo histórico
    rolling_max = prices.cummax()

    # Drawdown em %
    drawdown = (prices - rolling_max) / rolling_max

    # Marca início de cada drawdown
    is_dd_start = drawdown < 0

    # Duração do drawdown atual
    dd_duration = is_dd_start.astype(int).groupby(
        (is_dd_start != is_dd_start.shift()).cumsum()
    ).cumsum()

    return pd.DataFrame({
        'drawdown': drawdown,
        'duration': dd_duration
    })


def calculate_beta(returns: pd.Series,
                  market_returns: pd.Series,
                  rolling_window: Optional[int] = None) -> pd.Series:
    """
    Calcula beta em relação ao mercado.

    β_i = Cov(r_i,r_m)/Var(r_m)

    Args:
        returns: Retornos do portfólio
        market_returns: Retornos do benchmark
        rolling_window: Janela para beta móvel (opcional)

    Returns:
        Série de betas
    """
    if rolling_window:
        # Beta móvel
        covariance = returns.rolling(rolling_window).cov(market_returns)
        variance = market_returns.rolling(rolling_window).var()
        beta = covariance / variance
    else:
        # Beta total
        beta = returns.cov(market_returns) / market_returns.var()

    return beta


def calculate_risk_contribution(returns: pd.DataFrame,
                             weights: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula contribuição ao risco por ativo.

    RC_i = w_i × (Σw)_i / σ_p

    Args:
        returns: DataFrame de retornos
        weights: Dict com pesos

    Returns:
        Dict com contribuição por ativo
    """
    # Array de pesos
    w = np.array([weights.get(col, 0) for col in returns.columns])

    # Matriz de covariância
    cov = returns.cov() * 365

    # Volatilidade do portfólio
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))

    # Contribuição marginal ao risco
    mcr = np.dot(cov, w)

    # Contribuição ao risco
    rc = {}
    for i, symbol in enumerate(returns.columns):
        rc[symbol] = w[i] * mcr[i] / port_vol

    return rc


def calculate_conditional_var(returns: pd.DataFrame,
                            weights: Dict[str, float],
                            confidence: float = 0.95) -> Dict[str, float]:
    """
    Calcula CoVaR (VaR Condicional) por ativo.

    Args:
        returns: DataFrame de retornos
        weights: Dict com pesos
        confidence: Nível de confiança

    Returns:
        Dict com CoVaR por ativo
    """
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    var = calculate_var(portfolio_returns, confidence)['var']

    covar = {}
    for col in returns.columns:
        # Retornos condicionais quando portfólio < VaR
        cond_returns = returns[col][portfolio_returns < -var]
        covar[col] = -cond_returns.mean()

    return covar
