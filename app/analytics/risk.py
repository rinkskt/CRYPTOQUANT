"""
Risk Analytics Module

This module provides comprehensive risk analysis functions including Value at Risk (VaR),
Expected Shortfall (ES), Sharpe ratio, Sortino ratio, maximum drawdown, and volatility measures.

Functions:
- compute_var: Calculate Value at Risk
- compute_expected_shortfall: Calculate Expected Shortfall
- compute_sharpe_ratio: Calculate Sharpe ratio
- compute_sortino_ratio: Calculate Sortino ratio
- compute_max_drawdown: Calculate maximum drawdown
- compute_volatility: Calculate rolling volatility
- compute_var_contribution: Calculate VaR contribution by asset
- compute_risk_metrics: Compute comprehensive risk metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


def compute_var(returns_series: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Compute Value at Risk (VaR) for a returns series.

    Manual: VaR estimates the maximum potential loss over a specific time period with a given confidence level.
    For example, 95% VaR of -5% means there's a 5% chance of losing more than 5% in the period.
    Lower (more negative) VaR indicates higher risk.

    Parameters:
    - returns_series: Series of returns
    - confidence_level: Confidence level (0.95 for 95% VaR)
    - method: 'historical', 'parametric', or 'monte_carlo'

    Returns:
    - VaR value (negative number representing potential loss)
    """
    if returns_series.empty or len(returns_series) < 30:
        return 0.0

    returns = returns_series.dropna()

    if method == 'historical':
        # Historical VaR: percentile of historical returns
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    elif method == 'parametric':
        # Parametric VaR: assumes normal distribution
        try:
            mean = returns.mean()
            std = returns.std()
            # For normal distribution, VaR = mean + std * z_score
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + std * z_score
            return var
        except:
            return 0.0

    elif method == 'monte_carlo':
        # Monte Carlo VaR: simulate many scenarios
        try:
            # Simple bootstrap resampling
            n_simulations = 10000
            simulated_returns = np.random.choice(returns, size=(n_simulations, len(returns)), replace=True)
            simulated_portfolio_returns = simulated_returns.mean(axis=1)  # Equal weighted
            var = np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
            return var
        except:
            return 0.0

    else:
        return 0.0


def compute_expected_shortfall(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Compute Expected Shortfall (ES) - the expected loss given that VaR is exceeded.

    Manual: ES measures the average loss when losses exceed the VaR threshold. It's more conservative
    than VaR as it considers the tail of the distribution. For risk management, ES provides a better
    estimate of extreme loss potential.

    Parameters:
    - returns_series: Series of returns
    - confidence_level: Confidence level

    Returns:
    - Expected Shortfall value
    """
    if returns_series.empty or len(returns_series) < 30:
        return 0.0

    returns = returns_series.dropna()
    var_threshold = compute_var(returns, confidence_level)

    # ES is the average of returns below VaR threshold
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) > 0:
        return tail_losses.mean()
    else:
        return var_threshold  # Fallback to VaR


def compute_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = 0.02, annualize: bool = True) -> float:
    """
    Compute Sharpe ratio - risk-adjusted return measure.

    Manual: Sharpe ratio measures excess return per unit of risk (volatility). Higher Sharpe ratios
    indicate better risk-adjusted performance. A ratio > 1 is generally considered good.
    Annualized version assumes 252 trading days.

    Parameters:
    - returns_series: Series of returns
    - risk_free_rate: Annual risk-free rate (default 2%)
    - annualize: Whether to annualize the ratio

    Returns:
    - Sharpe ratio
    """
    if returns_series.empty or len(returns_series) < 30:
        return 0.0

    returns = returns_series.dropna()

    # Adjust for risk-free rate (assuming daily)
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    if excess_returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / excess_returns.std()

    if annualize:
        # Annualize: multiply by sqrt(252)
        sharpe *= np.sqrt(252)

    return sharpe


def compute_sortino_ratio(returns_series: pd.Series, risk_free_rate: float = 0.02, annualize: bool = True) -> float:
    """
    Compute Sortino ratio - downside risk-adjusted return measure.

    Manual: Sortino ratio is similar to Sharpe but only considers downside volatility (negative returns).
    It's more relevant for investors who are only concerned about downside risk, not upside volatility.
    Higher values indicate better downside risk-adjusted performance.

    Parameters:
    - returns_series: Series of returns
    - risk_free_rate: Annual risk-free rate
    - annualize: Whether to annualize the ratio

    Returns:
    - Sortino ratio
    """
    if returns_series.empty or len(returns_series) < 30:
        return 0.0

    returns = returns_series.dropna()

    # Calculate downside returns (only negative excess returns)
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    # Sortino ratio uses downside deviation
    downside_deviation = downside_returns.std()
    sortino = excess_returns.mean() / downside_deviation

    if annualize:
        sortino *= np.sqrt(252)

    return sortino


def compute_max_drawdown(price_series: pd.Series) -> Dict[str, Any]:
    """
    Compute maximum drawdown and related metrics.

    Manual: Maximum drawdown measures the largest peak-to-trough decline in portfolio value.
    It's a key risk metric showing the worst-case scenario. Recovery time shows how long
    it takes to recover from the maximum loss.

    Parameters:
    - price_series: Series of prices/values

    Returns:
    - Dict with max_drawdown, peak_date, trough_date, recovery_date
    """
    if price_series.empty or len(price_series) < 2:
        return {
            'max_drawdown': 0.0,
            'peak_date': None,
            'trough_date': None,
            'recovery_date': None,
            'current_drawdown': 0.0
        }

    prices = price_series.dropna()

    # Calculate cumulative maximum (rolling peak)
    rolling_max = prices.expanding().max()

    # Calculate drawdown: (current - peak) / peak
    drawdowns = (prices - rolling_max) / rolling_max

    # Find maximum drawdown
    max_drawdown = drawdowns.min()
    max_dd_idx = drawdowns.idxmin()

    # Find peak before the maximum drawdown
    peak_idx = rolling_max.loc[:max_dd_idx].idxmax()

    # Find recovery date (when price returns to previous peak)
    recovery_idx = None
    if max_drawdown < 0:
        # Find first date after trough where price >= peak
        recovery_candidates = prices.loc[max_dd_idx:]
        recovery_idx = recovery_candidates[recovery_candidates >= rolling_max.loc[peak_idx]].first_valid_index()

    # Current drawdown
    current_drawdown = drawdowns.iloc[-1]

    return {
        'max_drawdown': max_drawdown,
        'peak_date': peak_idx,
        'trough_date': max_dd_idx,
        'recovery_date': recovery_idx,
        'current_drawdown': current_drawdown
    }


def compute_volatility(returns_series: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    """
    Compute rolling volatility (standard deviation).

    Manual: Volatility measures the dispersion of returns. Higher volatility indicates higher risk.
    Rolling volatility shows how risk changes over time. Annualized volatility assumes 252 trading days.

    Parameters:
    - returns_series: Series of returns
    - window: Rolling window size
    - annualize: Whether to annualize volatility

    Returns:
    - Series of rolling volatility
    """
    if returns_series.empty:
        return pd.Series(dtype=float)

    returns = returns_series.dropna()

    # Rolling standard deviation
    volatility = returns.rolling(window=window).std()

    if annualize:
        # Annualize by multiplying by sqrt(252)
        volatility *= np.sqrt(252)

    return volatility


def compute_var_contribution(returns_df: pd.DataFrame, weights: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Compute Value at Risk contribution by asset.

    Manual: VaR contribution shows how much each asset contributes to the portfolio's total VaR.
    This helps identify which assets are the main sources of risk and guides risk management decisions.

    Parameters:
    - returns_df: DataFrame with asset returns
    - weights: Portfolio weights
    - confidence_level: VaR confidence level

    Returns:
    - Dict mapping assets to their VaR contribution
    """
    if returns_df.empty or not weights:
        return {}

    contributions = {}

    # Calculate portfolio VaR
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    total_weight = sum(weights.values())

    for asset, weight in weights.items():
        if asset in returns_df.columns:
            portfolio_returns += returns_df[asset] * (weight / total_weight)

    portfolio_var = compute_var(portfolio_returns, confidence_level)

    # Calculate marginal VaR contribution for each asset
    for asset in weights.keys():
        if asset in returns_df.columns:
            try:
                # Marginal contribution = weight * (dVaR/dWeight)
                # Simplified: weight * (asset_volatility / portfolio_volatility) * portfolio_VaR
                asset_vol = returns_df[asset].std()
                port_vol = portfolio_returns.std()

                if port_vol > 0:
                    marginal_contrib = (weights[asset] / total_weight) * (asset_vol / port_vol) * portfolio_var
                    contributions[asset] = marginal_contrib
                else:
                    contributions[asset] = 0.0

            except:
                contributions[asset] = 0.0

    return contributions


def compute_risk_metrics(returns_series: pd.Series, price_series: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Compute comprehensive risk metrics.

    Manual: This function provides a complete risk profile including VaR, ES, Sharpe ratio,
    maximum drawdown, and volatility. Use this for comprehensive risk assessment.

    Parameters:
    - returns_series: Series of returns
    - price_series: Optional series of prices for drawdown calculation

    Returns:
    - Dict with all risk metrics
    """
    metrics = {}

    if not returns_series.empty:
        # VaR and Expected Shortfall
        metrics['var_95'] = compute_var(returns_series, 0.95)
        metrics['var_99'] = compute_var(returns_series, 0.99)
        metrics['expected_shortfall_95'] = compute_expected_shortfall(returns_series, 0.95)

        # Risk-adjusted returns
        metrics['sharpe_ratio'] = compute_sharpe_ratio(returns_series)
        metrics['sortino_ratio'] = compute_sortino_ratio(returns_series)

        # Volatility
        metrics['volatility'] = returns_series.std() * np.sqrt(252)  # Annualized
        metrics['rolling_volatility_30d'] = compute_volatility(returns_series, 30).iloc[-1] if len(returns_series) >= 30 else None

    # Maximum drawdown (requires price series)
    if price_series is not None and not price_series.empty:
        dd_metrics = compute_max_drawdown(price_series)
        metrics.update(dd_metrics)

    return metrics
