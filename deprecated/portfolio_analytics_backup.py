"""
Portfolio Analytics Module

Provides advanced portfolio analytics and risk management calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import norm

def calculate_portfolio_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate basic portfolio metrics.
    
    Returns:
        Tuple of (return, volatility, sharpe_ratio)
    """
    returns = prices.pct_change().dropna()
    
    # Calculate portfolio return
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    
    # Calculate portfolio volatility
    cov_matrix = returns.cov() * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0 for crypto)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def calculate_risk_contribution(
    prices: pd.DataFrame,
    weights: np.ndarray
) -> Dict[str, float]:
    """
    Calculate risk contribution of each asset.
    """
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    
    # Portfolio volatility
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Marginal contribution to risk
    mcr = np.dot(cov_matrix, weights) / port_vol
    
    # Component contribution to risk
    ccr = weights * mcr
    
    # Create dictionary of risk contributions
    risk_contrib = {asset: contrib for asset, contrib in zip(prices.columns, ccr)}
    
    return risk_contrib

def optimize_portfolio(
    prices: pd.DataFrame,
    target_return: Optional[float] = None,
    target_risk: Optional[float] = None,
    constraints: Optional[Dict] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Optimize portfolio weights using Mean-Variance Optimization.
    """
    returns = prices.pct_change().dropna()
    n_assets = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mu = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def portfolio_return(weights):
        return np.sum(weights * mu)
    
    # Optimization constraints
    constraints_list = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
    
    if target_return is not None:
        constraints_list.append(
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
        )
    
    if target_risk is not None:
        constraints_list.append(
            {'type': 'eq', 'fun': lambda x: portfolio_volatility(x) - target_risk}
        )
    
    # Asset-specific constraints
    if constraints:
        for asset_idx, (min_weight, max_weight) in constraints.items():
            constraints_list.extend([
                {'type': 'ineq', 'fun': lambda x, idx=asset_idx: x[idx] - min_weight},
                {'type': 'ineq', 'fun': lambda x, idx=asset_idx: max_weight - x[idx]}
            ])
    
    # Optimization
    result = minimize(
        portfolio_volatility,  # Minimize volatility
        x0=np.array([1/n_assets] * n_assets),  # Equal weights start
        method='SLSQP',
        constraints=constraints_list,
        bounds=tuple((0, 1) for _ in range(n_assets))
    )
    
    optimal_weights = result.x
    opt_return = portfolio_return(optimal_weights)
    opt_risk = portfolio_volatility(optimal_weights)
    
    return optimal_weights, opt_return, opt_risk

def calculate_efficient_frontier(
    prices: pd.DataFrame,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the efficient frontier points.
    """
    returns = prices.pct_change().dropna()
    
    # Calculate range of returns
    n_assets = len(returns.columns)
    mu = returns.mean().values * 252
    min_ret = min(mu)
    max_ret = max(mu)
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    efficient_returns = []
    efficient_vols = []
    efficient_weights = []
    
    for target_return in target_returns:
        weights, _, risk = optimize_portfolio(
            prices,
            target_return=target_return
        )
        efficient_returns.append(target_return)
        efficient_vols.append(risk)
        efficient_weights.append(weights)
    
    return np.array(efficient_returns), np.array(efficient_vols), np.array(efficient_weights)

def calculate_var_es(
    prices: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES).
    """
    returns = prices.pct_change().dropna()
    portfolio_returns = np.sum(returns * weights, axis=1)
    
    # Calculate VaR
    var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    # Calculate ES
    es = -portfolio_returns[portfolio_returns <= -var].mean()
    
    return var, es

def calculate_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    Calculate portfolio beta relative to market.
    """
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    
    return beta