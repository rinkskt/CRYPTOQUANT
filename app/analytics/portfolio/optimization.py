"""
Portfolio Optimization Module

Implementa algoritmos de otimização de portfólio:
- Maximização de Sharpe Ratio
- Risk Parity
- Equal Weight
- Minimum Variance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import cvxopt as opt
from cvxopt import matrix, solvers


def optimize_max_sharpe(returns_df: pd.DataFrame,
                       risk_free_rate: float = 0.02,
                       bounds: Optional[List[Tuple]] = None) -> Dict[str, any]:
    """
    Otimiza portfólio para maximizar Sharpe Ratio.

    Args:
        returns_df: DataFrame com retornos dos ativos
        risk_free_rate: Taxa livre de risco
        bounds: Limites para pesos individuais

    Returns:
        Dict com pesos otimizados e métricas
    """
    if returns_df.empty:
        return {'weights': {}, 'sharpe': 0, 'return': 0, 'volatility': 0}

    n_assets = len(returns_df.columns)

    # Função objetivo (negativa do Sharpe)
    def objective(weights):
        portfolio_return = np.sum(returns_df.mean() * weights) * 365
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 365, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return -sharpe

    # Restrições
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Soma dos pesos = 1
    ]

    # Bounds
    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    # Peso inicial
    initial_weights = np.array([1/n_assets] * n_assets)

    # Otimização
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

    if result.success:
        optimal_weights = result.x
        weights_dict = dict(zip(returns_df.columns, optimal_weights))

        # Calcular métricas
        portfolio_return = np.sum(returns_df.mean() * optimal_weights) * 365
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_df.cov() * 365, optimal_weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

        return {
            'weights': weights_dict,
            'sharpe_ratio': sharpe,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'success': True
        }
    else:
        return {'weights': {}, 'sharpe': 0, 'return': 0, 'volatility': 0, 'success': False}


def risk_parity(returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Implementa estratégia Risk Parity.

    Args:
        returns_df: DataFrame com retornos

    Returns:
        Dict com pesos otimizados
    """
    if returns_df.empty:
        return {}

    n_assets = len(returns_df.columns)
    cov_matrix = returns_df.cov() * 365

    # Função objetivo: minimizar variância da contribuição ao risco
    def objective(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        risk_contrib = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
        target_risk = portfolio_vol / n_assets
        return np.sum((risk_contrib - target_risk)**2)

    # Restrições
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Mínimo 1%, máximo 50%

    # Otimização
    initial_weights = np.array([1/n_assets] * n_assets)
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

    if result.success:
        return dict(zip(returns_df.columns, result.x))
    else:
        return {}


def optimize_equal_weight(returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estratégia Equal Weight simples.

    Args:
        returns_df: DataFrame com retornos

    Returns:
        Dict com pesos iguais
    """
    if returns_df.empty:
        return {}

    n_assets = len(returns_df.columns)
    weight = 1.0 / n_assets

    return {asset: weight for asset in returns_df.columns}


def optimize_minimum_variance(returns_df: pd.DataFrame,
                            bounds: Optional[List[Tuple]] = None) -> Dict[str, any]:
    """
    Otimiza para mínima variância.

    Args:
        returns_df: DataFrame com retornos
        bounds: Limites para pesos

    Returns:
        Dict com pesos e métricas
    """
    if returns_df.empty:
        return {'weights': {}, 'volatility': 0, 'success': False}

    n_assets = len(returns_df.columns)
    cov_matrix = returns_df.cov() * 365

    # Função objetivo: minimizar variância
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # Restrições
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

    # Bounds
    if bounds is None:
        bounds = [(0, 1) for _ in range(n_assets)]

    # Otimização
    initial_weights = np.array([1/n_assets] * n_assets)
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

    if result.success:
        optimal_weights = result.x
        weights_dict = dict(zip(returns_df.columns, optimal_weights))

        # Calcular volatilidade
        portfolio_vol = np.sqrt(result.fun)

        return {
            'weights': weights_dict,
            'volatility': portfolio_vol,
            'success': True
        }
    else:
        return {'weights': {}, 'volatility': 0, 'success': False}


def calculate_efficient_frontier(returns_df: pd.DataFrame,
                               risk_free_rate: float = 0.02,
                               n_portfolios: int = 100) -> pd.DataFrame:
    """
    Calcula fronteira eficiente com simulação Monte Carlo.

    Args:
        returns_df: DataFrame com retornos
        risk_free_rate: Taxa livre de risco
        n_portfolios: Número de portfólios simulados

    Returns:
        DataFrame com retornos e volatilidades da fronteira
    """
    if returns_df.empty:
        return pd.DataFrame()

    n_assets = len(returns_df.columns)
    mean_returns = returns_df.mean() * 365
    cov_matrix = returns_df.cov() * 365

    # Simulação Monte Carlo
    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        # Pesos aleatórios
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Retorno e volatilidade
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Sharpe Ratio
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

        results[0,i] = portfolio_return
        results[1,i] = portfolio_vol
        results[2,i] = sharpe
        weights_record.append(weights)

    # Criar DataFrame
    frontier = pd.DataFrame(results.T, columns=['return', 'volatility', 'sharpe'])
    frontier['weights'] = weights_record

    return frontier


def rebalance_portfolio(current_weights: Dict[str, float],
                       target_weights: Dict[str, float],
                       current_prices: Dict[str, float],
                       portfolio_value: float) -> Dict[str, any]:
    """
    Calcula ordens de rebalanceamento.

    Args:
        current_weights: Pesos atuais
        target_weights: Pesos desejados
        current_prices: Preços atuais
        portfolio_value: Valor total do portfólio

    Returns:
        Dict com ordens de compra/venda
    """
    orders = {}

    for asset in target_weights.keys():
        current_weight = current_weights.get(asset, 0)
        target_weight = target_weights[asset]

        current_value = current_weight * portfolio_value
        target_value = target_weight * portfolio_value

        # Diferença em valor
        value_diff = target_value - current_value

        # Quantidade a comprar/vender
        if asset in current_prices and current_prices[asset] > 0:
            quantity = value_diff / current_prices[asset]
            orders[asset] = {
                'quantity': quantity,
                'value': value_diff,
                'action': 'buy' if quantity > 0 else 'sell'
            }

    return orders


__all__ = [
    'optimize_max_sharpe',
    'risk_parity',
    'optimize_equal_weight',
    'optimize_minimum_variance',
    'calculate_efficient_frontier',
    'rebalance_portfolio'
]
