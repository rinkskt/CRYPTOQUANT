"""
Portfolio Optimization Module

Implementa diferentes estratégias de otimização de portfólio:
- Equal Weight
- Minimum Variance
- Maximum Sharpe Ratio
- Risk Parity
- Maximum Diversification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import cvxopt as cv
from cvxopt import solvers

solvers.options['show_progress'] = False

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame):
        """
        Inicializa otimizador com dados de retornos.
        
        Args:
            returns: DataFrame de retornos
        """
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.mu = returns.mean().values * 252  # Retornos anualizados
        self.S = returns.cov().values * 252    # Covariância anualizada
        self.vols = returns.std().values * np.sqrt(252)  # Volatilidades anualizadas
        
    def equal_weight(self) -> Dict[str, float]:
        """Pesos iguais: w_i = 1/N"""
        weight = 1.0 / self.n_assets
        return {col: weight for col in self.returns.columns}
    
    def volatility_weighted(self) -> Dict[str, float]:
        """Pesos por volatilidade: w_i = (1/σ_i) / Σ(1/σ_j)"""
        inv_vols = 1.0 / self.vols
        weights = inv_vols / inv_vols.sum()
        return dict(zip(self.returns.columns, weights))
    
    def minimum_variance(self, constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Mínima variância: min w^T Σ w
        """
        # Prepare CVXOPT matrices
        P = cv.matrix(self.S)
        q = cv.matrix(np.zeros(self.n_assets))
        
        # Restrição: soma = 1
        A = cv.matrix(1.0, (1, self.n_assets))
        b = cv.matrix(1.0)
        
        # Restrição: pesos >= 0
        G = cv.matrix(-np.eye(self.n_assets))
        h = cv.matrix(np.zeros(self.n_assets))
        
        if constraints:
            if 'min_weight' in constraints:
                min_w = constraints['min_weight']
                G = cv.matrix(np.vstack((-np.eye(self.n_assets), np.eye(self.n_assets))))
                h = cv.matrix(np.hstack((np.zeros(self.n_assets), np.ones(self.n_assets) * min_w)))
            
            if 'max_weight' in constraints:
                max_w = constraints['max_weight']
                G = cv.matrix(np.vstack((np.eye(self.n_assets), -np.eye(self.n_assets))))
                h = cv.matrix(np.hstack((np.ones(self.n_assets) * max_w, np.zeros(self.n_assets))))
        
        sol = solvers.qp(P, q, G, h, A, b)
        if sol['status'] != 'optimal':
            raise ValueError("Optimization failed")
            
        weights = np.array(sol['x']).flatten()
        return dict(zip(self.returns.columns, weights))
    
    def maximum_sharpe(self, risk_free_rate: float = 0.0, 
                      constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Máximo Sharpe: max (μ_p - r_f) / σ_p
        """
        def objective(w):
            w = np.array(w)
            ret = np.sum(w * self.mu)
            vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            sharpe = (ret - risk_free_rate) / vol
            return -sharpe
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        if constraints:
            if 'min_weight' in constraints:
                bounds = [(constraints['min_weight'], 1.0) for _ in range(self.n_assets)]
            if 'max_weight' in constraints:
                bounds = [(0.0, constraints['max_weight']) for _ in range(self.n_assets)]
        
        w0 = np.array([1.0/self.n_assets] * self.n_assets)
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if not result.success:
            raise ValueError("Optimization failed")
            
        return dict(zip(self.returns.columns, result.x))
    
    def risk_parity(self, risk_target: Optional[float] = None) -> Dict[str, float]:
        """
        Paridade de risco: w_i × (Σw)_i = const
        """
        def risk_parity_objective(w):
            w = np.array(w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            risk_contrib = w * (np.dot(self.S, w)) / port_vol
            target_risk = port_vol / self.n_assets
            risk_diffs = risk_contrib - target_risk
            return np.sum(np.square(risk_diffs))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        if risk_target:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w.T, np.dot(self.S, w))) - risk_target
            })
        
        w0 = np.array([1.0/self.n_assets] * self.n_assets)
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        
        result = minimize(risk_parity_objective, w0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        if not result.success:
            raise ValueError("Optimization failed")
            
        return dict(zip(self.returns.columns, result.x))
    
    def maximum_diversification(self, constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Máxima diversificação: max (Σw_i σ_i) / σ_p
        """
        def diversification_ratio(w):
            w = np.array(w)
            weighted_vols = np.sum(w * self.vols)
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            return -(weighted_vols / port_vol)
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        if constraints:
            if 'min_weight' in constraints:
                bounds = [(constraints['min_weight'], 1.0) for _ in range(self.n_assets)]
            if 'max_weight' in constraints:
                bounds = [(0.0, constraints['max_weight']) for _ in range(self.n_assets)]
        
        w0 = np.array([1.0/self.n_assets] * self.n_assets)
        result = minimize(diversification_ratio, w0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints_list)
        
        if not result.success:
            raise ValueError("Optimization failed")
            
        return dict(zip(self.returns.columns, result.x))
    
    def efficient_frontier(self, risk_free_rate: float = 0.0,
                         n_points: int = 100) -> pd.DataFrame:
        """
        Calcula pontos da fronteira eficiente.
        """
        def portfolio_stats(w):
            w = np.array(w)
            ret = np.sum(w * self.mu)
            vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            sr = (ret - risk_free_rate) / vol
            return np.array([ret, vol, sr])
        
        # Range de retornos
        min_ret, max_ret = np.min(self.mu), np.max(self.mu)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_portfolios = []
        for target_ret in target_returns:
            # Minimiza volatilidade para cada retorno alvo
            def objective(w):
                return np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(w * self.mu) - target_ret}
            ]
            
            bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
            w0 = np.array([1.0/self.n_assets] * self.n_assets)
            
            result = minimize(objective, w0,
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)
            
            if result.success:
                stats = portfolio_stats(result.x)
                efficient_portfolios.append({
                    'return': stats[0],
                    'volatility': stats[1],
                    'sharpe_ratio': stats[2],
                    'weights': dict(zip(self.returns.columns, result.x))
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def optimize(self, strategy: str = 'sharpe',
                risk_free_rate: float = 0.0,
                constraints: Optional[Dict] = None,
                **kwargs) -> Dict[str, float]:
        """
        Interface única para todas as estratégias.
        """
        strategies = {
            'equal': self.equal_weight,
            'volatility': self.volatility_weighted,
            'min_variance': self.minimum_variance,
            'max_sharpe': self.maximum_sharpe,
            'risk_parity': self.risk_parity,
            'max_div': self.maximum_diversification
        }
        
        if strategy not in strategies:
            raise ValueError(f"Estratégia '{strategy}' não suportada")
            
        if strategy == 'max_sharpe':
            return strategies[strategy](risk_free_rate, constraints)
        elif strategy in ['min_variance', 'max_div']:
            return strategies[strategy](constraints)
        elif strategy == 'risk_parity' and 'risk_target' in kwargs:
            return strategies[strategy](kwargs['risk_target'])
        else:
            return strategies[strategy]()