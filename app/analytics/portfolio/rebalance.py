"""
Portfolio Rebalancing Module

Implementa análise de desvios e sugestões de rebalanceamento:
- Cálculo de desvios da alocação alvo
- Sugestões de trades para rebalancear
- Análise de impacto do rebalanceamento
- Triggers de rebalanceamento automático
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RebalanceTrade:
    """Representa uma operação de rebalanceamento."""
    symbol: str
    current_weight: float
    target_weight: float
    trade_weight: float
    position_value: float
    trade_amount: float
    trade_type: str  # 'buy' ou 'sell'
    price: float
    quantity: float


class PortfolioRebalancer:
    def __init__(self, 
                positions: Dict[str, Dict],
                target_weights: Dict[str, float],
                prices: Dict[str, float],
                total_value: float):
        """
        Inicializa rebalanceador com dados do portfólio.

        Args:
            positions: Dict com posições atuais
            target_weights: Dict com pesos alvo
            prices: Dict com preços atuais
            total_value: Valor total do portfólio
        """
        self.positions = positions
        self.target_weights = target_weights
        self.prices = prices
        self.total_value = total_value
        self.current_weights = self._calculate_current_weights()
        
    def _calculate_current_weights(self) -> Dict[str, float]:
        """Calcula pesos atuais do portfólio."""
        weights = {}
        for symbol, pos in self.positions.items():
            position_value = pos['qty'] * self.prices[symbol]
            weights[symbol] = position_value / self.total_value
        return weights
    
    def calculate_deviations(self, 
                          relative: bool = True) -> Dict[str, float]:
        """
        Calcula desvios da alocação alvo.

        Δw_i = w_atual,i - w_ideal,i

        Args:
            relative: Se True, retorna desvio relativo (%)

        Returns:
            Dict com desvios por ativo
        """
        deviations = {}
        for symbol in self.target_weights:
            current = self.current_weights.get(symbol, 0)
            target = self.target_weights[symbol]
            
            if relative:
                deviations[symbol] = (current - target) / target if target > 0 else np.inf
            else:
                deviations[symbol] = current - target
                
        return deviations
    
    def suggest_trades(self,
                     min_trade_size: float = 0.01,
                     rebalance_threshold: float = 0.05) -> List[RebalanceTrade]:
        """
        Sugere trades para rebalancear o portfólio.

        Args:
            min_trade_size: Tamanho mínimo do trade em % do portfólio
            rebalance_threshold: Limite de desvio para sugerir trade

        Returns:
            Lista de RebalanceTrade
        """
        trades = []
        deviations = self.calculate_deviations(relative=False)
        
        for symbol, deviation in deviations.items():
            # Ignora desvios pequenos
            if abs(deviation) < rebalance_threshold:
                continue
                
            position_value = self.positions[symbol]['qty'] * self.prices[symbol]
            trade_value = deviation * self.total_value
            
            # Ignora trades muito pequenos
            if abs(trade_value) < min_trade_size * self.total_value:
                continue
            
            trade = RebalanceTrade(
                symbol=symbol,
                current_weight=self.current_weights.get(symbol, 0),
                target_weight=self.target_weights[symbol],
                trade_weight=deviation,
                position_value=position_value,
                trade_amount=abs(trade_value),
                trade_type='sell' if deviation < 0 else 'buy',
                price=self.prices[symbol],
                quantity=abs(trade_value) / self.prices[symbol]
            )
            trades.append(trade)
            
        return trades
    
    def calculate_rebalancing_impact(self, 
                                  trades: List[RebalanceTrade]) -> Dict:
        """
        Calcula impacto do rebalanceamento.

        Args:
            trades: Lista de trades sugeridos

        Returns:
            Dict com métricas de impacto
        """
        # Soma trades por tipo
        total_buys = sum(t.trade_amount for t in trades if t.trade_type == 'buy')
        total_sells = sum(t.trade_amount for t in trades if t.trade_type == 'sell')
        
        # Simula pesos após rebalanceamento
        new_weights = self.current_weights.copy()
        for trade in trades:
            if trade.trade_type == 'buy':
                new_weights[trade.symbol] += trade.trade_weight
            else:
                new_weights[trade.symbol] -= trade.trade_weight
                
        # Calcula tracking error
        tracking_error = np.sqrt(sum(
            (new_weights.get(k, 0) - v) ** 2 
            for k, v in self.target_weights.items()
        ))
        
        return {
            'n_trades': len(trades),
            'total_turnover': (total_buys + total_sells) / self.total_value,
            'total_buys': total_buys,
            'total_sells': total_sells,
            'tracking_error': tracking_error,
            'new_weights': new_weights
        }
    
    def check_rebalancing_triggers(self, 
                                thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Verifica gatilhos de rebalanceamento.

        Args:
            thresholds: Dict com limites por tipo de trigger

        Returns:
            Dict indicando quais triggers foram ativados
        """
        triggers = {}
        
        # Desvio máximo absoluto
        if 'max_deviation' in thresholds:
            max_dev = max(abs(d) for d in self.calculate_deviations(relative=False).values())
            triggers['max_deviation'] = max_dev > thresholds['max_deviation']
            
        # Tracking error
        if 'tracking_error' in thresholds:
            te = np.sqrt(sum(
                (self.current_weights.get(k, 0) - v) ** 2 
                for k, v in self.target_weights.items()
            ))
            triggers['tracking_error'] = te > thresholds['tracking_error']
            
        # Número de ativos desviados
        if 'assets_deviated' in thresholds:
            n_deviated = sum(
                1 for d in self.calculate_deviations(relative=True).values()
                if abs(d) > thresholds.get('min_deviation', 0.05)
            )
            triggers['assets_deviated'] = n_deviated >= thresholds['assets_deviated']
            
        return triggers
    
    def get_rebalancing_summary(self,
                             min_trade_size: float = 0.01,
                             rebalance_threshold: float = 0.05) -> Dict:
        """
        Gera resumo completo do rebalanceamento.

        Args:
            min_trade_size: Tamanho mínimo do trade
            rebalance_threshold: Limite de desvio

        Returns:
            Dict com análise completa
        """
        # Calcula desvios
        deviations = self.calculate_deviations(relative=True)
        
        # Sugere trades
        trades = self.suggest_trades(min_trade_size, rebalance_threshold)
        
        # Calcula impacto
        impact = self.calculate_rebalancing_impact(trades)
        
        # Triggers padrão
        triggers = self.check_rebalancing_triggers({
            'max_deviation': 0.10,
            'tracking_error': 0.05,
            'assets_deviated': 3,
            'min_deviation': 0.05
        })
        
        return {
            'current_weights': self.current_weights,
            'target_weights': self.target_weights,
            'deviations': deviations,
            'trades': trades,
            'impact': impact,
            'triggers': triggers
        }