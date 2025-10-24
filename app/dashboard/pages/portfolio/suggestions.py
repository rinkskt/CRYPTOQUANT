"""
Portfolio Suggestions Module

Sistema de sugestões inteligentes para gerenciamento do portfólio.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .portfolio_form import PortfolioPosition

def analyze_stop_level(position: PortfolioPosition, history: pd.DataFrame) -> Dict:
    """
    Analisa se o nível de stop está adequado baseado na volatilidade.
    """
    if not position.stop_loss or not position.price_current:
        return {}
    
    # Calcula volatilidade diária
    returns = history[position.symbol].pct_change().std()
    daily_vol = returns * np.sqrt(252)
    
    # Stop atual em %
    current_stop_pct = (position.price_entry - position.stop_loss) / position.price_entry
    
    # Stop sugerido (2x ATR)
    atr_stop_pct = daily_vol * 2
    
    return {
        "stop_too_tight": current_stop_pct < atr_stop_pct,
        "stop_too_wide": current_stop_pct > atr_stop_pct * 2,
        "suggested_stop": position.price_entry * (1 - atr_stop_pct)
    }

def analyze_target_distribution(position: PortfolioPosition) -> Dict:
    """
    Analisa se a distribuição dos alvos está adequada.
    """
    if not all([position.target_1, position.target_2, position.target_3]):
        return {}
    
    # Distância dos alvos em relação ao preço de entrada
    t1_dist = (position.target_1 - position.price_entry) / position.price_entry
    t2_dist = (position.target_2 - position.price_entry) / position.price_entry
    t3_dist = (position.target_3 - position.price_entry) / position.price_entry
    
    # Verifica se os alvos têm uma distribuição balanceada
    ideal_ratio = [1, 2, 3]  # T1 = 1x, T2 = 2x, T3 = 3x
    actual_ratio = [t1_dist, t2_dist/2, t3_dist/3]
    
    return {
        "targets_balanced": np.std(actual_ratio) < 0.2,  # Baixo desvio padrão
        "suggested_targets": {
            "target_1": position.price_entry * (1 + t1_dist * ideal_ratio[0]),
            "target_2": position.price_entry * (1 + t1_dist * ideal_ratio[1]),
            "target_3": position.price_entry * (1 + t1_dist * ideal_ratio[2])
        }
    }

def analyze_realization_strategy(position: PortfolioPosition) -> Dict:
    """
    Sugere estratégia de realização baseada na proximidade dos alvos.
    """
    if not position.price_current:
        return {}
    
    # Distância atual para cada alvo
    distances = {
        "target_1": ((position.target_1 - position.price_current) / position.price_current) if position.target_1 else None,
        "target_2": ((position.target_2 - position.price_current) / position.price_current) if position.target_2 else None,
        "target_3": ((position.target_3 - position.price_current) / position.price_current) if position.target_3 else None
    }
    
    # Realização atual em cada alvo
    realized = {
        "target_1": position.realized_1 or 0,
        "target_2": position.realized_2 or 0,
        "target_3": position.realized_3 or 0
    }
    
    suggestions = []
    
    # Verifica proximidade dos alvos
    for target, dist in distances.items():
        if dist is not None:
            target_num = int(target[-1])
            if dist < 0.02 and realized[target] < (0.3 * target_num):  # Dentro de 2% do alvo
                suggestions.append(f"Próximo do {target} - considere realizar {30*target_num}% da posição")
            elif dist < 0.05 and realized[target] < (0.15 * target_num):  # Dentro de 5% do alvo
                suggestions.append(f"Aproximando do {target} - prepare-se para realizar {15*target_num}% da posição")
    
    return {
        "realization_suggestions": suggestions
    }

def analyze_portfolio_balance(portfolio: List[PortfolioPosition]) -> Dict:
    """
    Analisa o balanceamento do portfólio e sugere ajustes.
    """
    if not portfolio:
        return {}
    
    # Calcula alocação atual
    total_value = sum(pos.qty * pos.price_current for pos in portfolio if pos.price_current)
    current_allocation = {
        pos.symbol: (pos.qty * pos.price_current) / total_value 
        for pos in portfolio if pos.price_current
    }
    
    # Compara com alvos
    rebalance_suggestions = []
    for pos in portfolio:
        if not pos.allocation_target:
            continue
        
        current = current_allocation.get(pos.symbol, 0)
        diff = pos.allocation_target - current
        
        if abs(diff) > 0.05:  # Diferença maior que 5%
            if diff > 0:
                rebalance_suggestions.append(f"Aumentar {pos.symbol} em {diff*100:.1f}%")
            else:
                rebalance_suggestions.append(f"Reduzir {pos.symbol} em {-diff*100:.1f}%")
    
    return {
        "current_allocation": current_allocation,
        "rebalance_suggestions": rebalance_suggestions,
        "portfolio_diversified": len(portfolio) >= 5 and max(current_allocation.values()) < 0.3
    }

def get_portfolio_suggestions(portfolio: List[PortfolioPosition], history: pd.DataFrame) -> List[Dict]:
    """
    Gera sugestões para cada posição no portfólio.
    """
    suggestions = []
    
    # Analisa cada posição individualmente
    for pos in portfolio:
        pos_suggestions = {
            "symbol": pos.symbol,
            "stop_analysis": analyze_stop_level(pos, history),
            "target_analysis": analyze_target_distribution(pos),
            "realization": analyze_realization_strategy(pos)
        }
        suggestions.append(pos_suggestions)
    
    # Analisa o portfólio como um todo
    portfolio_analysis = analyze_portfolio_balance(portfolio)
    
    return {
        "positions": suggestions,
        "portfolio": portfolio_analysis
    }