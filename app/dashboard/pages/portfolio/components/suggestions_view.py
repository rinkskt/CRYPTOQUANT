"""
Dashboard component para exibição de sugestões do portfólio.
"""

import streamlit as st
from typing import Dict, List
from ..suggestions import get_portfolio_suggestions
from ..portfolio_form import PortfolioPosition
import pandas as pd

def render_suggestions(portfolio: List[PortfolioPosition], history: pd.DataFrame):
    """Renderiza as sugestões do portfólio em um componente Streamlit."""
    
    suggestions = get_portfolio_suggestions(portfolio, history)
    
    st.subheader("📊 Sugestões do Portfólio")
    
    # Análise do portfólio como um todo
    with st.expander("Análise Geral do Portfólio", expanded=True):
        portfolio_analysis = suggestions["portfolio"]
        
        # Diversificação
        if portfolio_analysis.get("portfolio_diversified"):
            st.success("✅ Portfólio bem diversificado")
        else:
            st.warning("⚠️ Considere melhorar a diversificação do portfólio")
        
        # Sugestões de rebalanceamento
        if portfolio_analysis.get("rebalance_suggestions"):
            st.write("Sugestões de rebalanceamento:")
            for suggestion in portfolio_analysis["rebalance_suggestions"]:
                st.info(f"🔄 {suggestion}")
    
    # Análise individual das posições
    for pos_suggestion in suggestions["positions"]:
        symbol = pos_suggestion["symbol"]
        with st.expander(f"Análise: {symbol}"):
            
            # Análise de Stop Loss
            stop_analysis = pos_suggestion["stop_analysis"]
            if stop_analysis:
                st.subheader("Stop Loss")
                if stop_analysis.get("stop_too_tight"):
                    st.warning("⚠️ Stop loss muito apertado considerando a volatilidade")
                    st.write(f"Stop sugerido: {stop_analysis['suggested_stop']:.2f}")
                elif stop_analysis.get("stop_too_wide"):
                    st.warning("⚠️ Stop loss muito largo considerando a volatilidade")
                    st.write(f"Stop sugerido: {stop_analysis['suggested_stop']:.2f}")
                else:
                    st.success("✅ Stop loss bem posicionado")
            
            # Análise dos Alvos
            target_analysis = pos_suggestion["target_analysis"]
            if target_analysis:
                st.subheader("Alvos de Preço")
                if target_analysis.get("targets_balanced"):
                    st.success("✅ Distribuição de alvos bem balanceada")
                else:
                    st.warning("⚠️ Considere reajustar os alvos:")
                    suggested = target_analysis["suggested_targets"]
                    cols = st.columns(3)
                    with cols[0]:
                        st.write(f"T1: {suggested['target_1']:.2f}")
                    with cols[1]:
                        st.write(f"T2: {suggested['target_2']:.2f}")
                    with cols[2]:
                        st.write(f"T3: {suggested['target_3']:.2f}")
            
            # Sugestões de Realização
            realization = pos_suggestion["realization"]
            if realization.get("realization_suggestions"):
                st.subheader("Sugestões de Realização")
                for suggestion in realization["realization_suggestions"]:
                    st.info(f"💰 {suggestion}")
                    
def render_portfolio_metrics(portfolio: List[PortfolioPosition]):
    """Renderiza métricas gerais do portfólio."""
    
    if not portfolio:
        return
    
    st.subheader("📈 Métricas do Portfólio")
    
    # Cálculo de métricas
    total_value = sum(pos.qty * pos.price_current for pos in portfolio if pos.price_current)
    total_cost = sum(pos.qty * pos.price_entry for pos in portfolio)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_value / total_cost - 1) * 100
    
    # Layout em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Valor Total", f"${total_value:,.2f}")
    
    with col2:
        st.metric("P&L", f"${total_pnl:,.2f}", 
                 delta=f"{total_pnl_pct:+.1f}%")
    
    with col3:
        # Cálculo de diversificação
        n_assets = len(portfolio)
        max_allocation = max((pos.qty * pos.price_current / total_value) 
                           for pos in portfolio if pos.price_current)
        st.metric("Diversificação", 
                 f"{n_assets} ativos",
                 delta=f"Max {max_allocation*100:.1f}% por ativo")