"""
Portfolio Module

Este módulo contém todas as funcionalidades relacionadas à gestão e análise de portfólio:
- Performance e métricas
- Análise de risco
- Otimização e rebalanceamento
- Simulações e alertas
"""

import streamlit as st

from .performance import show_performance_tab
from .risk import show_risk_tab
from .optimization import show_optimization_tab
from .rebalance import show_rebalance_tab
from .portfolio_form import show_portfolio_form

def show_portfolio_page():
    """
    Display the portfolio analysis page.
    """
    st.header("📈 Gestão de Portfólio")
    
    # Formulário de edição do portfólio
    show_portfolio_form()
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance",
        "⚖️ Risco",
        "🎯 Otimização",
        "♻️ Rebalanceamento"
    ])
    
    with tab1:
        show_performance_tab()
    
    with tab2:
        show_risk_tab()
    
    with tab3:
        show_optimization_tab()
    
    with tab4:
        show_rebalance_tab()

__all__ = ['show_portfolio_page']
