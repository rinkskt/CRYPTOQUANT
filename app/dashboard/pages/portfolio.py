"""
Portfolio Dashboard Page

This module provides the portfolio analysis page for the Streamlit dashboard.
Shows portfolio composition, performance, risk metrics, and optimization insights.
"""

import streamlit as st

from app.dashboard.pages.portfolio.performance import show_performance_tab
from app.dashboard.pages.portfolio.risk import show_risk_tab
from app.dashboard.pages.portfolio.optimization import show_optimization_tab
from app.dashboard.pages.portfolio.rebalance import show_rebalance_tab
from app.dashboard.pages.portfolio.spread_analysis import show_spread_analysis_tab
from app.dashboard.pages.portfolio.portfolio_form import show_portfolio_form

def show_portfolio_page():
    """
    Display the portfolio analysis page.
    """
    st.header("📈 Gestão de Portfólio")
    
    # Formulário de edição do portfólio
    show_portfolio_form()
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance",
        "⚖️ Risco",
        "🎯 Otimização",
        "♻️ Rebalanceamento",
        "📈 Spread Analysis"
    ])

    with tab1:
        show_performance_tab()

    with tab2:
        show_risk_tab()

    with tab3:
        show_optimization_tab()

    with tab4:
        show_rebalance_tab()

    with tab5:
        show_spread_analysis_tab()

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual - Análise de Portfólio", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo:
        Analisar e otimizar sua carteira de criptomoedas através de múltiplas perspectivas.

        ### 📊 Abas e funcionalidades:
        - **Performance**: Veja retornos, volatilidade e comparação com benchmark
        - **Risco**: Analise VaR, Drawdown e sensibilidade do portfólio
        - **Otimização**: Receba sugestões de alocação baseadas em diferentes estratégias
        - **Rebalanceamento**: Ajuste sua carteira para a alocação ideal
        - **Spread Analysis**: Analise spreads entre pares de ativos

        ### 💡 Como interpretar:
        - **Beta > 1**: Seu portfólio é mais volátil que o mercado
        - **Sharpe Ratio**: Retorno por unidade de risco (maior = melhor)
        - **VaR 95%**: Potencial perda em condições normais de mercado
        - **Drawdown Máximo**: Maior perda acumulada histórica

        ### 🔧 Formulário de edição:
        - Adicione novos ativos com símbolo, quantidade e preço de entrada
        - Edite posições existentes diretamente na tabela
        - Marque ativos para exclusão usando a coluna "Excluir"
        - As mudanças são salvas automaticamente

        *Use os controles laterais para ajustar períodos e parâmetros*
        ''')
