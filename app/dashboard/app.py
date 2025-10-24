"""
Aplicação principal do dashboard em Streamlit para análise quantitativa de criptomoedas.
"""

import streamlit as st
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Agora importa os módulos necessários
from app.dashboard.pages.overview import show_overview_page
from app.dashboard.pages.correlation import show_correlation_page
from app.dashboard.pages.portfolio import show_portfolio_page
from app.dashboard.pages.asset_detail import show_asset_detail_page
from app.dashboard.pages.lab import show_lab_page
from app.dashboard.pages.alerts import show_alerts_page
from app.dashboard.pages.rolling import show_rolling_page

def main():
    """
    Main application function.
    """
    # Configuração da página
    st.set_page_config(
        page_title="CryptoQuant Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personalizado para melhorar aparência
    st.markdown('''
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
    </style>
    ''', unsafe_allow_html=True)

    # ===== MENU PRINCIPAL =====
    st.sidebar.title("🚀 CryptoQuant Dashboard")
    st.sidebar.markdown("---")

    # Inicializar página padrão
    if 'page' not in st.session_state:
        st.session_state.page = "📈 Visão Geral"

    # GRUPO: ANÁLISE DO MERCADO
    st.sidebar.markdown("### 📊 Análise de Mercado")
    if st.sidebar.button("📈 Visão Geral", use_container_width=True):
        st.session_state.page = "📈 Visão Geral"
    if st.sidebar.button("🔗 Correlações", use_container_width=True):
        st.session_state.page = "🔗 Correlações"
    if st.sidebar.button("📊 Detalhes do Ativo", use_container_width=True):
        st.session_state.page = "📊 Detalhes do Ativo"

    # GRUPO: PORTFÓLIO & RISCO
    st.sidebar.markdown("### 💼 Gestão de Portfólio")
    if st.sidebar.button("🎯 Análise de Portfólio", use_container_width=True):
        st.session_state.page = "Portfólio"

    # GRUPO: FERRAMENTAS AVANÇADAS
    st.sidebar.markdown("### 🔧 Ferramentas")
    if st.sidebar.button("🧪 Laboratório Quant", use_container_width=True):
        st.session_state.page = "Laboratório"
    if st.sidebar.button("🔄 Dados em Tempo Real", use_container_width=True):
        st.session_state.page = "Rolling"

    # GRUPO: MONITORAMENTO
    st.sidebar.markdown("### ⚠️ Monitoramento")
    if st.sidebar.button("🚨 Sistema de Alertas", use_container_width=True):
        st.session_state.page = "Alertas"

    page = st.session_state.page

    st.sidebar.markdown("---")

    # STATUS DO SISTEMA MODERNIZADO
    st.sidebar.markdown("### 🖥️ Status do Sistema")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("API", "🟢", "Online")
    col2.metric("DB", "🟢", "Online")

    col3, col4 = st.sidebar.columns(2)
    col3.metric("Análise", "🟢", "Online")
    col4.metric("ETL", "🟢", "Online")

    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0 • CryptoQuant Analytics")

    # Main content area
    if page == "📈 Visão Geral":
        show_overview_page()
    elif page == "🔗 Correlações":
        show_correlation_page()
    elif page == "Portfólio":
        show_portfolio_page()
    elif page == "📊 Detalhes do Ativo":
        show_asset_detail_page()
    elif page == "Laboratório":
        show_lab_page()
    elif page == "Alertas":
        show_alerts_page()
    elif page == "Rolling":
        show_rolling_page()
    else:
        # Default para Visão Geral
        show_overview_page()


if __name__ == "__main__":
    main()
