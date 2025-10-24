"""
Streamlit Dashboard
"""

import streamlit as st
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Agora importa os módulos necessários
from app.dashboard.pages.overview import show_overview_page
from app.dashboard.pages.correlation import show_correlation_page
from app.dashboard.pages.portfolio import show_portfolio_page
from app.dashboard.pages.asset_detail import show_asset_detail_page
from app.dashboard.pages.lab import show_lab_page
from app.dashboard.pages.alerts import show_alerts_page

# Page configuration
st.set_page_config(
    page_title="Quantitative Crypto Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navegação")
pages = {
    "📊 Visão Geral": "overview",
    "🔄 Correlação": "correlation",
    "📈 Portfólio": "portfolio",
    "💹 Detalhes do Ativo": "asset_detail",
    "🧪 Laboratório": "lab",
    "🔔 Alertas": "alerts"
}

selected_page = st.sidebar.selectbox("Selecione uma página", list(pages.keys()))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Dashboard desenvolvido para análise quantitativa de criptomoedas")

# Show selected page
if pages[selected_page] == "overview":
    show_overview_page()
elif pages[selected_page] == "correlation":
    show_correlation_page()
elif pages[selected_page] == "portfolio":
    show_portfolio_page()
elif pages[selected_page] == "asset_detail":
    show_asset_detail_page()
elif pages[selected_page] == "lab":
    show_lab_page()
elif pages[selected_page] == "alerts":
    show_alerts_page()