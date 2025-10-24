"""
Quantitative Crypto Dashboard

Main Streamlit application with multipage navigation.
"""

import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Quantitative Crypto Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .alert-low {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Main application function.
    """
    # Sidebar navigation
    st.sidebar.title("📊 Quantitative Crypto Dashboard")

    # Page selection
    pages = {
        "� Portfólio": "portfolio",
        "🔗 Correlação": "correlation",
        "� Visão Geral": "overview",
        "📈 Detalhes do Ativo": "asset_detail",
        "🧪 Laboratório": "lab",
        "🚨 Alertas": "alerts"
    }

    selected_page = st.sidebar.selectbox(
        "Navegação",
        options=list(pages.keys()),
        index=0  # Portfólio como página inicial
    )

    # Status indicator
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Status do Sistema")

    # Mock system status (in real implementation, check actual services)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric("API", "Online 🟢")
        st.metric("Database", "Online 🟢")

    with col2:
        st.metric("Analytics", "Online 🟢")
        st.metric("ETL", "Online 🟢")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Quantitative Crypto Dashboard v1.0*")

    # Main content area
    if pages[selected_page] == "overview":
        from app.dashboard.pages.overview import show_overview_page
        show_overview_page()
    elif pages[selected_page] == "correlation":
        from app.dashboard.pages.correlation import show_correlation_page
        show_correlation_page()
    elif pages[selected_page] == "portfolio":
        from app.dashboard.pages.portfolio import show_portfolio_page
        show_portfolio_page()
    elif pages[selected_page] == "asset_detail":
        from app.dashboard.pages.asset_detail import show_asset_detail_page
        show_asset_detail_page()
    elif pages[selected_page] == "lab":
        from app.dashboard.pages.lab import show_lab_page
        show_lab_page()
    elif pages[selected_page] == "alerts":
        from app.dashboard.pages.alerts import show_alerts_page
        show_alerts_page()

    # Footer
    st.markdown("---")
    st.markdown("*Dashboard desenvolvido para análise quantitativa de criptomoedas*")


if __name__ == "__main__":
    main()
