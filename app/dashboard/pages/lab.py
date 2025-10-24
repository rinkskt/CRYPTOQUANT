"""
Lab Page - Strategy Simulator and Custom Indicators

This page provides tools for:
- Strategy backtesting
- Cointegration testing
- Custom technical indicators
- Portfolio simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from app.dashboard.pages.portfolio.portfolio_form import load_portfolio, PortfolioPosition
from app.analytics.portfolio import (
    get_binance_data,
    compute_portfolio_value,
    compute_portfolio_returns,
    calculate_volatility,
    calculate_correlations
)


def show_lab_page():
    """
    Main function to display the Lab page.
    """
    st.title("🧪 Lab - Strategy Simulator")

    # Create tabs for different lab tools
    tab1, tab2, tab3, tab4 = st.tabs([
        "Strategy Simulator",
        "Cointegration Tester",
        "Custom Indicators",
        "Portfolio Simulator"
    ])

    with tab1:
        show_strategy_simulator()

    with tab2:
        show_cointegration_tester_tab()

    with tab3:
        show_custom_indicators()

    with tab4:
        show_portfolio_simulator()


def show_strategy_simulator():
    """
    Strategy backtesting simulator.
    """
    st.header("Strategy Simulator")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Strategy Parameters")

        # Asset selection
        asset = st.selectbox(
            "Select Asset",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
            key="strategy_asset"
        )

        # Strategy type
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Moving Average Crossover", "RSI Divergence", "Bollinger Bands", "MACD"],
            key="strategy_type"
        )

        # Time period
        period = st.slider("Analysis Period (days)", 30, 365, 90, key="strategy_period")

        # Strategy parameters based on type
        if strategy_type == "Moving Average Crossover":
            fast_ma = st.slider("Fast MA", 5, 50, 10, key="fast_ma")
            slow_ma = st.slider("Slow MA", 20, 200, 50, key="slow_ma")

        elif strategy_type == "RSI Divergence":
            rsi_period = st.slider("RSI Period", 7, 21, 14, key="rsi_period")
            rsi_overbought = st.slider("Overbought Level", 65, 85, 70, key="rsi_overbought")
            rsi_oversold = st.slider("Oversold Level", 15, 35, 30, key="rsi_oversold")

        elif strategy_type == "Bollinger Bands":
            bb_period = st.slider("BB Period", 10, 50, 20, key="bb_period")


def show_cointegration_tester_tab():
    """
    Cointegration testing tool for pairs trading.
    """
    st.header("Cointegration Tester")

    col1, col2 = st.columns(2)

    with col1:
        asset1 = st.selectbox(
            "Asset 1",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
            key="cointegration_asset1"
        )

    with col2:
        asset2 = st.selectbox(
            "Asset 2",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
            key="cointegration_asset2"
        )

    period = st.slider("Analysis Period (days)", 30, 365, 90, key="cointegration_period")

    if st.button("Test Cointegration", key="test_cointegration"):
        try:
            # Load data for both assets
            data1 = get_binance_data(asset1, interval="1d", limit=period)
            data2 = get_binance_data(asset2, interval="1d", limit=period)

            if data1 is None or data2 is None:
                st.error("Failed to load data for one or both assets.")
                return

            # Convert to series
            prices1 = pd.Series(data1['close'].values, index=pd.to_datetime(data1['timestamp']))
            prices2 = pd.Series(data2['close'].values, index=pd.to_datetime(data2['timestamp']))

            # Normalize prices
            norm1 = prices1 / prices1.iloc[0]
            norm2 = prices2 / prices2.iloc[0]

            # Calculate spread
            spread = norm1 - norm2

            # Simple cointegration test (correlation)
            correlation = norm1.corr(norm2)

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Correlation", f"{correlation:.3f}")

            with col2:
                st.metric("Spread Mean", f"{spread.mean():.4f}")

            with col3:
                st.metric("Spread Std", f"{spread.std():.4f}")

            # Plot normalized prices and spread
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=norm1.index,
                y=norm1.values,
                name=asset1,
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=norm2.index,
                y=norm2.values,
                name=asset2,
                line=dict(color='red')
            ))

            fig.update_layout(
                title=f"Normalized Prices: {asset1} vs {asset2}",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=400
            )

            st.plotly_chart(fig, config={'responsive': True})

            # Spread chart
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=spread.index,
                y=spread.values,
                name="Spread",
                line=dict(color='green')
            ))

            fig2.add_hline(y=0, line_dash="dash", line_color="black")

            fig2.update_layout(
                title="Price Spread",
                xaxis_title="Date",
                yaxis_title="Spread",
                height=300
            )

            st.plotly_chart(fig2, config={'responsive': True})

        except Exception as e:
            st.error(f"Error testing cointegration: {str(e)}")


def show_custom_indicators():
    """
    Custom technical indicators builder.
    """
    st.header("Custom Indicators")
    st.info("Custom indicators feature coming soon!")


def show_portfolio_simulator():
    """
    Portfolio simulation tool.
    """
    st.header("Portfolio Simulator")
    st.info("Portfolio simulation feature coming soon!")

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual de Instruções - Laboratório Quant", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta página:
        Ferramentas avançadas para desenvolvimento e teste de estratégias quantitativas de trading.

        ### 📊 Abas e funcionalidades:
        - **Strategy Simulator**: Teste estratégias de trading com backtesting histórico
        - **Cointegration Tester**: Analise pares de ativos para estratégias de spread trading
        - **Custom Indicators**: Crie e teste indicadores técnicos personalizados
        - **Portfolio Simulator**: Simule performance de carteiras sob diferentes cenários

        ### 🔧 Como usar as ferramentas:
        1. **Simulator de Estratégias**: Selecione ativo, tipo de estratégia e parâmetros
        2. **Teste de Cointegração**: Escolha dois ativos e analise sua relação de longo prazo
        3. **Indicadores Customizados**: Construa indicadores usando dados históricos
        4. **Simulação de Portfólio**: Teste alocações sob diferentes condições de mercado

        ### 💡 Dicas importantes:
        - Use dados históricos suficientes para evitar overfitting
        - Considere custos de transação nas simulações
        - Teste estratégias em diferentes condições de mercado
        - Monitore métricas de risco além dos retornos

        ### ⚠️ Avisos:
        - Resultados passados não garantem performance futura
        - Sempre teste estratégias com dados out-of-sample
        - Considere diversificação e gerenciamento de risco

        *Última atualização: v2.0*
        ''')
