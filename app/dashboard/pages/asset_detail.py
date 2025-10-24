"""
Individual Asset Analysis Page

This module provides detailed analysis for individual assets.
Shows price charts, z-score history, correlations, and cointegration pairs.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from app.dashboard.api_client import (
    get_assets, get_ohlcv_data, get_analytics_data
)
from app.dashboard.components.zscore_plot import create_zscore_plot
from app.analytics.stats import compute_zscore, compute_rolling_stats


def show_asset_detail_page():
    """
    Display the individual asset analysis page.
    """
    st.header("📈 Análise Individual do Ativo")

    # Sidebar controls
    st.sidebar.header("Configurações do Ativo")

    # Asset selection
    assets = get_assets()
    if not assets:
        st.error("Não foi possível carregar os ativos.")
        return

    asset_symbols = [asset['symbol'] for asset in assets]
    selected_asset = st.sidebar.selectbox(
        "Selecione o Ativo:",
        options=asset_symbols,
        index=0
    )

    # Timeframe selection
    timeframes = {
        "1 Semana": 7,
        "1 Mês": 30,
        "3 Meses": 90,
        "6 Meses": 180,
        "1 Ano": 365
    }

    selected_timeframe = st.sidebar.selectbox(
        "Período:",
        options=list(timeframes.keys()),
        index=2  # Default to 3 months
    )

    days = timeframes[selected_timeframe]

    # Load asset data
    with st.spinner(f"Carregando dados de {selected_asset}..."):
        asset_data = load_asset_data(selected_asset, days)

    if not asset_data:
        st.error(f"Não foi possível carregar dados para {selected_asset}")
        return

    # Header with key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = asset_data['price_data']['close'].iloc[-1]
        st.metric("Preço Atual", f"${current_price:.4f}")

    with col2:
        price_change = asset_data['price_data']['close'].iloc[-1] - asset_data['price_data']['close'].iloc[-2]
        change_pct = (price_change / asset_data['price_data']['close'].iloc[-2]) * 100
        st.metric("Variação 24h", f"{change_pct:+.2f}%", delta=f"${price_change:+.4f}")

    with col3:
        volume = asset_data['price_data']['volume'].iloc[-1]
        st.metric("Volume 24h", f"{volume/1e6:.1f}M")

    with col4:
        zscore = asset_data['zscore_current']
        st.metric("Z-score Atual", f"{zscore:.2f}")

    # Main chart - Price with envelopes
    st.subheader("Gráfico de Preço + Médias + Envelope")

    fig = create_price_chart(asset_data['price_data'], selected_asset)
    st.plotly_chart(fig, config={'responsive': True})

    # Z-score history
    st.subheader("Histórico do Z-score")

    zscore_fig = create_zscore_plot(asset_data['zscore_history'])
    st.plotly_chart(zscore_fig, config={'responsive': True})

    # Correlations section
    st.subheader("Correlação com BTC e ETH")

    corr_data = get_asset_correlations(selected_asset)
    if corr_data:
        col1, col2 = st.columns(2)

        with col1:
            if 'BTC/USDT' in corr_data:
                btc_corr = corr_data['BTC/USDT']
                st.metric("Correlação com BTC", f"{btc_corr:.3f}")

                # Correlation chart with BTC
                fig_btc = create_correlation_chart(
                    asset_data['price_data']['close'],
                    corr_data.get('btc_prices', pd.Series()),
                    "BTC/USDT",
                    selected_asset
                )
                st.plotly_chart(fig_btc, config={'responsive': True})

        with col2:
            if 'ETH/USDT' in corr_data:
                eth_corr = corr_data['ETH/USDT']
                st.metric("Correlação com ETH", f"{eth_corr:.3f}")

                # Correlation chart with ETH
                fig_eth = create_correlation_chart(
                    asset_data['price_data']['close'],
                    corr_data.get('eth_prices', pd.Series()),
                    "ETH/USDT",
                    selected_asset
                )
                st.plotly_chart(fig_eth, config={'responsive': True})

    # Cointegration pairs
    st.subheader("Pares Cointegrados")

    coint_pairs = get_cointegrated_pairs_for_asset(selected_asset)
    if coint_pairs:
        # Display as table
        pairs_df = pd.DataFrame(coint_pairs)
        pairs_df = pairs_df[['pair_asset', 'correlation', 'p_value', 'beta', 'half_life', 'zscore']].copy()

        pairs_df.columns = ['Par', 'Correlação', 'P-value', 'Beta', 'Half-life', 'Z-score']

        # Format
        pairs_df['Correlação'] = pairs_df['Correlação'].apply(lambda x: f"{x:.3f}")
        pairs_df['P-value'] = pairs_df['P-value'].apply(lambda x: f"{x:.4f}")
        pairs_df['Beta'] = pairs_df['Beta'].apply(lambda x: f"{x:.4f}")
        pairs_df['Half-life'] = pairs_df['Half-life'].apply(lambda x: f"{x:.1f}")
        pairs_df['Z-score'] = pairs_df['Z-score'].apply(lambda x: f"{x:.2f}")

        st.dataframe(pairs_df, width='stretch')

        # Button to analyze specific pair
        if st.button("Analisar Par Selecionado"):
            st.info("Funcionalidade será implementada - redirecionar para página de correlação")
    else:
        st.info("Nenhum par cointegrado encontrado para este ativo.")

    # Technical analysis summary
    st.subheader("Resumo da Análise Técnica")

    # Calculate some technical indicators
    tech_summary = calculate_technical_summary(asset_data['price_data'])

    col1, col2, col3 = st.columns(3)

    with col1:
        rsi = tech_summary.get('rsi', 50)
        rsi_status = "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Neutro"
        st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)

    with col2:
        volatility = tech_summary.get('volatility', 0)
        st.metric("Volatilidade", f"{volatility:.2f}%")

    with col3:
        trend = tech_summary.get('trend', 'Lateral')
        st.metric("Tendência", trend)


def load_asset_data(asset_symbol, days):
    """
    Load comprehensive asset data.
    """
    try:
        # Get OHLCV data
        data = get_ohlcv_data(asset_symbol, limit=days)
        if not data:
            return None

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Calculate z-score
        zscore_series = compute_zscore(df['close'])

        return {
            'price_data': df,
            'zscore_history': zscore_series,
            'zscore_current': zscore_series.iloc[-1] if not zscore_series.empty else 0
        }

    except Exception as e:
        st.error(f"Erro ao carregar dados do ativo: {e}")
        return None


def create_price_chart(price_data, asset_symbol):
    """
    Create price chart with moving averages and envelopes.
    """
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name='Price'
    ))

    # Moving averages
    ma20 = price_data['close'].rolling(window=20).mean()
    ma50 = price_data['close'].rolling(window=50).mean()

    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=ma20,
        name='MA20',
        line=dict(color='orange', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=ma50,
        name='MA50',
        line=dict(color='purple', width=1)
    ))

    # Bollinger Bands (simplified envelope)
    std = price_data['close'].rolling(window=20).std()
    upper_band = ma20 + (std * 2)
    lower_band = ma20 - (std * 2)

    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=upper_band,
        name='Upper Band',
        line=dict(color='gray', width=1, dash='dot'),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=lower_band,
        name='Lower Band',
        line=dict(color='gray', width=1, dash='dot'),
        opacity=0.5
    ))

    fig.update_layout(
        title=f"{asset_symbol} - Preço com Médias Móveis",
        xaxis_title="Data",
        yaxis_title="Preço (USDT)",
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


def get_asset_correlations(asset_symbol):
    """
    Get correlations with major assets (BTC, ETH).
    """
    try:
        # Get price data for asset and benchmarks
        asset_data = get_ohlcv_data(asset_symbol, limit=90)
        btc_data = get_ohlcv_data("BTC/USDT", limit=90)
        eth_data = get_ohlcv_data("ETH/USDT", limit=90)

        if not asset_data or not btc_data or not eth_data:
            return None

        # Convert to series
        asset_prices = pd.Series([d['close'] for d in asset_data])
        btc_prices = pd.Series([d['close'] for d in btc_data])
        eth_prices = pd.Series([d['close'] for d in eth_data])

        # Calculate correlations
        btc_corr = asset_prices.corr(btc_prices)
        eth_corr = asset_prices.corr(eth_prices)

        return {
            'BTC/USDT': btc_corr,
            'ETH/USDT': eth_corr,
            'btc_prices': btc_prices,
            'eth_prices': eth_prices
        }

    except Exception as e:
        st.error(f"Erro ao calcular correlações: {e}")
        return None


def create_correlation_chart(asset_prices, benchmark_prices, benchmark_symbol, asset_symbol):
    """
    Create correlation visualization chart.
    """
    # Normalize prices
    asset_norm = asset_prices / asset_prices.iloc[0]
    benchmark_norm = benchmark_prices / benchmark_prices.iloc[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=asset_norm.index,
        y=asset_norm.values,
        name=asset_symbol,
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=benchmark_norm.index,
        y=benchmark_norm.values,
        name=benchmark_symbol,
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f"Correlação: {asset_symbol} vs {benchmark_symbol}",
        xaxis_title="Data",
        yaxis_title="Preço Normalizado",
        height=300
    )

    return fig


def get_cointegrated_pairs_for_asset(asset_symbol):
    """
    Get cointegrated pairs for the selected asset.
    """
    # This would typically come from analytics API
    # For now, return mock data
    return [
        {
            'pair_asset': 'ADA/USDT',
            'correlation': 0.85,
            'p_value': 0.02,
            'beta': 1.15,
            'half_life': 12.5,
            'zscore': 1.8
        },
        {
            'pair_asset': 'DOT/USDT',
            'correlation': 0.78,
            'p_value': 0.03,
            'beta': 0.95,
            'half_life': 15.2,
            'zscore': -0.5
        }
    ]


def calculate_technical_summary(price_data):
    """
    Calculate technical analysis summary.
    """
    try:
        close = price_data['close']

        # RSI calculation (simplified)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50

        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * 100 * np.sqrt(365)  # Annualized

        # Trend (simplified)
        ma50 = close.rolling(window=50).mean()
        ma200 = close.rolling(window=200).mean()

        if len(close) < 200:
            trend = "Dados Insuficientes"
        elif close.iloc[-1] > ma50.iloc[-1] and ma50.iloc[-1] > ma200.iloc[-1]:
            trend = "Alta"
        elif close.iloc[-1] < ma50.iloc[-1] and ma50.iloc[-1] < ma200.iloc[-1]:
            trend = "Baixa"
        else:
            trend = "Lateral"

        return {
            'rsi': current_rsi,
            'volatility': volatility,
            'trend': trend
        }

    except Exception as e:
        return {
            'rsi': 50,
            'volatility': 0,
            'trend': 'N/A'
        }

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual de Instruções - Detalhes do Ativo", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta página:
        Fornecer análise técnica detalhada de ativos individuais, incluindo gráficos de preço,
        indicadores técnicos e relacionamentos com outros ativos.

        ### 📊 Como interpretar os gráficos:
        - **Gráfico de Preço**: Candlestick com médias móveis (MA20, MA50) e Bandas de Bollinger
        - **Z-Score**: Mede desvios do preço em relação à média histórica (valores extremos indicam reversões)
        - **Correlação**: Mostra relacionamento com BTC e ETH (valores próximos de 1 = alta correlação)
        - **Pares Cointegrados**: Ativos com relacionamento estatístico estável para pairs trading

        ### 🔧 Como usar as ferramentas:
        1. Selecione o ativo desejado no painel lateral
        2. Escolha o período de análise (1 semana a 1 ano)
        3. Analise os indicadores técnicos no resumo
        4. Verifique correlações com ativos principais
        5. Explore pares cointegrados para oportunidades de trading

        ### 💡 Dicas importantes:
        - **Z-score > 2**: Preço muito acima da média histórica (possível reversão baixista)
        - **Z-score < -2**: Preço muito abaixo da média histórica (possível reversão altista)
        - **RSI > 70**: Ativo sobrecomprado (considere vender)
        - **RSI < 30**: Ativo sobrevendido (considere comprar)
        - **Correlação alta com BTC**: Ativo segue tendências do mercado geral

        ### ⚠️ Riscos da análise técnica:
        - Indicadores passados não garantem performance futura
        - Use sempre stop-loss para proteger posições
        - Considere análise fundamental junto com técnica
        - Monitore liquidez e volume do ativo

        *Última atualização: v2.0*
        ''')
