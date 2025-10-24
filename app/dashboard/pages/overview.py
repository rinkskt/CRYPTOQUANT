"""
Market Overview Dashboard Page

This module provides the market overview page for the quantitative crypto dashboard.
Shows global market indicators, portfolio summary, and correlation heatmap.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from ..api_client import (
    get_assets, get_ohlcv_data, get_correlation_matrix,
    get_analytics_data, get_cointegrated_pairs
)
from ..components.heatmap import create_correlation_heatmap
from ...analytics.stats import compute_zscore


def show_overview_page():
    """
    Display the market overview page.
    """
    st.header("📊 Visão Geral do Mercado")

    # Sidebar controls
    st.sidebar.header("Configurações da Visão Geral")

    # Refresh button
    if st.sidebar.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.rerun()

    # Market indicators section
    st.subheader("Indicadores Globais do Mercado")

    # Get market data
    market_data = get_market_indicators()

    if market_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            btc_dominance = market_data.get('btc_dominance', 0)
            st.metric("BTC Dominance", f"{btc_dominance:.1f}%")

        with col2:
            total_volume = market_data.get('total_volume_24h', 0)
            st.metric("Volume Total 24h", f"${total_volume/1e9:.1f}B")

        with col3:
            avg_return = market_data.get('avg_return_top10', 0)
            st.metric("Retorno Médio Top 10", f"{avg_return:.2f}%")

        with col4:
            volatility = market_data.get('market_volatility', 0)
            st.metric("Volatilidade", f"{volatility:.2f}%")

    # Correlation heatmap
    st.subheader("Mapa de Calor de Correlação")

    corr_data = get_correlation_matrix()
    if corr_data:
        # Use component for heatmap
        fig = create_correlation_heatmap(corr_data)
        st.plotly_chart(fig, config={'responsive': True})

        # Click interaction for cointegration
        st.info("💡 Clique em um par de ativos para ver análise de cointegração")

    # Portfolio summary
    st.subheader("Resumo da Carteira")

    # Get portfolio data (simplified - in real implementation would get from user portfolio)
    portfolio_summary = get_portfolio_summary()

    if portfolio_summary:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl = portfolio_summary.get('total_pnl', 0)
            st.metric("P&L Total", f"${pnl:,.2f}")

        with col2:
            beta = portfolio_summary.get('portfolio_beta', 0)
            st.metric("Beta da Carteira", f"{beta:.2f}")

        with col3:
            avg_zscore = portfolio_summary.get('avg_zscore_deviation', 0)
            st.metric("Desvio Médio Z-score", f"{avg_zscore:.2f}")

        with col4:
            cointegration = portfolio_summary.get('avg_cointegration', 0)
            st.metric("Cointegração Média", f"{cointegration:.2f}")

    # Top assets ranking
    st.subheader("Ranking de Moedas")

    assets_data = get_top_assets_ranking()
    if assets_data:
        # Display as table
        df = pd.DataFrame(assets_data)
        df = df[['symbol', 'return_24h', 'volatility', 'zscore', 'volume']]

        # Format columns
        df['return_24h'] = df['return_24h'].apply(lambda x: f"{x:.2f}%")
        df['volatility'] = df['volatility'].apply(lambda x: f"{x:.2f}%")
        df['zscore'] = df['zscore'].apply(lambda x: f"{x:.2f}")
        df['volume'] = df['volume'].apply(lambda x: f"${x/1e6:.1f}M")

        st.dataframe(df, width='stretch')

        st.info("💡 Clique em uma moeda para ir para análise individual")

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual de Instruções - Visão Geral do Mercado", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta página:
        Fornecer uma visão abrangente do mercado de criptomoedas, incluindo indicadores globais,
        correlações entre ativos e resumo da carteira.

        ### 📊 Como interpretar os gráficos:
        - **Indicadores Globais**: Mostra métricas agregadas do mercado como dominância do BTC, volume total e volatilidade
        - **Mapa de Calor de Correlação**: Valores próximos a 1 indicam forte correlação positiva entre ativos
        - **Resumo da Carteira**: Métricas de performance e risco da sua carteira atual
        - **Ranking de Moedas**: Classificação dos ativos por volume e outras métricas

        ### 🔧 Como usar as ferramentas:
        1. Use o botão "🔄 Atualizar Dados" para obter informações mais recentes
        2. Clique em pares de ativos no mapa de correlação para análise de cointegração
        3. Clique em moedas do ranking para análise detalhada individual
        4. Configure filtros no painel lateral para personalizar as visualizações

        ### 💡 Dicas importantes:
        - Monitore a volatilidade do mercado para decisões de timing
        - Ativos com baixa correlação podem diversificar melhor sua carteira
        - Z-scores extremos podem indicar oportunidades de reversão
        - Use o beta da carteira para entender exposição ao mercado

        *Última atualização: v2.0*
        ''')


def get_market_indicators():
    """
    Calculate global market indicators.
    """
    try:
        # Get all assets
        assets = get_assets()
        if not assets:
            return None

        # Get recent OHLCV data for calculations
        market_stats = {
            'btc_dominance': 0,
            'total_volume_24h': 0,
            'avg_return_top10': 0,
            'market_volatility': 0
        }

        volumes = []
        returns = []
        btc_volume = 0

        for asset in assets[:20]:  # Top 20 assets
            data = get_ohlcv_data(asset['symbol'], limit=2)  # Last 2 days
            if data and len(data) >= 2:
                # Calculate 24h return
                current_price = data[-1]['close']
                prev_price = data[-2]['close']
                ret_24h = ((current_price - prev_price) / prev_price) * 100
                returns.append(ret_24h)

                # Volume
                volume_24h = data[-1]['volume']
                volumes.append(volume_24h)

                if asset['symbol'] == 'BTC/USDT':
                    btc_volume = volume_24h

        if volumes and returns:
            market_stats['total_volume_24h'] = sum(volumes)
            market_stats['avg_return_top10'] = np.mean(sorted(returns, reverse=True)[:10])
            market_stats['market_volatility'] = np.std(returns)

            # BTC dominance approximation
            if btc_volume > 0:
                market_stats['btc_dominance'] = (btc_volume / sum(volumes)) * 100

        return market_stats

    except Exception as e:
        st.error(f"Erro ao calcular indicadores de mercado: {e}")
        return None


def get_portfolio_summary():
    """
    Get portfolio summary metrics.
    """
    # This is a placeholder - in real implementation would get from user's portfolio
    # For now, return mock data
    return {
        'total_pnl': 1250.75,
        'portfolio_beta': 1.15,
        'avg_zscore_deviation': 0.85,
        'avg_cointegration': 0.72
    }


def get_top_assets_ranking():
    """
    Get ranking of top assets by various metrics.
    """
    try:
        assets = get_assets()
        if not assets:
            return None

        ranking_data = []

        for asset in assets[:15]:  # Top 15 assets
            data = get_ohlcv_data(asset['symbol'], limit=30)  # Last 30 days
            if data and len(data) >= 2:
                # Calculate metrics
                prices = [d['close'] for d in data]
                volumes = [d['volume'] for d in data]

                # 24h return
                ret_24h = ((prices[-1] - prices[-2]) / prices[-2]) * 100

                # Volatility (30-day)
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * 100

                # Z-score (simplified)
                zscore = compute_zscore(pd.Series(prices)).iloc[-1]

                # Volume
                avg_volume = np.mean(volumes)

                ranking_data.append({
                    'symbol': asset['symbol'],
                    'return_24h': ret_24h,
                    'volatility': volatility,
                    'zscore': zscore,
                    'volume': avg_volume
                })

        # Sort by volume
        ranking_data.sort(key=lambda x: x['volume'], reverse=True)
        return ranking_data

    except Exception as e:
        st.error(f"Erro ao calcular ranking de ativos: {e}")
        return None
