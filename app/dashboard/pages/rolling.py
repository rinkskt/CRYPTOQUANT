"""
Rolling Correlations Dashboard Page

This module provides the rolling correlations analysis page for the Streamlit dashboard.
Shows how correlations between assets change over time.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from app.dashboard.api_client import (
    get_assets, get_ohlcv_data, get_rolling_correlation
)
from app.analytics.rolling import latest_rolling_corr_matrix, rolling_correlation


def show_rolling_page():
    """
    Display the rolling correlations analysis page.
    """
    st.header("🔄 Correlações Rolling")

    # Sidebar controls
    st.sidebar.header("Configurações de Rolling")

    # Rolling window size
    window_sizes = [7, 14, 30, 60, 90, 180]
    window_days = st.sidebar.selectbox(
        "Janela Rolling (dias):",
        options=window_sizes,
        index=2,  # Default to 30 days
        help="Período usado para calcular correlações rolling"
    )

    # Asset selection
    assets = get_assets()
    if not assets:
        st.error("Não foi possível carregar os ativos.")
        return

    asset_symbols = [asset['symbol'] for asset in assets]

    # Multi-select for assets to analyze
    selected_assets = st.sidebar.multiselect(
        "Selecione ativos para análise:",
        options=asset_symbols,
        default=asset_symbols[:8] if len(asset_symbols) >= 8 else asset_symbols,
        help="Selecione os ativos para incluir na análise de correlação"
    )

    if len(selected_assets) < 2:
        st.warning("Selecione pelo menos 2 ativos para análise de correlação.")
        return

    # Load rolling correlation data
    with st.spinner("Carregando dados de correlação rolling..."):
        corr_data = get_rolling_correlation(window=window_days)

    if not corr_data or 'correlation_matrix' not in corr_data:
        st.error("Não foi possível carregar dados de correlação rolling.")
        st.info("Certifique-se de que o pipeline de analytics foi executado.")
        return

    # Extract correlation matrix
    corr_matrix = pd.DataFrame(corr_data['correlation_matrix'])

    # Filter to selected assets
    available_assets = [asset for asset in selected_assets if asset in corr_matrix.columns]
    if len(available_assets) < 2:
        st.warning("Dados insuficientes para os ativos selecionados.")
        return

    filtered_corr = corr_matrix.loc[available_assets, available_assets]

    # Main content
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Ativos Analisados", len(available_assets))

    with col2:
        avg_corr = filtered_corr.where(np.triu(np.ones_like(filtered_corr), k=1).astype(bool)).stack().mean()
        st.metric("Correlação Média", f"{avg_corr:.3f}")

    with col3:
        max_corr = filtered_corr.where(np.triu(np.ones_like(filtered_corr), k=1).astype(bool)).stack().max()
        st.metric("Correlação Máxima", f"{max_corr:.3f}")

    # Correlation heatmap
    st.subheader("Matriz de Correlação Atual")

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=filtered_corr.values,
        x=filtered_corr.columns,
        y=filtered_corr.index,
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=np.round(filtered_corr.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=f"Correlação Rolling ({window_days} dias) - {datetime.now().strftime('%Y-%m-%d')}",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=600
    )

    st.plotly_chart(fig, config={'responsive': True})

    # Correlation evolution over time
    st.subheader("Evolução das Correlações")

    # Get historical price data for selected assets
    price_data = {}
    for symbol in available_assets[:5]:  # Limit to 5 assets for performance
        data = get_ohlcv_data(symbol, limit=500)  # Last 500 days
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')[['close']].rename(columns={'close': symbol})
            price_data[symbol] = df

    if len(price_data) >= 2:
        # Merge price data
        merged_prices = pd.concat(price_data.values(), axis=1, join='inner')

        if not merged_prices.empty:
            # Calculate rolling correlations over time
            rolling_corr_series = {}

            # Calculate correlation between first two assets over time
            asset1, asset2 = available_assets[0], available_assets[1]

            if asset1 in merged_prices.columns and asset2 in merged_prices.columns:
                returns1 = merged_prices[asset1].pct_change()
                returns2 = merged_prices[asset2].pct_change()

                # Rolling correlation
                rolling_corr = returns1.rolling(window=window_days).corr(returns2)
                rolling_corr_series[f"{asset1} vs {asset2}"] = rolling_corr

            # Calculate correlation between first asset and others
            for asset in available_assets[2:4]:  # Limit to 3 pairs for readability
                if asset in merged_prices.columns:
                    returns_other = merged_prices[asset].pct_change()
                    rolling_corr = returns1.rolling(window=window_days).corr(returns_other)
                    rolling_corr_series[f"{asset1} vs {asset}"] = rolling_corr

            if rolling_corr_series:
                # Plot rolling correlations
                fig = go.Figure()

                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, (pair_name, corr_series) in enumerate(rolling_corr_series.items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=corr_series.index,
                        y=corr_series.values,
                        name=pair_name,
                        line=dict(color=color, width=2)
                    ))

                fig.update_layout(
                    title=f"Evolução da Correlação Rolling ({window_days} dias)",
                    xaxis_title="Data",
                    yaxis_title="Correlação",
                    height=400
                )

                st.plotly_chart(fig, config={'responsive': True})

    # Asset correlation ranking
    st.subheader("Ranking de Correlações")

    # Get upper triangle of correlation matrix (excluding diagonal)
    corr_pairs = []
    for i in range(len(filtered_corr.columns)):
        for j in range(i+1, len(filtered_corr.columns)):
            asset1 = filtered_corr.columns[i]
            asset2 = filtered_corr.columns[j]
            corr_value = filtered_corr.iloc[i, j]
            corr_pairs.append({
                'pair': f"{asset1} vs {asset2}",
                'correlation': corr_value,
                'abs_correlation': abs(corr_value)
            })

    if corr_pairs:
        pairs_df = pd.DataFrame(corr_pairs)
        pairs_df = pairs_df.sort_values('abs_correlation', ascending=False)

        # Display top correlations
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Pares Mais Correlacionados:**")
            top_positive = pairs_df[pairs_df['correlation'] > 0].head(5)
            if not top_positive.empty:
                for _, row in top_positive.iterrows():
                    st.write(f"📈 {row['pair']}: {row['correlation']:.3f}")

        with col2:
            st.write("**Pares Menos Correlacionados:**")
            top_negative = pairs_df[pairs_df['correlation'] < 0].head(5)
            if not top_negative.empty:
                for _, row in top_negative.iterrows():
                    st.write(f"📉 {row['pair']}: {row['correlation']:.3f}")

    # Correlation stability analysis
    st.subheader("Estabilidade das Correlações")

    # Calculate correlation statistics
    corr_values = filtered_corr.where(np.triu(np.ones_like(filtered_corr), k=1).astype(bool)).stack()

    if not corr_values.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Correlação Média", f"{corr_values.mean():.3f}")

        with col2:
            st.metric("Desvio Padrão", f"{corr_values.std():.3f}")

        with col3:
            st.metric("Correlação Mínima", f"{corr_values.min():.3f}")

        with col4:
            st.metric("Correlação Máxima", f"{corr_values.max():.3f}")

        # Distribution of correlations
        st.subheader("Distribuição das Correlações")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=corr_values.values,
            nbinsx=20,
            name='Correlações',
            marker_color='lightblue'
        ))

        fig.update_layout(
            title="Distribuição dos Valores de Correlação",
            xaxis_title="Correlação",
            yaxis_title="Frequência",
            height=300
        )

        st.plotly_chart(fig, config={'responsive': True})

    # Insights and recommendations
    st.subheader("💡 Insights")

    if not corr_values.empty:
        avg_corr = corr_values.mean()
        corr_std = corr_values.std()

        if avg_corr > 0.7:
            st.info("📊 **Mercado Altamente Correlacionado**: Alta correlação média sugere movimento de mercado sincronizado. Considere diversificação em diferentes classes de ativos.")
        elif avg_corr > 0.3:
            st.info("⚖️ **Correlação Moderada**: Correlações mistas oferecem oportunidades de diversificação dentro do universo de cripto.")
        else:
            st.info("🔀 **Baixa Correlação**: Boa diversificação possível. Monitore mudanças nas correlações ao longo do tempo.")

        if corr_std > 0.3:
            st.warning("⚠️ **Correlações Voláteis**: Alta variabilidade nas correlações indica instabilidade no mercado. Reavalie exposições regularmente.")
        else:
            st.success("✅ **Correlações Estáveis**: Relacionamentos relativamente estáveis facilitam planejamento de portfólio de longo prazo.")
