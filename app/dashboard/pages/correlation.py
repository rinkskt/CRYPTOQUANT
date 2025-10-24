"""
Correlation and Cointegration Analysis Page

This module provides the correlation and cointegration analysis page.
Shows pair selection, cointegration charts, and spread analysis.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from app.dashboard.api_client import (
    get_assets, get_ohlcv_data, get_cointegrated_pairs
)
from app.dashboard.components.cointeg_chart import create_cointegration_chart
from app.analytics.cointegration import test_cointegration
from app.analytics.spread import calculate_spread
from app.analytics.stats import compute_zscore


def show_correlation_page():
    """
    Display the correlation and cointegration analysis page.
    """
    st.header("🔗 Análise de Correlação e Cointegração")

    # Sidebar controls
    st.sidebar.header("Controles de Análise")

    # Asset selection
    assets = get_assets()
    if not assets:
        st.error("Não foi possível carregar os ativos.")
        return

    asset_symbols = [asset['symbol'] for asset in assets]

    col1, col2 = st.sidebar.columns(2)

    with col1:
        asset1 = st.selectbox(
            "Ativo 1:",
            options=asset_symbols,
            index=0 if asset_symbols else 0
        )

    with col2:
        asset2 = st.selectbox(
            "Ativo 2:",
            options=asset_symbols,
            index=1 if len(asset_symbols) > 1 else 0
        )

    # Cointegration filter
    show_only_cointegrated = st.sidebar.checkbox(
        "Mostrar apenas pares cointegrados (p < 0.05)",
        value=True
    )

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = st.sidebar.date_input(
        "Período:",
        value=(start_date.date(), end_date.date())
    )

    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())

    # Main content - two columns layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Seletor de Pares")

        # Load pair data
        with st.spinner("Carregando análise de pares..."):
            pair_data = load_pair_analysis(asset1, asset2, start_date, end_date)

        if pair_data:
            # Display pair metrics
            st.metric("Correlação", f"{pair_data['correlation']:.3f}")
            st.metric("P-value Cointegração", f"{pair_data['p_value']:.4f}")
            st.metric("Beta", f"{pair_data['beta']:.4f}")
            st.metric("Half-life", f"{pair_data['half_life']:.1f} períodos")
            st.metric("Z-score Atual", f"{pair_data['current_zscore']:.2f}")

            # Cointegration status
            if pair_data['p_value'] < 0.05:
                st.success("✅ Pares Cointegrados")
            else:
                st.warning("❌ Pares Não Cointegrados")

        # Cointegrated pairs table
        st.subheader("Pares Cointegrados")

        cointegrated_pairs = get_cointegrated_pairs()
        if cointegrated_pairs:
            # Filter if requested
            if show_only_cointegrated:
                filtered_pairs = [p for p in cointegrated_pairs if p.get('p_value', 1) < 0.05]
            else:
                filtered_pairs = cointegrated_pairs

            if filtered_pairs:
                # Create DataFrame for display
                pairs_df = pd.DataFrame(filtered_pairs)
                pairs_df = pairs_df[[
                    'asset_x', 'asset_y', 'correlation', 'p_value',
                    'beta', 'half_life', 'latest_zscore'
                ]].copy()

                pairs_df.columns = [
                    'Ativo X', 'Ativo Y', 'Correlação', 'P-value',
                    'Beta', 'Half-life', 'Z-score'
                ]

                # Format
                pairs_df['Correlação'] = pairs_df['Correlação'].apply(lambda x: f"{x:.3f}")
                pairs_df['P-value'] = pairs_df['P-value'].apply(lambda x: f"{x:.4f}")
                pairs_df['Beta'] = pairs_df['Beta'].apply(lambda x: f"{x:.4f}")
                pairs_df['Half-life'] = pairs_df['Half-life'].apply(lambda x: f"{x:.1f}")
                pairs_df['Z-score'] = pairs_df['Z-score'].apply(lambda x: f"{x:.2f}")

                st.dataframe(pairs_df, width='stretch')

                # Button to visualize selected pair
                if st.button("Visualizar Par Selecionado"):
                    # This would trigger visualization in right column
                    st.rerun()
            else:
                st.info("Nenhum par cointegrado encontrado com os filtros atuais.")

    with right_col:
        st.subheader("Gráfico de Cointegração")

        if pair_data and pair_data['price_data'] is not None:
            # Create cointegration chart
            fig = create_cointegration_chart(
                pair_data['price_data'],
                asset1, asset2,
                pair_data['beta'],
                pair_data['current_zscore']
            )
            st.plotly_chart(fig, config={'responsive': True})

            # Spread analysis
            st.subheader("Análise do Spread")

            spread_data = pair_data['spread_data']
            if spread_data is not None:
                # Z-score chart
                fig_zscore = go.Figure()

                fig_zscore.add_trace(go.Scatter(
                    x=spread_data.index,
                    y=spread_data['zscore'],
                    name='Z-Score',
                    line=dict(color='blue', width=2)
                ))

                # Add threshold lines
                fig_zscore.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
                fig_zscore.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")
                fig_zscore.add_hline(y=1, line_dash="dot", line_color="orange", annotation_text="+1σ")
                fig_zscore.add_hline(y=-1, line_dash="dot", line_color="orange", annotation_text="-1σ")
                fig_zscore.add_hline(y=0, line_dash="solid", line_color="gray")

                fig_zscore.update_layout(
                    title="Z-Score do Spread",
                    xaxis_title="Data",
                    yaxis_title="Z-Score",
                    height=300
                )

                st.plotly_chart(fig_zscore, config={'responsive': True})

                # Trading signals
                current_z = pair_data['current_zscore']
                if abs(current_z) >= 2:
                    signal = "🚨 FORTE"
                    action = "SHORT spread" if current_z > 0 else "LONG spread"
                elif abs(current_z) >= 1.5:
                    signal = "⚠️ MODERADO"
                    action = "SHORT spread" if current_z > 0 else "LONG spread"
                else:
                    signal = "✅ NEUTRO"
                    action = "AGUARDAR"

                st.info(f"**Sinal Atual: {signal}**\n\n**Ação Sugerida: {action}**\n\n**Z-score: {current_z:.2f}**")

        else:
            st.info("Selecione dois ativos diferentes para visualizar a análise de cointegração.")

        # Rolling correlation
        st.subheader("Correlação Rolling")

        if pair_data and pair_data['price_data'] is not None:
            # Calculate rolling correlation
            window = 30  # 30-day rolling
            returns1 = pair_data['price_data'][asset1].pct_change()
            returns2 = pair_data['price_data'][asset2].pct_change()

            rolling_corr = returns1.rolling(window=window).corr(returns2)

            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                name=f'Correlação Rolling ({window} dias)',
                line=dict(color='green', width=2)
            ))

            fig_corr.update_layout(
                title=f"Correlação Rolling: {asset1} vs {asset2}",
                xaxis_title="Data",
                yaxis_title="Correlação",
                height=250
            )

            st.plotly_chart(fig_corr, config={'responsive': True})

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual de Instruções - Correlações", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta página:
        Analisar relacionamentos entre pares de ativos através de correlação e cointegração para estratégias de pairs trading.

        ### 📊 Como interpretar os gráficos:
        - **Gráfico de Cointegração**: Mostra preços normalizados e linha de regressão entre dois ativos
        - **Z-Score do Spread**: Indica desvios do spread em relação à média (linhas em ±2σ indicam oportunidades)
        - **Correlação Rolling**: Mostra como a correlação evolui ao longo do tempo

        ### 🔧 Como usar as ferramentas:
        1. Selecione dois ativos diferentes nos controles laterais
        2. Configure o período de análise desejado
        3. Analise as métricas de cointegração (p-value < 0.05 indica cointegração)
        4. Observe o Z-score atual para sinais de trading
        5. Use a tabela de pares cointegrados para descobrir oportunidades

        ### 💡 Dicas importantes:
        - **P-value < 0.05**: Indica que os ativos são cointegrados (relação estatisticamente significativa)
        - **Z-score > +2**: Spread está alto - considere vender o ativo Y e comprar o X
        - **Z-score < -2**: Spread está baixo - considere comprar o ativo Y e vender o X
        - **Half-life**: Tempo médio para o spread reverter à média (menor = melhor)
        - Monitore a correlação rolling para identificar mudanças nos relacionamentos

        ### ⚠️ Riscos do Pairs Trading:
        - Cointegração pode quebrar durante eventos extremos
        - Considere custos de transação bid-ask spreads
        - Use stop-loss adequados para spreads
        - Teste estratégias em diferentes regimes de mercado

        *Última atualização: v2.0*
        ''')


def load_pair_analysis(asset1, asset2, start_date, end_date):
    """
    Load and analyze pair data.
    """
    if asset1 == asset2:
        return None

    try:
        # Get price data
        data1 = get_ohlcv_data(asset1, start_date=start_date, end_date=end_date, limit=500)
        data2 = get_ohlcv_data(asset2, start_date=start_date, end_date=end_date, limit=500)

        if not data1 or not data2:
            return None

        # Convert to DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        if df1.empty or df2.empty:
            return None

        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'])

        # Merge on timestamp
        price_data = pd.merge(
            df1[['timestamp', 'close']].rename(columns={'close': asset1}),
            df2[['timestamp', 'close']].rename(columns={'close': asset2}),
            on='timestamp',
            how='inner'
        ).set_index('timestamp')

        if price_data.empty:
            return None

        # Calculate cointegration
        coint_result = test_cointegration(price_data[asset1], price_data[asset2])

        # Calculate spread and z-score
        beta = coint_result.get('beta', 1.0)
        spread = price_data[asset2] - beta * price_data[asset1]
        zscore_series = compute_zscore(spread)

        return {
            'price_data': price_data,
            'correlation': price_data[asset1].corr(price_data[asset2]),
            'p_value': coint_result.get('p_value', 1.0),
            'beta': beta,
            'half_life': coint_result.get('half_life', 0),
            'current_zscore': zscore_series.iloc[-1] if not zscore_series.empty else 0,
            'spread_data': pd.DataFrame({
                'spread': spread,
                'zscore': zscore_series
            })
        }

    except Exception as e:
        st.error(f"Erro ao carregar análise de pares: {e}")
        return None
