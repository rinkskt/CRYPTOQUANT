"""
Portfolio Risk Module

Análise e visualização de risco do portfólio.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from app.analytics.portfolio.risk import calculate_risk_contribution
from app.analytics.stats import compute_zscore
from .portfolio_form import load_portfolio, PortfolioPosition
from ...api_client import get_ohlcv_data
from app.analytics.portfolio import get_portfolio_data

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcula Value at Risk (VaR) do portfólio.
    """
    return -np.percentile(returns, (1 - confidence) * 100)

def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcula Expected Shortfall (ES) do portfólio.
    """
    var = calculate_var(returns, confidence)
    return -returns[returns <= -var].mean()

def create_risk_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Cria heatmap de correlação interativo.
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Matriz de Correlação",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=500
    )

    return fig

def create_risk_contribution_chart(risk_contrib: Dict[str, float]) -> go.Figure:
    """
    Cria gráfico de pizza mostrando contribuição de risco por ativo.
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_contrib.keys()),
        values=list(risk_contrib.values()),
        hole=.3,
        textinfo='label+percent',
        textposition='inside'
    )])

    fig.update_layout(
        title="Contribuição de Risco por Ativo",
        showlegend=False,
        height=400
    )

    return fig

def show_risk_tab():
    """
    Mostra a aba de análise de risco do portfólio.
    """
    st.header("⚖️ Análise de Risco")

    # Carrega dados do portfólio
    portfolio = load_portfolio()
    if not portfolio:
        st.warning("Nenhuma posição encontrada. Adicione ativos ao seu portfólio primeiro.")
        return

    # Configurações
    col1, col2 = st.columns(2)
    with col1:
        lookback_days = st.selectbox(
            "Período de Análise",
            options=[30, 90, 180, 365],
            index=1,
            format_func=lambda x: f"Últimos {x} dias",
            key="risk_lookback_days"
        )

    with col2:
        confidence = st.slider(
            "Nível de Confiança",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f",
            key="risk_confidence"
        )

    # Prepara dados
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    symbols = [pos.symbol for pos in portfolio]
    weights = {pos.symbol: pos.qty * pos.price_entry for pos in portfolio}
    total_value = sum(weights.values())
    weights = {k: v/total_value for k, v in weights.items()}

    # Carrega dados históricos usando o novo módulo
    try:
        interval = "1d"
        limit = lookback_days

        portfolio_raw = get_portfolio_data(symbols, interval=interval, limit=limit)

        # Converte para DataFrame com timestamps como índice
        portfolio_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in portfolio_raw:
                df = portfolio_raw[symbol].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                portfolio_data[symbol] = df['close']

        if portfolio_data.empty:
            st.error("Não foi possível carregar dados históricos.")
            return

    except Exception as e:
        st.error(f"Erro ao carregar dados históricos: {str(e)}")
        return

    # Calcula retornos
    returns = portfolio_data.pct_change().dropna()
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)

    # Métricas de risco principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        volatility = portfolio_returns.std() * np.sqrt(252)
        st.metric(
            "Volatilidade Anualizada",
            f"{volatility*100:.2f}%"
        )

    with col2:
        var = calculate_var(portfolio_returns, confidence)
        st.metric(
            f"VaR ({confidence*100:.0f}%)",
            f"{var*100:.2f}%"
        )

    with col3:
        es = calculate_expected_shortfall(portfolio_returns, confidence)
        st.metric(
            f"Expected Shortfall",
            f"{es*100:.2f}%"
        )

    with col4:
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
        st.metric(
            "Máximo Drawdown",
            f"{max_drawdown*100:.2f}%"
        )

    # Matriz de correlação
    st.subheader("Correlações e Risco")
    col1, col2 = st.columns(2)

    with col1:
        corr_matrix = returns.corr()
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True
        }
        st.plotly_chart(
            create_risk_heatmap(corr_matrix),
            config=config,
            use_container_width=True
        )

    with col2:
        risk_contrib = calculate_risk_contribution(returns, weights)
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True
        }
        st.plotly_chart(
            create_risk_contribution_chart(risk_contrib),
            config=config,
            use_container_width=True
        )

    # Análise de drawdown
    st.subheader("Análise de Drawdown")
    cumulative_returns = (1 + portfolio_returns).cumprod()
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Drawdown do Portfólio",
        xaxis_title="Data",
        yaxis_title="Drawdown (%)",
        yaxis=dict(tickformat='.1f'),
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, config={'responsive': True})

    # Z-scores dos ativos
    st.subheader("Z-scores dos Ativos")
    zscore_window = st.slider(
        "Janela para Z-score (dias)",
        min_value=5,
        max_value=60,
        value=20
    )

    zscores = pd.DataFrame()
    for symbol in symbols:
        zscores[symbol] = compute_zscore(portfolio_data[symbol], window=zscore_window)

    # Mostra últimos z-scores
    st.write("Z-scores Atuais:")
    latest_zscores = zscores.iloc[-1].to_frame('Z-score').round(2)
    
    def color_zscore(val):
        # Handle Series/DataFrame input
        if hasattr(val, '__len__') and not isinstance(val, (int, float)):
            return ['background-color: transparent'] * len(val)

        # Handle scalar values
        if abs(val) > 2:
            return 'background-color: #ff4444'  # Red for high Z-score
        elif abs(val) > 1:
            return 'background-color: #ffeb3b'  # Yellow for medium Z-score
        else:
            return 'background-color: #90ee90'  # Green for normal Z-score
    
    st.dataframe(latest_zscores.style.apply(color_zscore))

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual - Análise de Risco", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta aba:
        Avaliar os riscos do portfólio através de métricas estatísticas e análise de correlação.

        ### 📊 Como interpretar as métricas:
        - **Volatilidade Anualizada**: Variabilidade esperada dos retornos (valores altos = alto risco)
        - **VaR (95%)**: Potencial perda máxima em condições normais de mercado
        - **Expected Shortfall**: Perda média nas piores situações (mais conservador que VaR)
        - **Máximo Drawdown**: Maior perda acumulada histórica
        - **Z-Score**: Desvio padrão do preço (valores extremos indicam reversões)

        ### 🔧 Como usar as ferramentas:
        1. Configure o período de análise e nível de confiança
        2. Analise as métricas de risco principais
        3. Observe a matriz de correlação (valores próximos de 1 = alta correlação)
        4. Verifique a contribuição de risco por ativo
        5. Analise o drawdown histórico e Z-scores atuais

        ### 💡 Dicas importantes:
        - **VaR 95%**: Interprete como "há 95% de chance de não perder mais que este valor"
        - **Correlação alta**: Reduz diversidade do portfólio (risco concentrado)
        - **Z-score > 2**: Ativo muito acima da média histórica (possível venda)
        - **Z-score < -2**: Ativo muito abaixo da média histórica (possível compra)

        ### ⚠️ Considerações importantes:
        - Métricas passadas não garantem riscos futuros
        - Use VaR junto com Expected Shortfall para visão completa
        - Monitore correlações que podem mudar em crises de mercado
        - Considere liquidez dos ativos para cenários de estresse

        *Última atualização: v2.0*
        ''')
