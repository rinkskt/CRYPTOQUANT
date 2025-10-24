"""
Portfolio Spread Analysis Tab

Análise de spread entre portfólio e ativo de referência.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

from app.analytics.portfolio import get_portfolio_data, analyze_spread_full
from .portfolio_form import load_portfolio, PortfolioPosition


def create_spread_chart(spread: pd.Series) -> go.Figure:
    """
    Cria gráfico do spread do portfólio vs benchmark.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread.values,
        mode='lines',
        name='Spread',
        line=dict(color='blue', width=2)
    ))

    # Linha zero de referência
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Equilíbrio")

    fig.update_layout(
        title="Spread do Portfólio vs Benchmark",
        xaxis_title="Data",
        yaxis_title="Spread Normalizado",
        hovermode="x unified",
        showlegend=False,
        height=400
    )

    return fig


def create_zscore_chart(zscore: pd.Series) -> go.Figure:
    """
    Cria gráfico do Z-Score com bandas de desvio.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=zscore.index,
        y=zscore.values,
        mode='lines',
        name='Z-Score',
        line=dict(color='purple', width=2)
    ))

    # Bandas de desvio padrão
    fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="+2σ")
    fig.add_hline(y=-2, line_dash="dash", line_color="orange", annotation_text="-2σ")
    fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="+3σ")
    fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3σ")
    fig.add_hline(y=0, line_dash="solid", line_color="black", annotation_text="Média")

    fig.update_layout(
        title="Z-Score do Spread",
        xaxis_title="Data",
        yaxis_title="Z-Score",
        hovermode="x unified",
        showlegend=False,
        height=400
    )

    return fig


def create_rolling_correlation_chart(rolling_corr: pd.Series) -> go.Figure:
    """
    Cria gráfico da correlação móvel.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        mode='lines',
        name='Correlação Móvel (30d)',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title="Correlação Móvel (30 Dias)",
        xaxis_title="Data",
        yaxis_title="Correlação",
        yaxis=dict(range=[-1, 1]),
        hovermode="x unified",
        showlegend=False,
        height=400
    )

    return fig


def show_spread_analysis_tab():
    """
    Mostra a aba de análise de spread do portfólio.
    """
    st.header("📊 Análise de Spread")

    # Carrega dados do portfólio
    portfolio = load_portfolio()
    if not portfolio:
        st.warning("Nenhuma posição encontrada. Adicione ativos ao seu portfólio primeiro.")
        return

    # Configurações
    col1, col2, col3 = st.columns(3)

    with col1:
        lookback_days = st.selectbox(
            "Período de Análise",
            options=[30, 90, 180, 365],
            index=1,
            format_func=lambda x: f"Últimos {x} dias",
            key="spread_lookback_days"
        )

    with col2:
        benchmark_symbol = st.selectbox(
            "Ativo de Referência",
            options=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            index=0,
            key="spread_benchmark"
        )

    with col3:
        window = st.slider(
            "Janela para Correlação (dias)",
            min_value=10,
            max_value=60,
            value=30,
            key="spread_window"
        )

    # Prepara dados
    symbols = [pos.symbol for pos in portfolio]
    weights = {pos.symbol: pos.qty * pos.price_entry for pos in portfolio}
    total_value = sum(weights.values())
    weights = {k: v/total_value for k, v in weights.items()}

    # Carrega dados históricos usando o módulo unificado
    try:
        interval = "1d"
        limit = lookback_days

        portfolio_raw = get_portfolio_data(symbols, interval=interval, limit=limit)

        # Carrega dados do benchmark
        benchmark_raw = get_portfolio_data([benchmark_symbol], interval=interval, limit=limit)

        if benchmark_symbol not in benchmark_raw:
            st.error(f"Não foi possível carregar dados do benchmark {benchmark_symbol}.")
            return

        # Converte para séries normalizadas
        portfolio_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in portfolio_raw:
                df = portfolio_raw[symbol].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                portfolio_data[symbol] = df['close']

        if portfolio_data.empty:
            st.error("Não foi possível carregar dados históricos do portfólio.")
            return

        # Calcula valor ponderado do portfólio
        portfolio_value = (portfolio_data * pd.Series(weights)).sum(axis=1)

        # Dados do benchmark
        benchmark_df = benchmark_raw[benchmark_symbol].copy()
        benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
        benchmark_df = benchmark_df.set_index('timestamp')
        benchmark_value = benchmark_df['close']

        # Análise completa do spread
        spread_analysis = analyze_spread_full(portfolio_value, benchmark_value, window)

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Beta de Cointegração",
            f"{spread_analysis['beta']:.3f}"
        )

    with col2:
        st.metric(
            "Correlação Atual",
            f"{spread_analysis['corr']:.3f}"
        )

    with col3:
        current_zscore = spread_analysis['zscore'].iloc[-1]
        st.metric(
            "Z-Score Atual",
            f"{current_zscore:.2f}"
        )

    with col4:
        spread_std = spread_analysis['spread'].std()
        st.metric(
            "Volatilidade do Spread",
            f"{spread_std:.4f}"
        )

    # Gráficos
    st.subheader("Análise Visual do Spread")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_spread_chart(spread_analysis['spread']), config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}, use_container_width=True)

    with col2:
        st.plotly_chart(create_zscore_chart(spread_analysis['zscore']), config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}, use_container_width=True)

    # Correlação móvel
    st.plotly_chart(create_rolling_correlation_chart(spread_analysis['rolling_corr']), config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}, use_container_width=True)

    # Interpretação dos resultados
    st.subheader("Interpretação dos Resultados")

    current_zscore = spread_analysis['zscore'].iloc[-1]
    beta = spread_analysis['beta']
    corr = spread_analysis['corr']

    if abs(current_zscore) > 2:
        signal = "🚨 FORTE DESVIO" if current_zscore > 0 else "🚨 FORTE DESVIO NEGATIVO"
        color = "red"
    elif abs(current_zscore) > 1:
        signal = "⚠️ DESVIO MODERADO" if current_zscore > 0 else "⚠️ DESVIO MODERADO NEGATIVO"
        color = "orange"
    else:
        signal = "✅ EQUILÍBRIO"
        color = "green"

    st.markdown(f"### Sinal Atual: <span style='color:{color};'>{signal}</span>", unsafe_allow_html=True)

    # Explicação
    st.write("**Interpretação:**")
    st.write(f"- **Beta ({beta:.3f})**: Indica a sensibilidade do portfólio ao benchmark. Beta > 1 significa maior volatilidade.")
    st.write(f"- **Correlação ({corr:.3f})**: Mede o grau de movimento conjunto. Valores próximos de 1 indicam forte correlação.")
    st.write(f"- **Z-Score ({current_zscore:.2f})**: Desvio padrão do spread. Valores extremos (>2 ou <-2) sugerem oportunidades de arbitragem estatística.")

    # Tabela de dados recentes
    st.subheader("Dados Recentes")
    recent_data = pd.DataFrame({
        'Data': spread_analysis['spread'].tail(10).index.strftime('%Y-%m-%d'),
        'Spread': spread_analysis['spread'].tail(10).round(4),
        'Z-Score': spread_analysis['zscore'].tail(10).round(2)
    })
    st.dataframe(recent_data.set_index('Data'), width='stretch')

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual - Análise de Spread", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta aba:
        Analisar o spread entre o portfólio e um ativo de referência para identificar oportunidades de arbitragem estatística.

        ### 📊 Como interpretar as métricas:
        - **Beta de Cointegração**: Mede a relação de longo prazo entre os ativos
        - **Correlação Atual**: Grau de movimento conjunto no período analisado
        - **Z-Score Atual**: Desvio padrão do spread (valores extremos indicam desequilíbrios)
        - **Volatilidade do Spread**: Estabilidade da diferença entre os preços

        ### 🔧 Como usar as ferramentas:
        1. Selecione o período de análise e ativo de referência
        2. Configure a janela para correlação móvel
        3. Analise os gráficos: spread, Z-score e correlação
        4. Observe os sinais de equilíbrio vs desvio
        5. Monitore dados recentes para timing de entrada/saída

        ### 💡 Dicas importantes:
        - **Z-score > 2**: Portfólio sobrevalorizado vs benchmark (considere vender)
        - **Z-score < -2**: Portfólio subvalorizado vs benchmark (considere comprar)
        - **Correlação alta**: Spread mais previsível e confiável
        - **Bandas de desvio**: Z-score entre -2 e +2 indica equilíbrio normal

        ### ⚠️ Considerações importantes:
        - Estratégias de spread funcionam melhor em mercados eficientes
        - Considere custos de transação ao implementar a estratégia
        - Monitore mudanças na correlação ao longo do tempo
        - Use stops para limitar perdas em caso de quebra da relação

        *Última atualização: v2.0*
        ''')
