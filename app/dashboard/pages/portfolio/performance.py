"""
Portfolio Performance Module

Análise e visualização de performance do portfólio.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from ...api_client import get_ohlcv_data
from .portfolio_form import load_portfolio, PortfolioPosition
from app.analytics.portfolio import get_portfolio_data, compute_returns

def calculate_portfolio_metrics(portfolio_data: pd.DataFrame, 
                             weights: Dict[str, float],
                             benchmark_data: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calcula métricas principais do portfólio.
    """
    # Calcula retornos diários
    returns = portfolio_data.pct_change()
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    # Métricas básicas
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Beta (se houver benchmark)
    beta = None
    if benchmark_data is not None:
        benchmark_returns = benchmark_data.pct_change()
        beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "beta": beta
    }

def create_performance_chart(portfolio_data: pd.DataFrame,
                           weights: Dict[str, float],
                           benchmark_symbol: Optional[str] = None) -> go.Figure:
    """
    Cria gráfico de performance do portfólio.
    """
    # Calcula valor do portfólio normalizado
    returns = portfolio_data.pct_change()
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    
    # Linha do portfólio
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        name="Portfólio",
        line=dict(color="blue", width=2)
    ))
    
    # Adiciona benchmark se fornecido
    if benchmark_symbol and benchmark_symbol in portfolio_data.columns:
        benchmark_value = (1 + portfolio_data[benchmark_symbol].pct_change()).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name=f"Benchmark ({benchmark_symbol})",
            line=dict(color="gray", width=1, dash="dot")
        ))
    
    fig.update_layout(
        title="Evolução do Portfólio",
        xaxis_title="Data",
        yaxis_title="Valor Normalizado",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig

def show_performance_tab():
    """
    Mostra a aba de performance do portfólio.
    """
    st.header("📈 Performance do Portfólio")
    
    try:
        # Carrega dados do portfólio
        portfolio = load_portfolio()
        if not portfolio:
            st.warning("Nenhuma posição encontrada. Adicione ativos ao seu portfólio primeiro.")
            return
            
        # Filtra apenas os campos válidos para análise
        positions = []
        for pos in portfolio:
            if isinstance(pos, PortfolioPosition):
                positions.append(pos)
            else:
                st.error(f"Dados inválidos encontrados no portfólio: {pos}")
                return
                
        portfolio = positions
        
    except Exception as e:
        st.error(f"Erro ao carregar dados do portfólio: {str(e)}")
        return
    
    # Configurações de período
    col1, col2, col3 = st.columns(3)
    with col1:
        lookback_days = st.selectbox(
            "Período de Análise",
            options=[7, 30, 90, 180, 365],
            index=2,
            format_func=lambda x: f"Últimos {x} dias"
        )
    
    with col2:
        benchmark = st.selectbox(
            "Benchmark",
            options=["BTCUSDT", "ETHUSDT", None],
            format_func=lambda x: x if x else "Nenhum"
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
        # Converte período para limite de dados (aproximado)
        interval = "1d"  # dados diários
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
    
    # Calcula e mostra métricas
    metrics = calculate_portfolio_metrics(
        portfolio_data, 
        weights,
        portfolio_data[benchmark] if benchmark else None
    )
    
    # Mostra métricas em colunas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Retorno Total",
            f"{metrics['total_return']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Retorno Anualizado",
            f"{metrics['annualized_return']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Volatilidade Anualizada",
            f"{metrics['volatility']*100:.2f}%"
        )
    
    with col4:
        if metrics['beta'] is not None:
            st.metric(
                f"Beta vs {benchmark}",
                f"{metrics['beta']:.2f}"
            )
        else:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # Gráfico de performance
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'responsive': True
    }
    st.plotly_chart(
        create_performance_chart(portfolio_data, weights, benchmark),
        config=config,
        use_container_width=True
    )
    
    # Análise de performance por ativo
    st.subheader("Performance por Ativo")
    
    asset_metrics = []
    for symbol in symbols:
        returns = portfolio_data[symbol].pct_change()
        metrics = {
            "Symbol": symbol,
            "Retorno": returns.sum(),
            "Volatilidade": returns.std() * np.sqrt(252),
            "Peso": weights[symbol]
        }
        asset_metrics.append(metrics)
    
    asset_metrics_df = pd.DataFrame(asset_metrics)
    st.dataframe(
        asset_metrics_df.style.format({
            "Retorno": "{:.2%}",
            "Volatilidade": "{:.2%}",
            "Peso": "{:.2%}"
        }),
        width='stretch'
    )

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual - Análise de Performance", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta aba:
        Analisar o desempenho histórico do portfólio, incluindo retornos, volatilidade e comparação com benchmarks.

        ### 📊 Como interpretar as métricas:
        - **Retorno Total**: Ganho/perda acumulado no período analisado
        - **Retorno Anualizado**: Taxa de retorno projetada para um ano
        - **Volatilidade Anualizada**: Medida de risco baseada na variabilidade dos retornos
        - **Beta**: Sensibilidade do portfólio ao benchmark (Beta > 1 = mais volátil)
        - **Sharpe Ratio**: Retorno por unidade de risco (valores maiores = melhor)

        ### 🔧 Como usar as ferramentas:
        1. Selecione o período de análise desejado
        2. Escolha um benchmark para comparação (BTC, ETH ou nenhum)
        3. Analise as métricas principais no topo
        4. Observe o gráfico de evolução do portfólio
        5. Verifique a performance individual de cada ativo

        ### 💡 Dicas importantes:
        - **Retorno Anualizado**: Use para comparar com outras oportunidades de investimento
        - **Volatilidade**: Portfólios com volatilidade > 50% são considerados de alto risco
        - **Sharpe Ratio > 1**: Performance ajustada ao risco considerada boa
        - **Beta vs BTC**: Valores próximos de 1 indicam correlação forte com o mercado

        ### ⚠️ Considerações importantes:
        - Resultados passados não garantem performance futura
        - Use sempre períodos longos para análise mais robusta
        - Considere custos de transação ao avaliar retornos

        *Última atualização: v2.0*
        ''')
