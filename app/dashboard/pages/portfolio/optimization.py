"""
Portfolio Optimization Module

Otimização e recomendações de alocação do portfólio.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.optimize import minimize

from app.analytics.portfolio.rebalance import PortfolioRebalancer
from .portfolio_form import load_portfolio, PortfolioPosition
from ...api_client import get_ohlcv_data
from app.analytics.portfolio import get_portfolio_data

def calculate_portfolio_metrics(weights: np.ndarray, 
                             returns: pd.DataFrame,
                             risk_free_rate: float = 0.03) -> Tuple[float, float, float]:
    """
    Calcula retorno, volatilidade e Sharpe ratio do portfólio.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    
    return portfolio_return, portfolio_std, sharpe_ratio

def optimize_portfolio(returns: pd.DataFrame,
                     risk_free_rate: float = 0.03,
                     target_return: Optional[float] = None) -> Dict[str, float]:
    """
    Otimiza o portfólio usando o modelo de Markowitz.
    """
    n_assets = len(returns.columns)
    
    def objective(weights):
        return -calculate_portfolio_metrics(weights, returns, risk_free_rate)[2]
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # soma dos pesos = 1
    ]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    if result.success:
        return dict(zip(returns.columns, result.x))
    else:
        return dict(zip(returns.columns, initial_weights))

def generate_efficient_frontier(returns: pd.DataFrame,
                              n_portfolios: int = 100) -> pd.DataFrame:
    """
    Gera a fronteira eficiente de Markowitz.
    """
    # Range de retornos alvos
    min_ret = returns.mean().min() * 252
    max_ret = returns.mean().max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_portfolios)
    
    results = []
    for target_ret in target_returns:
        weights = optimize_portfolio(returns, target_return=target_ret)
        ret, vol, sharpe = calculate_portfolio_metrics(
            np.array(list(weights.values())),
            returns
        )
        results.append({
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'weights': weights
        })
    
    return pd.DataFrame(results)

def create_efficient_frontier_chart(ef_data: pd.DataFrame,
                                 current_portfolio: Optional[Tuple[float, float]] = None) -> go.Figure:
    """
    Cria gráfico da fronteira eficiente.
    """
    fig = go.Figure()
    
    # Fronteira eficiente
    fig.add_trace(go.Scatter(
        x=ef_data['volatility'],
        y=ef_data['return'],
        mode='lines',
        name='Fronteira Eficiente',
        line=dict(color='blue', width=2)
    ))
    
    # Portfólio atual
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio[1]],
            y=[current_portfolio[0]],
            mode='markers',
            name='Portfólio Atual',
            marker=dict(color='red', size=10)
        ))
    
    # Portfólio de máximo Sharpe
    max_sharpe_idx = ef_data['sharpe'].idxmax()
    fig.add_trace(go.Scatter(
        x=[ef_data.loc[max_sharpe_idx, 'volatility']],
        y=[ef_data.loc[max_sharpe_idx, 'return']],
        mode='markers',
        name='Máximo Sharpe',
        marker=dict(color='green', size=10)
    ))
    
    fig.update_layout(
        title="Fronteira Eficiente",
        xaxis_title="Volatilidade Anualizada",
        yaxis_title="Retorno Esperado Anualizado",
        height=600,
        showlegend=True
    )
    
    return fig

def show_optimization_tab():
    """
    Mostra a aba de otimização do portfólio.
    """
    st.header("🎯 Otimização de Portfólio")
    
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
            key="opt_lookback_days"
        )
    
    with col2:
        risk_free_rate = st.number_input(
            "Taxa Livre de Risco (a.a.)",
            min_value=0.0,
            max_value=0.2,
            value=0.03,
            format="%.2f",
            key="opt_risk_free_rate"
        )
    
    # Prepara dados
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    symbols = [pos.symbol for pos in portfolio]
    current_weights = {pos.symbol: pos.qty * pos.price_entry for pos in portfolio}
    total_value = sum(current_weights.values())
    current_weights = {k: v/total_value for k, v in current_weights.items()}
    
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
    
    # Calcula métricas do portfólio atual
    current_weights_array = np.array(list(current_weights.values()))
    current_return, current_vol, current_sharpe = calculate_portfolio_metrics(
        current_weights_array, returns, risk_free_rate
    )
    
    # Gera fronteira eficiente
    with st.spinner("Calculando fronteira eficiente..."):
        ef_data = generate_efficient_frontier(returns)
    
    # Mostra gráfico da fronteira eficiente
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'responsive': True
    }
    st.plotly_chart(
        create_efficient_frontier_chart(ef_data, (current_return, current_vol)),
        config=config,
        use_container_width=True
    )
    
    # Mostra diferentes estratégias de otimização
    st.subheader("Estratégias de Alocação")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Portfólio Atual**")
        current_metrics = pd.DataFrame({
            'Métrica': ['Retorno Esperado', 'Volatilidade', 'Sharpe Ratio'],
            'Valor': [
                f"{current_return*100:.2f}%",
                f"{current_vol*100:.2f}%",
                f"{current_sharpe:.2f}"
            ]
        })
        st.dataframe(current_metrics.set_index('Métrica'), width='stretch')
    
    with col2:
        st.write("**Máximo Sharpe Ratio**")
        max_sharpe_idx = ef_data['sharpe'].idxmax()
        max_sharpe_metrics = pd.DataFrame({
            'Métrica': ['Retorno Esperado', 'Volatilidade', 'Sharpe Ratio'],
            'Valor': [
                f"{ef_data.loc[max_sharpe_idx, 'return']*100:.2f}%",
                f"{ef_data.loc[max_sharpe_idx, 'volatility']*100:.2f}%",
                f"{ef_data.loc[max_sharpe_idx, 'sharpe']:.2f}"
            ]
        })
        st.dataframe(max_sharpe_metrics.set_index('Métrica'), width='stretch')
    
    with col3:
        st.write("**Risk Parity**")
        # Calcula pesos usando inverse volatility
        vols = returns.std()
        inv_vols = 1/vols
        risk_parity_weights = inv_vols/inv_vols.sum()
        rp_return, rp_vol, rp_sharpe = calculate_portfolio_metrics(
            risk_parity_weights.values, returns, risk_free_rate
        )
        
        risk_parity_metrics = pd.DataFrame({
            'Métrica': ['Retorno Esperado', 'Volatilidade', 'Sharpe Ratio'],
            'Valor': [
                f"{rp_return*100:.2f}%",
                f"{rp_vol*100:.2f}%",
                f"{rp_sharpe:.2f}"
            ]
        })
        st.dataframe(risk_parity_metrics.set_index('Métrica'), width='stretch')
    
    # Mostra composições sugeridas
    st.subheader("Composições Sugeridas")
    
    compositions = pd.DataFrame({
        'Atual': pd.Series(current_weights),
        'Máximo Sharpe': pd.Series(ef_data.loc[max_sharpe_idx, 'weights']),
        'Risk Parity': risk_parity_weights
    })
    
    # Formata percentuais
    st.dataframe(
        compositions.style.format("{:.2%}"),
        width='stretch'
    )
    
    # Sugestões de rebalanceamento
    st.subheader("Sugestões de Rebalanceamento")
    
    strategy = st.radio(
        "Escolha a estratégia alvo:",
        ["Máximo Sharpe", "Risk Parity"],
        key="opt_strategy_choice"
    )
    
    target_weights = (ef_data.loc[max_sharpe_idx, 'weights'] 
                     if strategy == "Máximo Sharpe" 
                     else risk_parity_weights)
    
    # Calculate deviations using PortfolioRebalancer
    rebalancer = PortfolioRebalancer(
        positions={symbol: {'qty': pos.qty} for symbol, pos in zip(symbols, portfolio)},
        target_weights=target_weights,
        prices={symbol: pos.price_entry for symbol, pos in zip(symbols, portfolio)},
        total_value=total_value
    )
    deviations = rebalancer.calculate_deviations(relative=False)
    
    rebal_df = pd.DataFrame({
        'Peso Atual': pd.Series(current_weights),
        'Peso Alvo': target_weights,
        'Diferença': pd.Series(deviations)
    })
    
    # Adiciona sugestão de ação
    rebal_df['Ação'] = rebal_df['Diferença'].apply(
        lambda x: 'COMPRAR' if x > 0.02 else 'VENDER' if x < -0.02 else 'MANTER'
    )
    
    st.dataframe(
        rebal_df.style.format({
            'Peso Atual': '{:.2%}',
            'Peso Alvo': '{:.2%}',
            'Diferença': '{:.2%}'
        }),
        width='stretch'
    )

    # ===== MANUAL DA PÁGINA =====
    st.markdown("---")
    with st.expander("📖 Manual - Otimização de Portfólio", expanded=False):
        st.markdown('''
        ### 🎯 Objetivo desta aba:
        Encontrar a alocação ótima de ativos baseada em diferentes estratégias de otimização.

        ### 📊 Como interpretar as métricas:
        - **Retorno Esperado**: Projeção de retorno baseada em dados históricos
        - **Volatilidade**: Risco medido pela variabilidade dos retornos
        - **Sharpe Ratio**: Retorno por unidade de risco (valores maiores = melhor)
        - **Fronteira Eficiente**: Curva de melhores combinações risco-retorno

        ### 🔧 Como usar as ferramentas:
        1. Configure o período de análise e taxa livre de risco
        2. Analise a fronteira eficiente (curva azul)
        3. Compare estratégias: Atual, Máximo Sharpe e Risk Parity
        4. Veja composições sugeridas para cada estratégia
        5. Use sugestões de rebalanceamento para ajustar o portfólio

        ### 💡 Dicas importantes:
        - **Máximo Sharpe**: Melhor relação risco-retorno teórica
        - **Risk Parity**: Distribui risco igualmente entre ativos
        - **Fronteira Eficiente**: Pontos à esquerda são ineficientes
        - **Rebalanceamento**: Considere custos de transação

        ### ⚠️ Considerações importantes:
        - Otimizações baseadas em dados passados não garantem futuro
        - Considere restrições pessoais (liquidez, convicções)
        - Monitore e rebalanceie periodicamente o portfólio
        - Diversifique além de criptoativos para reduzir risco total

        *Última atualização: v2.0*
        ''')
