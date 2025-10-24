"""
Implementa a funcionalidade de rebalanceamento de portfólio.
"""
from datetime import datetime, timedelta
import streamlit as st
from .market_data import get_current_prices, get_historical_data
from .portfolio_form import load_portfolio, PortfolioPosition
def create_rebalancing_chart(current_weights, target_weights):
    """Placeholder for rebalancing chart"""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(current_weights.keys()), y=list(current_weights.values()), name='Current'))
    fig.add_trace(go.Bar(x=list(target_weights.keys()), y=list(target_weights.values()), name='Target'))
    fig.update_layout(title='Current vs Target Allocation', barmode='group')
    return fig

def calculate_rebalancing_impact(historical_data, current_weights, target_weights):
    """Placeholder for rebalancing impact calculation"""
    return {
        'current_vol': 0.15,
        'target_vol': 0.12,
        'current_sharpe': 1.2,
        'target_sharpe': 1.4
    }

def suggest_rebalancing_trades(portfolio, target_weights, current_prices):
    """Placeholder for trade suggestions"""
    import pandas as pd
    # Simple placeholder implementation
    trades = []
    for pos in portfolio:
        symbol = pos.symbol
        current_value = pos.qty * current_prices.get(symbol, 0)
        target_weight = target_weights.get(symbol, 0)
        total_value = sum(p.qty * current_prices.get(p.symbol, 0) for p in portfolio)
        target_value = total_value * target_weight
        diff_value = target_value - current_value
        diff_qty = diff_value / current_prices.get(symbol, 1) if current_prices.get(symbol, 1) > 0 else 0

        trades.append({
            'symbol': symbol,
            'current_qty': pos.qty,
            'target_qty': pos.qty + diff_qty,
            'diff_qty': diff_qty,
            'diff_value': diff_value,
            'action': 'Buy' if diff_qty > 0 else 'Sell',
            'price': current_prices.get(symbol, 0)
        })

    return pd.DataFrame(trades)

def show_rebalance_tab():
    """
    Mostra a aba de rebalanceamento do portfólio.
    """    
    st.header("♻️ Rebalanceamento de Portfólio")
    
    try:
        # Carrega dados do portfólio
        portfolio = load_portfolio()
        if not portfolio:
            st.warning("Nenhuma posição encontrada. Adicione ativos ao seu portfólio primeiro.")
            return
            
        # Obtém preços atuais
        symbols = [pos.symbol for pos in portfolio]
        current_prices = get_current_prices(symbols)
        
        if not current_prices:
            st.error("Não foi possível obter preços atuais. Tente novamente mais tarde.")
            return
        
        # Calcula alocação atual
        total_value = sum(pos.qty * current_prices[pos.symbol] for pos in portfolio)
        current_weights = {
            pos.symbol: (pos.qty * current_prices[pos.symbol]) / total_value 
            for pos in portfolio
        }
        
        # Interface para definir pesos alvo
        st.subheader("Definir Alocação Alvo")
        target_weights = {}
        
        cols = st.columns(3)
        for i, pos in enumerate(portfolio):
            with cols[i % 3]:
                target_weights[pos.symbol] = st.number_input(
                    f"Peso Alvo - {pos.symbol}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_weights[pos.symbol] * 100),
                    key=f"target_{pos.symbol}"
                ) / 100
        
        # Normaliza os pesos alvo
        total_target = sum(target_weights.values())
        if total_target > 0:
            target_weights = {k: v/total_target for k, v in target_weights.items()}
        
        # Mostra gráfico comparativo
        if st.checkbox("Mostrar Gráfico Comparativo", value=True):
            fig = create_rebalancing_chart(current_weights, target_weights)
            st.plotly_chart(fig, config={'responsive': True})
        
        # Calcula impacto do rebalanceamento
        st.subheader("Impacto do Rebalanceamento")
        
        # Obtém dados históricos para análise
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        historical_data = get_historical_data(symbols, start_date, end_date)
        
        if not historical_data.empty:
            impact = calculate_rebalancing_impact(historical_data, current_weights, target_weights)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Volatilidade Atual", f"{impact['current_vol']:.1%}")
            with col2:
                st.metric("Nova Volatilidade", f"{impact['target_vol']:.1%}")
            with col3:
                st.metric("Sharpe Atual", f"{impact['current_sharpe']:.2f}")
            with col4:
                st.metric("Novo Sharpe", f"{impact['target_sharpe']:.2f}")
        
        # Sugere trades específicos
        st.subheader("Trades Sugeridos")
        trades_df = suggest_rebalancing_trades(portfolio, target_weights, current_prices)
        
        if not trades_df.empty:
            # Configuração das colunas
            column_config = {
                "symbol": "Symbol",
                "current_qty": st.column_config.NumberColumn("Qtd. Atual", format="%.4f"),
                "target_qty": st.column_config.NumberColumn("Qtd. Alvo", format="%.4f"),
                "diff_qty": st.column_config.NumberColumn("Diferença", format="%.4f"),
                "diff_value": st.column_config.NumberColumn("Valor ($)", format="%.2f"),
                "action": "Ação",
                "price": st.column_config.NumberColumn("Preço", format="%.2f")
            }

            st.dataframe(
                trades_df,
                column_config=column_config,
                width='stretch'
            )
        else:
            st.info("Não foi possível calcular sugestões de trades.")

        # ===== MANUAL DA PÁGINA =====
        st.markdown("---")
        with st.expander("📖 Manual - Rebalanceamento de Portfólio", expanded=False):
            st.markdown('''
            ### 🎯 Objetivo desta aba:
            Ajustar a composição do portfólio para atingir pesos-alvo desejados.

            ### 📊 Como interpretar os resultados:
            - **Peso Atual**: Participação atual de cada ativo no portfólio
            - **Peso Alvo**: Participação desejada após rebalanceamento
            - **Diferença**: Ajuste necessário (positivo = comprar, negativo = vender)
            - **Ação**: Sugestão baseada na diferença (>2% = comprar/vender)

            ### 🔧 Como usar as ferramentas:
            1. Defina os pesos-alvo desejados para cada ativo (em %)
            2. Visualize o gráfico comparativo atual vs alvo
            3. Analise o impacto no risco e retorno esperado
            4. Revise as sugestões específicas de trades
            5. Execute os trades considerando custos de transação

            ### 💡 Dicas importantes:
            - **Rebalanceamento periódico**: Mantenha a alocação ideal ao longo do tempo
            - **Tolerância**: Use banda de 2-5% para evitar transações excessivas
            - **Custos**: Considere taxas de corretagem e impacto no preço
            - **Liquidez**: Verifique volume de negociação dos ativos

            ### ⚠️ Considerações importantes:
            - Rebalanceamento frequente pode aumentar custos desnecessariamente
            - Considere o contexto de mercado antes de executar trades
            - Monitore impostos sobre ganhos de capital
            - Avalie se o rebalanceamento ainda faz sentido dado o novo contexto

            *Última atualização: v2.0*
            ''')

    except Exception as e:
        st.error(f"Erro ao processar rebalanceamento: {str(e)}")
