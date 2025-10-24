"""
Portfolio Manager Component

Provides reusable components for portfolio management and rebalancing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

def create_portfolio_editor(
    current_portfolio: pd.DataFrame,
    prices: Dict[str, float]
) -> pd.DataFrame:
    """
    Creates an editable portfolio table with current positions and metrics.
    """
    # Add editing capabilities to the portfolio table
    edited_df = st.data_editor(
        current_portfolio,
        column_config={
            "Asset": st.column_config.TextColumn("Ativo"),
            "Quantity": st.column_config.NumberColumn("Quantidade", min_value=0.0),
            "Avg_Price": st.column_config.NumberColumn("Preço Médio", min_value=0.0),
            "Current_Value": st.column_config.NumberColumn("Valor Atual", format="$%.2f"),
            "Change_Pct": st.column_config.NumberColumn("Variação %", format="%.2f%%"),
            "Weight": st.column_config.NumberColumn("Peso %", format="%.2f%%"),
            "Risk_Contrib": st.column_config.NumberColumn("Contrib. Risco", format="%.2f%%")
        },
        disabled=["Current_Value", "Change_Pct", "Weight", "Risk_Contrib"],
        num_rows="dynamic"
    )
    
    return edited_df

def create_efficient_frontier_chart(
    returns: np.ndarray,
    volatilities: np.ndarray,
    sharpe_ratios: np.ndarray,
    current_portfolio: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Creates an interactive efficient frontier plot.
    """
    fig = go.Figure()
    
    # Add efficient frontier line
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='lines',
        name='Fronteira Eficiente',
        line=dict(color='blue', width=2)
    ))
    
    # Add current portfolio point if provided
    if current_portfolio:
        current_vol, current_ret = current_portfolio
        fig.add_trace(go.Scatter(
            x=[current_vol],
            y=[current_ret],
            mode='markers',
            name='Portfólio Atual',
            marker=dict(
                color='red',
                size=10,
                symbol='star'
            )
        ))
    
    # Add Sharpe ratio as color gradient
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers',
        name='Sharpe Ratio',
        marker=dict(
            size=8,
            color=sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        )
    ))
    
    fig.update_layout(
        title='Fronteira Eficiente de Markowitz',
        xaxis_title='Volatilidade (%)',
        yaxis_title='Retorno Esperado (%)',
        showlegend=True
    )
    
    return fig

def create_risk_contribution_chart(
    risk_contrib: Dict[str, float]
) -> go.Figure:
    """
    Creates a pie chart showing risk contribution by asset.
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_contrib.keys()),
        values=list(risk_contrib.values()),
        hole=.3
    )])
    
    fig.update_layout(
        title='Contribuição de Risco por Ativo'
    )
    
    return fig

def create_rebalancing_chart(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float]
) -> go.Figure:
    """
    Creates a comparison chart between current and target weights.
    """
    fig = go.Figure()
    
    assets = list(current_weights.keys())
    
    # Add current weights
    fig.add_trace(go.Bar(
        name='Peso Atual',
        x=assets,
        y=list(current_weights.values()),
        marker_color='lightblue'
    ))
    
    # Add target weights
    fig.add_trace(go.Bar(
        name='Peso Alvo',
        x=assets,
        y=list(target_weights.values()),
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='Comparação de Pesos: Atual vs Alvo',
        xaxis_title='Ativos',
        yaxis_title='Peso (%)',
        barmode='group'
    )
    
    return fig