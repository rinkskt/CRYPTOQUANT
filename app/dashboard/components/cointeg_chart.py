"""
Cointegration Chart Component

This module provides reusable cointegration chart components.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np


def create_cointegration_chart(price_data, asset1, asset2, beta, current_zscore):
    """
    Create a comprehensive cointegration analysis chart.

    Args:
        price_data: DataFrame with price data for both assets
        asset1: First asset symbol
        asset2: Second asset symbol
        beta: Cointegration beta coefficient
        current_zscore: Current z-score value

    Returns:
        plotly Figure object
    """
    if price_data is None or price_data.empty:
        return None

    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Preços Normalizados: {asset1} vs {asset2}',
            f'Spread: {asset2} - {beta:.4f} × {asset1}',
            'Z-Score do Spread'
        ),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Normalize prices
    normalized_prices = price_data.div(price_data.iloc[0])

    # Plot 1: Normalized prices
    fig.add_trace(
        go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[asset1],
            name=f'{asset1} (normalizado)',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[asset2],
            name=f'{asset2} (normalizado)',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Calculate spread
    spread = price_data[asset2] - beta * price_data[asset1]

    # Plot 2: Spread
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name='Spread',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    # Add spread mean line
    spread_mean = spread.mean()
    fig.add_hline(
        y=spread_mean,
        line_dash="dash",
        line_color="gray",
        row=2, col=1,
        annotation_text=f"Média: {spread_mean:.4f}"
    )

    # Calculate z-score
    zscore = (spread - spread.mean()) / spread.std()

    # Plot 3: Z-score
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            name='Z-Score',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )

    # Add z-score threshold lines
    for threshold in [-2, -1.5, 0, 1.5, 2]:
        line_color = "red" if abs(threshold) >= 2 else "orange" if abs(threshold) >= 1.5 else "gray"
        line_dash = "dash" if abs(threshold) >= 1.5 else "solid"

        fig.add_hline(
            y=threshold,
            line_dash=line_dash,
            line_color=line_color,
            row=3, col=1
        )

    # Add current z-score annotation
    latest_date = zscore.index[-1]
    fig.add_annotation(
        x=latest_date,
        y=current_zscore,
        text=f"Z-score Atual: {current_zscore:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=-30,
        font=dict(size=12, color="purple"),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Preço Normalizado", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=3, col=1)

    # Update x-axis title only on bottom plot
    fig.update_xaxes(title_text="Data", row=3, col=1)

    return fig


def create_spread_analysis_chart(spread_series, zscore_series, half_life):
    """
    Create a detailed spread analysis chart.

    Args:
        spread_series: Pandas Series with spread values
        zscore_series: Pandas Series with z-score values
        half_life: Mean reversion half-life

    Returns:
        plotly Figure object
    """
    if spread_series is None or spread_series.empty:
        return None

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'Análise do Spread',
            f'Z-Score (Half-life: {half_life:.1f} períodos)'
        )
    )

    # Plot 1: Spread with confidence bands
    spread_mean = spread_series.mean()
    spread_std = spread_series.std()

    fig.add_trace(
        go.Scatter(
            x=spread_series.index,
            y=spread_series.values,
            name='Spread',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Add confidence bands
    fig.add_hrect(
        y0=spread_mean - 2*spread_std,
        y1=spread_mean + 2*spread_std,
        fillcolor="lightblue",
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1, col=1
    )

    fig.add_hline(
        y=spread_mean,
        line_dash="solid",
        line_color="red",
        row=1, col=1,
        annotation_text=f"Média: {spread_mean:.4f}"
    )

    # Plot 2: Z-score
    if zscore_series is not None and not zscore_series.empty:
        fig.add_trace(
            go.Scatter(
                x=zscore_series.index,
                y=zscore_series.values,
                name='Z-Score',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )

        # Add z-score thresholds
        for threshold in [-2, -1.5, 0, 1.5, 2]:
            color = "red" if abs(threshold) >= 2 else "orange" if abs(threshold) >= 1.5 else "gray"
            dash = "dash" if abs(threshold) >= 1.5 else "solid"

            fig.add_hline(
                y=threshold,
                line_dash=dash,
                line_color=color,
                row=2, col=1
            )

    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_xaxes(title_text="Data", row=2, col=1)

    return fig


def create_cointegration_network(cointegration_pairs, assets):
    """
    Create a network visualization of cointegration relationships.

    Args:
        cointegration_pairs: List of cointegration pairs
        assets: List of all assets

    Returns:
        plotly Figure object
    """
    try:
        import networkx as nx
    except ImportError:
        st.warning("NetworkX não instalado. Instale com: pip install networkx")
        return None

    # Create network graph
    G = nx.Graph()

    # Add all assets as nodes
    for asset in assets:
        G.add_node(asset)

    # Add edges for cointegrated pairs
    if cointegration_pairs:
        for pair in cointegration_pairs:
            if pair.get('cointegrated', False):
                asset1 = pair.get('asset_x') or pair.get('asset1')
                asset2 = pair.get('asset_y') or pair.get('asset2')
                correlation = pair.get('correlation', 0)

                if asset1 and asset2 and asset1 in assets and asset2 in assets:
                    G.add_edge(asset1, asset2, weight=abs(correlation))

    # Calculate positions
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_sizes.append(20 + len(list(G.neighbors(node))) * 5)  # Size based on connections

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_sizes,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(
                thickness=15,
                title='Conexões',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="Rede de Cointegração",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))

    return fig


def create_pair_trading_signals(price_data, asset1, asset2, zscore_series, entry_threshold=2, exit_threshold=0):
    """
    Create a chart showing pair trading signals.

    Args:
        price_data: DataFrame with price data
        asset1: First asset symbol
        asset2: Second asset symbol
        zscore_series: Z-score series
        entry_threshold: Z-score threshold for entering trades
        exit_threshold: Z-score threshold for exiting trades

    Returns:
        plotly Figure object
    """
    if price_data is None or zscore_series is None or zscore_series.empty:
        return None

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Preços: {asset1} vs {asset2}',
            'Sinais de Trading (Z-Score)'
        )
    )

    # Plot 1: Prices
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data[asset1],
            name=asset1,
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data[asset2],
            name=asset2,
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Plot 2: Z-score with signals
    fig.add_trace(
        go.Scatter(
            x=zscore_series.index,
            y=zscore_series.values,
            name='Z-Score',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Add threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text=f"Entrada: +{entry_threshold}σ")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text=f"Entrada: -{entry_threshold}σ")
    fig.add_hline(y=exit_threshold, line_dash="solid", line_color="green", row=2, col=1,
                  annotation_text=f"Saída: {exit_threshold}σ")

    # Identify trading signals
    long_signals = zscore_series[zscore_series <= -entry_threshold]
    short_signals = zscore_series[zscore_series >= entry_threshold]
    exit_signals = zscore_series[abs(zscore_series) <= exit_threshold]

    # Add signal markers
    if not long_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=long_signals.index,
                y=long_signals.values,
                mode='markers',
                name='Sinal Long',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=2, col=1
        )

    if not short_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=short_signals.values,
                mode='markers',
                name='Sinal Short',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_xaxes(title_text="Data", row=2, col=1)

    return fig
