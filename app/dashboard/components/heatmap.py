"""
Correlation Heatmap Component

This module provides reusable heatmap components for correlation visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_correlation_heatmap(corr_data):
    """
    Create an interactive correlation heatmap.

    Args:
        corr_data: Dictionary with 'assets' and 'matrix' keys, or DataFrame

    Returns:
        plotly Figure object
    """
    if isinstance(corr_data, dict):
        # API response format
        assets = corr_data.get('assets', [])
        matrix = corr_data.get('matrix', [])

        if not assets or not matrix:
            return None

        corr_df = pd.DataFrame(matrix, index=assets, columns=assets)
    elif isinstance(corr_data, pd.DataFrame):
        corr_df = corr_data
    else:
        return None

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=np.round(corr_df.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{y} vs %{x}<br>Correlação: %{z:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title="Mapa de Calor de Correlação",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=600,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        )
    )

    # Add colorbar
    fig.update_traces(
        colorbar=dict(
            title="Correlação",
            titleside="right",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
        )
    )

    return fig


def create_rolling_correlation_heatmap(corr_matrices, dates):
    """
    Create animated heatmap for rolling correlations.

    Args:
        corr_matrices: List of correlation matrices
        dates: List of corresponding dates

    Returns:
        plotly Figure object with animation
    """
    if not corr_matrices or not dates:
        return None

    # Prepare data for animation
    assets = corr_matrices[0].columns.tolist()
    frames = []

    for i, (corr_df, date) in enumerate(zip(corr_matrices, dates)):
        frame = go.Frame(
            data=[go.Heatmap(
                z=corr_df.values,
                x=assets,
                y=assets,
                colorscale='RdYlGn',
                zmin=-1,
                zmax=1,
                text=np.round(corr_df.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8}
            )],
            name=str(date.date()) if hasattr(date, 'date') else str(date)
        )
        frames.append(frame)

    # Initial plot
    initial_corr = corr_matrices[0]
    fig = go.Figure(
        data=[go.Heatmap(
            z=initial_corr.values,
            x=assets,
            y=assets,
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            text=np.round(initial_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        )],
        frames=frames
    )

    # Update layout
    fig.update_layout(
        title="Evolução da Correlação (Rolling)",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=600,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 500, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 300}
                }]
            }, {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }]
            }]
        }]
    )

    # Add slider
    fig.update_layout(
        sliders=[{
            "active": 0,
            "steps": [{
                "args": [[str(date.date()) if hasattr(date, 'date') else str(date)], {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300}
                }],
                "label": str(date.date()) if hasattr(date, 'date') else str(date),
                "method": "animate"
            } for date in dates],
            "currentvalue": {"prefix": "Data: "},
            "transition": {"duration": 300, "easing": "cubic-in-out"}
        }]
    )

    return fig


def create_pairwise_heatmap(corr_matrix, highlight_pairs=None):
    """
    Create heatmap with optional highlighting of specific pairs.

    Args:
        corr_matrix: Correlation matrix DataFrame
        highlight_pairs: List of tuples (asset1, asset2) to highlight

    Returns:
        plotly Figure object
    """
    # Create base heatmap
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

    # Highlight specific pairs if provided
    if highlight_pairs:
        for pair in highlight_pairs:
            asset1, asset2 = pair
            if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                # Add annotation or marker for highlighted pairs
                corr_value = corr_matrix.loc[asset1, asset2]

                fig.add_annotation(
                    x=asset2,
                    y=asset1,
                    text=f"★{corr_value:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=20,
                    ay=-20,
                    font=dict(size=12, color="red")
                )

    fig.update_layout(
        title="Matriz de Correlação com Destaques",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=600
    )

    return fig


def create_correlation_network(corr_matrix, threshold=0.7):
    """
    Create a network visualization of correlations above threshold.

    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Minimum correlation to show connection

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

    # Add nodes
    for asset in corr_matrix.index:
        G.add_node(asset)

    # Add edges for correlations above threshold
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            asset1 = corr_matrix.index[i]
            asset2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]

            if abs(corr) >= threshold:
                G.add_edge(asset1, asset2, weight=abs(corr))

    # Calculate positions
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))

    node_trace.marker.color = node_adjacencies

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f"Rede de Correlação (Threshold: {threshold})",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    return fig
