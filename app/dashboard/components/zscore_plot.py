"""
Z-Score Plot Component

This module provides reusable z-score plotting components.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_zscore_plot(zscore_series, title="Z-Score Analysis"):
    """
    Create a z-score plot with threshold bands.

    Args:
        zscore_series: Pandas Series with z-score values
        title: Plot title

    Returns:
        plotly Figure object
    """
    if zscore_series is None or zscore_series.empty:
        return None

    fig = go.Figure()

    # Z-score line
    fig.add_trace(go.Scatter(
        x=zscore_series.index,
        y=zscore_series.values,
        name='Z-Score',
        line=dict(color='blue', width=2),
        hovertemplate='Data: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
    ))

    # Add threshold bands
    fig.add_hrect(
        y0=2, y1=3,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Zona Crítica Superior",
        annotation_position="top right"
    )

    fig.add_hrect(
        y0=-3, y1=-2,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Zona Crítica Inferior",
        annotation_position="bottom right"
    )

    fig.add_hrect(
        y0=1.5, y1=2,
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Zona de Alerta Superior"
    )

    fig.add_hrect(
        y0=-2, y1=-1.5,
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Zona de Alerta Inferior"
    )

    # Add horizontal reference lines
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
    fig.add_hline(y=1.5, line_dash="dot", line_color="orange", annotation_text="+1.5σ")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="Média")
    fig.add_hline(y=-1.5, line_dash="dot", line_color="orange", annotation_text="-1.5σ")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title="Z-Score",
        height=400,
        hovermode="x unified"
    )

    return fig


def create_zscore_distribution(zscore_series, title="Distribuição de Z-Scores"):
    """
    Create a histogram of z-score distribution.

    Args:
        zscore_series: Pandas Series with z-score values
        title: Plot title

    Returns:
        plotly Figure object
    """
    if zscore_series is None or zscore_series.empty:
        return None

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=zscore_series.values,
        nbinsx=30,
        name='Z-Score Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Add normal distribution overlay
    x_range = np.linspace(zscore_series.min(), zscore_series.max(), 100)
    normal_dist = np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi)

    # Scale to match histogram
    hist_values, bin_edges = np.histogram(zscore_series.values, bins=30)
    scale_factor = hist_values.max() / normal_dist.max()

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist * scale_factor,
        name='Distribuição Normal',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Add vertical lines for thresholds
    for threshold in [-2, -1.5, 0, 1.5, 2]:
        fig.add_vline(x=threshold, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Z-Score",
        yaxis_title="Frequência",
        height=300,
        bargap=0.1
    )

    return fig


def create_zscore_with_price(price_series, zscore_series, title="Preço vs Z-Score"):
    """
    Create a dual-axis plot showing price and z-score.

    Args:
        price_series: Pandas Series with price data
        zscore_series: Pandas Series with z-score data
        title: Plot title

    Returns:
        plotly Figure object
    """
    if price_series is None or price_series.empty or zscore_series is None or zscore_series.empty:
        return None

    fig = go.Figure()

    # Price line (primary y-axis)
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        name='Preço',
        line=dict(color='green', width=2),
        yaxis='y1'
    ))

    # Z-score line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=zscore_series.index,
        y=zscore_series.values,
        name='Z-Score',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))

    # Add z-score threshold lines on secondary axis
    fig.add_hline(y=2, line_dash="dash", line_color="red", yref="y2", annotation_text="+2σ")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", yref="y2", annotation_text="-2σ")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", yref="y2")

    # Update layout with dual y-axes
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis=dict(
            title="Preço",
            titlefont=dict(color="green"),
            tickfont=dict(color="green")
        ),
        yaxis2=dict(
            title="Z-Score",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying="y",
            side="right"
        ),
        height=400,
        hovermode="x unified"
    )

    return fig


def create_zscore_heatmap(zscore_matrix, title="Mapa de Calor de Z-Scores"):
    """
    Create a heatmap of z-scores across multiple assets/time.

    Args:
        zscore_matrix: DataFrame with z-scores (assets as columns, time as index)
        title: Plot title

    Returns:
        plotly Figure object
    """
    if zscore_matrix is None or zscore_matrix.empty:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=zscore_matrix.values,
        x=zscore_matrix.columns,
        y=zscore_matrix.index,
        colorscale='RdBu_r',  # Red-Blue reversed (red for positive z-scores)
        zmin=-3,
        zmax=3,
        text=np.round(zscore_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False,
        hovertemplate='%{y}<br>%{x}<br>Z-Score: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Ativos",
        yaxis_title="Data",
        height=500,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        )
    )

    # Add colorbar
    fig.update_traces(
        colorbar=dict(
            title="Z-Score",
            titleside="right",
            tickvals=[-3, -2, -1, 0, 1, 2, 3],
            ticktext=["-3σ", "-2σ", "-1σ", "0", "+1σ", "+2σ", "+3σ"]
        )
    )

    return fig


def create_zscore_alert_zones(zscore_series, title="Zonas de Alerta de Z-Score"):
    """
    Create a plot highlighting different alert zones based on z-score.

    Args:
        zscore_series: Pandas Series with z-score values
        title: Plot title

    Returns:
        plotly Figure object
    """
    if zscore_series is None or zscore_series.empty:
        return None

    fig = go.Figure()

    # Create colored background zones
    zones = [
        (-float('inf'), -2, "red", "Crítico Baixo"),
        (-2, -1.5, "orange", "Alerta Baixo"),
        (-1.5, -1, "yellow", "Atenção Baixo"),
        (-1, 1, "green", "Normal"),
        (1, 1.5, "yellow", "Atenção Alto"),
        (1.5, 2, "orange", "Alerta Alto"),
        (2, float('inf'), "red", "Crítico Alto")
    ]

    for i, (lower, upper, color, label) in enumerate(zones):
        mask = (zscore_series >= lower) & (zscore_series < upper)
        if mask.any():
            zone_data = zscore_series[mask]

            fig.add_trace(go.Scatter(
                x=zone_data.index,
                y=zone_data.values,
                name=label,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=True
            ))

    # Add threshold lines
    for threshold in [-2, -1.5, -1, 0, 1, 1.5, 2]:
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title="Z-Score",
        height=400,
        hovermode="x unified",
        legend_title="Zonas de Alerta"
    )

    return fig
