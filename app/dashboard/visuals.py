"""
Dashboard Visualizations Module

This module provides helper functions for creating Plotly visualizations used in the crypto dashboard.
All functions return plotly.graph_objs.Figure objects that can be displayed in Streamlit.

Functions:
- plot_prices_with_spread: Plot asset prices with spread and z-score bands
- plot_zscore: Plot z-score with entry/exit thresholds
- plot_heatmap: Create correlation heatmap
- plot_portfolio_metrics: Plot portfolio value, returns, and beta
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_prices_with_spread(prices_df, spread_series, zscore_series, bands=None, title="Prices and Spread"):
    """
    Plot asset prices with spread and z-score bands.

    Manual: This visualization shows the price movements of two cointegrated assets along with their spread
    (price difference) and z-score (standardized spread). The z-score helps identify trading opportunities:
    when it exceeds ±2σ, it may signal a mean-reversion opportunity. The bands show ±1σ and ±2σ levels.

    Parameters:
    - prices_df: DataFrame with columns 'asset1_price', 'asset2_price', 'timestamp'
    - spread_series: Series of spread values (asset1 - asset2)
    - zscore_series: Series of z-score values
    - bands: dict with keys 'mean', 'std', 'upper_1', 'lower_1', 'upper_2', 'lower_2'
    - title: Chart title

    Returns:
    - plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Asset Prices', 'Spread and Z-Score')
    )

    # Prices
    fig.add_trace(
        go.Scatter(x=prices_df['timestamp'], y=prices_df['asset1_price'],
                  name='Asset 1', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=prices_df['timestamp'], y=prices_df['asset2_price'],
                  name='Asset 2', line=dict(color='red')),
        row=1, col=1
    )

    # Spread
    fig.add_trace(
        go.Scatter(x=spread_series.index, y=spread_series.values,
                  name='Spread', line=dict(color='green')),
        row=2, col=1
    )

    # Z-score
    fig.add_trace(
        go.Scatter(x=zscore_series.index, y=zscore_series.values,
                  name='Z-Score', line=dict(color='orange')),
        row=2, col=1
    )

    # Bands
    if bands:
        fig.add_hline(y=bands['mean'], line_dash="dash", line_color="gray",
                     annotation_text="Mean", row=2, col=1)
        fig.add_hline(y=bands['upper_1'], line_dash="dot", line_color="yellow",
                     annotation_text="+1σ", row=2, col=1)
        fig.add_hline(y=bands['lower_1'], line_dash="dot", line_color="yellow",
                     annotation_text="-1σ", row=2, col=1)
        fig.add_hline(y=bands['upper_2'], line_dash="dash", line_color="red",
                     annotation_text="+2σ", row=2, col=1)
        fig.add_hline(y=bands['lower_2'], line_dash="dash", line_color="red",
                     annotation_text="-2σ", row=2, col=1)

    fig.update_layout(height=600, title_text=title)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread / Z-Score", row=2, col=1)

    return fig


def plot_zscore(zscore_series, thresholds=None, title="Z-Score Analysis"):
    """
    Plot z-score with entry/exit thresholds.

    Manual: The z-score indicates how many standard deviations the spread is from its mean.
    Trading signals are generated when z-score crosses predefined thresholds:
    - Entry signals: when |z-score| > 2 (statistically significant deviation)
    - Exit signals: when |z-score| < 0.5 (spread has reverted to mean)
    Green areas show potential long/short opportunities.

    Parameters:
    - zscore_series: Series of z-score values
    - thresholds: dict with 'entry' and 'exit' thresholds (default: {'entry': 2, 'exit': 0.5})
    - title: Chart title

    Returns:
    - plotly Figure object
    """
    if thresholds is None:
        thresholds = {'entry': 2, 'exit': 0.5}

    fig = go.Figure()

    # Z-score line
    fig.add_trace(
        go.Scatter(x=zscore_series.index, y=zscore_series.values,
                  name='Z-Score', line=dict(color='blue', width=2))
    )

    # Threshold lines
    fig.add_hline(y=thresholds['entry'], line_dash="dash", line_color="red",
                 annotation_text=f"Entry (+{thresholds['entry']}σ)")
    fig.add_hline(y=-thresholds['entry'], line_dash="dash", line_color="red",
                 annotation_text=f"Entry (-{thresholds['entry']}σ)")
    fig.add_hline(y=thresholds['exit'], line_dash="dot", line_color="green",
                 annotation_text=f"Exit (+{thresholds['exit']}σ)")
    fig.add_hline(y=-thresholds['exit'], line_dash="dot", line_color="green",
                 annotation_text=f"Exit (-{thresholds['exit']}σ)")

    # Zero line
    fig.add_hline(y=0, line_color="black", annotation_text="Mean")

    # Fill areas for signals
    fig.add_hrect(y0=thresholds['entry'], y1=thresholds['entry']*2, fillcolor="lightgreen", opacity=0.3,
                 annotation_text="Short Signal Area", annotation_position="top right")
    fig.add_hrect(y0=-thresholds['entry']*2, y1=-thresholds['entry'], fillcolor="lightcoral", opacity=0.3,
                 annotation_text="Long Signal Area", annotation_position="bottom right")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Z-Score",
        height=400
    )

    return fig


def plot_heatmap(corr_df, title="Correlation Heatmap"):
    """
    Create correlation heatmap.

    Manual: This heatmap shows the correlation coefficients between different crypto assets.
    Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).
    Darker colors indicate stronger correlations. Use this to identify assets that move together
    or in opposite directions, which is useful for portfolio diversification and pair trading.

    Parameters:
    - corr_df: DataFrame with correlation matrix
    - title: Chart title

    Returns:
    - plotly Figure object
    """
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title=title
    )

    fig.update_layout(height=600)
    fig.update_coloraxes(colorbar_title="Correlation")

    return fig


def plot_portfolio_metrics(portfolio_df, title="Portfolio Metrics"):
    """
    Plot portfolio value, returns, and beta.

    Manual: This dashboard shows key portfolio performance metrics:
    - Portfolio Value: Total value of holdings over time
    - Returns: Periodic returns (log returns for better statistical properties)
    - Beta: Portfolio's sensitivity to market movements (vs benchmark)
    Use these metrics to assess portfolio performance, risk, and market correlation.

    Parameters:
    - portfolio_df: DataFrame with columns 'value', 'returns', 'beta', 'timestamp'
    - title: Chart title

    Returns:
    - plotly Figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Portfolio Value', 'Portfolio Returns', 'Portfolio Beta')
    )

    # Portfolio Value
    fig.add_trace(
        go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['value'],
                  name='Portfolio Value', line=dict(color='green')),
        row=1, col=1
    )

    # Returns
    fig.add_trace(
        go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['returns'],
                  name='Returns', line=dict(color='blue')),
        row=2, col=1
    )

    # Beta
    fig.add_trace(
        go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['beta'],
                  name='Beta', line=dict(color='red')),
        row=3, col=1
    )

    # Beta reference line
    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                 annotation_text="Market Beta = 1", row=3, col=1)

    fig.update_layout(height=800, title_text=title)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)
    fig.update_yaxes(title_text="Beta", row=3, col=1)

    return fig
