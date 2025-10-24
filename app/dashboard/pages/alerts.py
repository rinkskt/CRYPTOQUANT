"""
Alerts and Monitoring Page

This module provides the alerts and monitoring page.
Shows active alerts, monitoring dashboard, and notification settings.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from app.dashboard.api_client import (
    get_assets, get_ohlcv_data, get_analytics_data
)
from app.analytics.stats import compute_zscore


def show_alerts_page():
    """
    Display the alerts and monitoring page.
    """
    st.header("🚨 Alertas e Monitoramento")

    # Sidebar controls
    st.sidebar.header("Configurações de Alertas")

    # Alert types
    alert_types = st.sidebar.multiselect(
        "Tipos de Alerta:",
        options=[
            "Z-score > 2 (Sobrecompra)",
            "Z-score < -2 (Sobrevenda)",
            "Cointegração Perdida",
            "Volatilidade Alta",
            "Quebra de Correlação"
        ],
        default=["Z-score > 2 (Sobrecompra)", "Z-score < -2 (Sobrevenda)"]
    )

    # Asset selection for monitoring
    assets = get_assets()
    if not assets:
        st.error("Não foi possível carregar os ativos.")
        return

    asset_symbols = [asset['symbol'] for asset in assets]

    monitor_assets = st.sidebar.multiselect(
        "Ativos para Monitorar:",
        options=asset_symbols,
        default=asset_symbols[:10] if len(asset_symbols) >= 10 else asset_symbols
    )

    # Refresh interval
    refresh_interval = st.sidebar.selectbox(
        "Intervalo de Atualização:",
        options=["Manual", "30 segundos", "1 minuto", "5 minutos"],
        index=0
    )

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Painel de Alertas", "📋 Histórico", "⚙️ Configurações"])

    with tab1:
        show_alerts_dashboard(monitor_assets, alert_types)

    with tab2:
        show_alerts_history(monitor_assets)

    with tab3:
        show_alerts_settings()


def show_alerts_dashboard(monitor_assets, alert_types):
    """
    Show the alerts dashboard.
    """
    st.subheader("Painel de Alertas Ativos")

    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Atualizar Alertas"):
            st.cache_data.clear()
            st.rerun()

    # Generate alerts
    with st.spinner("Verificando alertas..."):
        active_alerts = generate_alerts(monitor_assets, alert_types)

    if active_alerts:
        # Display active alerts
        for alert in active_alerts:
            if alert['severity'] == 'high':
                st.error(f"🚨 **{alert['type']}** - {alert['asset']}: {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"⚠️ **{alert['type']}** - {alert['asset']}: {alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['type']}** - {alert['asset']}: {alert['message']}")

        # Alerts summary
        st.subheader("Resumo de Alertas")

        alert_counts = pd.DataFrame(active_alerts)
        if not alert_counts.empty:
            severity_counts = alert_counts['severity'].value_counts()

            col1, col2, col3 = st.columns(3)

            with col1:
                high_count = severity_counts.get('high', 0)
                st.metric("Alertas Críticos", high_count)

            with col2:
                medium_count = severity_counts.get('medium', 0)
                st.metric("Alertas Moderados", medium_count)

            with col3:
                low_count = severity_counts.get('low', 0)
                st.metric("Alertas Baixos", low_count)

    else:
        st.success("✅ Nenhum alerta ativo no momento.")

    # Monitoring charts
    st.subheader("Monitoramento em Tempo Real")

    if monitor_assets:
        # Z-score monitoring for selected assets
        zscore_data = get_zscore_monitoring(monitor_assets[:5])  # Limit to 5 for performance

        if zscore_data:
            fig = go.Figure()

            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (asset, zscores) in enumerate(zscore_data.items()):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=zscores.index,
                    y=zscores.values,
                    name=asset,
                    line=dict(color=color, width=2)
                ))

            # Add threshold lines
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
            fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")
            fig.add_hline(y=0, line_dash="solid", line_color="gray")

            fig.update_layout(
                title="Monitoramento de Z-scores",
                xaxis_title="Data",
                yaxis_title="Z-score",
                height=400
            )

            st.plotly_chart(fig, config={'responsive': True})


def generate_alerts(monitor_assets, alert_types):
    """
    Generate alerts based on current market conditions.
    """
    alerts = []

    for asset in monitor_assets:
        try:
            # Get recent data
            data = get_ohlcv_data(asset, limit=30)  # Last 30 days
            if not data:
                continue

            df = pd.DataFrame(data)
            prices = df['close']

            # Calculate z-score
            zscore_series = compute_zscore(prices)
            current_zscore = zscore_series.iloc[-1] if not zscore_series.empty else 0

            # Z-score alerts
            if "Z-score > 2 (Sobrecompra)" in alert_types and current_zscore > 2:
                alerts.append({
                    'asset': asset,
                    'type': 'Z-score Alto',
                    'message': f'Z-score = {current_zscore:.2f} (Sobrecompra)',
                    'severity': 'high',
                    'timestamp': datetime.now()
                })

            if "Z-score < -2 (Sobrevenda)" in alert_types and current_zscore < -2:
                alerts.append({
                    'asset': asset,
                    'type': 'Z-score Baixo',
                    'message': f'Z-score = {current_zscore:.2f} (Sobrevenda)',
                    'severity': 'high',
                    'timestamp': datetime.now()
                })

            # Volatility alert
            if "Volatilidade Alta" in alert_types:
                returns = prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(365)  # Annualized
                if volatility > 1.0:  # 100% annualized volatility threshold
                    alerts.append({
                        'asset': asset,
                        'type': 'Volatilidade Alta',
                        'message': f'Volatilidade anualizada = {volatility:.1%}',
                        'severity': 'medium',
                        'timestamp': datetime.now()
                    })

            # Cointegration alerts (simplified - would need pair data)
            # This is a placeholder for future implementation

        except Exception as e:
            continue  # Skip assets with errors

    return alerts


def get_zscore_monitoring(assets):
    """
    Get z-score data for monitoring.
    """
    zscore_data = {}

    for asset in assets:
        try:
            data = get_ohlcv_data(asset, limit=30)
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

                zscores = compute_zscore(df['close'])
                zscore_data[asset] = zscores
        except Exception as e:
            continue

    return zscore_data


def show_alerts_history(monitor_assets):
    """
    Show alerts history.
    """
    st.subheader("Histórico de Alertas")

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    date_range = st.date_input(
        "Período:",
        value=(start_date.date(), end_date.date())
    )

    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())

    # Mock historical alerts (in real implementation, this would come from database)
    historical_alerts = generate_mock_history(monitor_assets, start_date, end_date)

    if historical_alerts:
        # Convert to DataFrame
        alerts_df = pd.DataFrame(historical_alerts)
        alerts_df = alerts_df.sort_values('timestamp', ascending=False)

        # Display as table
        display_df = alerts_df[['timestamp', 'asset', 'type', 'message', 'severity']].copy()
        display_df.columns = ['Data/Hora', 'Ativo', 'Tipo', 'Mensagem', 'Severidade']

        # Color code severity
        def color_severity(val):
            if val == 'high':
                return 'background-color: #f8d7da'
            elif val == 'medium':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'

        styled_df = display_df.style.applymap(color_severity, subset=['Severidade'])
        st.dataframe(styled_df, width='stretch')

        # Alerts by asset chart
        st.subheader("Alertas por Ativo")

        asset_counts = alerts_df['asset'].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=asset_counts.index,
            y=asset_counts.values,
            name='Alertas'
        ))

        fig.update_layout(
            title="Número de Alertas por Ativo",
            xaxis_title="Ativo",
            yaxis_title="Número de Alertas",
            height=300
        )

        st.plotly_chart(fig, config={'responsive': True})

    else:
        st.info("Nenhum alerta histórico encontrado no período selecionado.")


def generate_mock_history(assets, start_date, end_date):
    """
    Generate mock historical alerts for demonstration.
    """
    alerts = []
    current_time = start_date

    while current_time <= end_date:
        # Random alerts for demonstration
        if np.random.random() < 0.1:  # 10% chance of alert per day
            asset = np.random.choice(assets)
            alert_types = ['Z-score Alto', 'Z-score Baixo', 'Volatilidade Alta']
            alert_type = np.random.choice(alert_types)

            severity = 'high' if 'Z-score' in alert_type else 'medium'

            alerts.append({
                'timestamp': current_time,
                'asset': asset,
                'type': alert_type,
                'message': f'Alerta simulado para {asset}',
                'severity': severity
            })

        current_time += timedelta(days=1)

    return alerts


def show_alerts_settings():
    """
    Show alerts configuration settings.
    """
    st.subheader("Configurações de Notificações")

    # Email notifications
    st.write("**Notificações por E-mail**")
    email_enabled = st.checkbox("Habilitar notificações por e-mail", value=False)

    if email_enabled:
        email_address = st.text_input("Endereço de e-mail:")
        email_frequency = st.selectbox(
            "Frequência:",
            options=["Imediato", "Diário", "Semanal"],
            index=0
        )

    # Telegram notifications
    st.write("**Notificações via Telegram**")
    telegram_enabled = st.checkbox("Habilitar notificações via Telegram", value=False)

    if telegram_enabled:
        telegram_token = st.text_input("Token do Bot Telegram:")
        telegram_chat_id = st.text_input("Chat ID:")

    # Webhook notifications
    st.write("**Notificações via Webhook**")
    webhook_enabled = st.checkbox("Habilitar notificações via Webhook", value=False)

    if webhook_enabled:
        webhook_url = st.text_input("URL do Webhook:")

    # Alert thresholds
    st.subheader("Limites de Alerta")

    col1, col2 = st.columns(2)

    with col1:
        zscore_threshold = st.slider(
            "Limite Z-score:",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Limite para alertas de sobrecompra/sobrevenda"
        )

        volatility_threshold = st.slider(
            "Limite Volatilidade (%):",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Limite para alertas de volatilidade alta (anualizada)"
        )

    with col2:
        correlation_threshold = st.slider(
            "Limite Correlação:",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Limite para alertas de quebra de correlação"
        )

    # Save settings button
    if st.button("💾 Salvar Configurações", type="primary"):
        st.success("Configurações salvas com sucesso!")

        # In real implementation, save to database/user preferences
        settings = {
            'email_enabled': email_enabled,
            'email_address': email_address if email_enabled else None,
            'telegram_enabled': telegram_enabled,
            'webhook_enabled': webhook_enabled,
            'zscore_threshold': zscore_threshold,
            'volatility_threshold': volatility_threshold / 100,  # Convert to decimal
            'correlation_threshold': correlation_threshold
        }

        st.json(settings)  # Display saved settings
