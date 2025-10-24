"""
Data Loader Module for Binance API

Este módulo gerencia a coleta de dados da Binance API para criptomoedas.
Suporta dados históricos de preços, volumes e outras métricas de mercado.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import time
import streamlit as st
import requests


def get_binance_klines(symbol="BTCUSDT", interval="1h", limit=500):
    """
    Função padrão para obter dados da Binance via API pública.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


class BinanceDataLoader:
    """
    Classe para carregar dados da Binance API usando requests (sem chave API).
    """

    def __init__(self):
        pass  # Não precisa de client

    @st.cache_data(ttl=3600)  # Cache por 1 hora
    def get_historical_data(_self,
                           symbol: str,
                           interval: str = '1d',
                           limit: int = 500,
                           start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega dados históricos da Binance via API pública.

        Args:
            symbol: Par de trading (ex: 'BTCUSDT')
            interval: Intervalo temporal ('1m', '5m', '1h', '1d', etc.)
            limit: Número máximo de candles
            start_date: Data inicial (formato 'YYYY-MM-DD')

        Returns:
            DataFrame com dados OHLCV
        """
        try:
            return get_binance_klines(symbol, interval, limit)
        except Exception as e:
            print(f"Error loading data: {e}")
            return _self._get_mock_data(symbol, interval, limit)

    def _get_mock_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Gera dados mock para desenvolvimento/teste.
        """
        # Cria timestamps
        end_date = datetime.now()
        if interval == '1d':
            start_date = end_date - timedelta(days=limit)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        elif interval == '1h':
            start_date = end_date - timedelta(hours=limit)
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
        else:
            start_date = end_date - timedelta(days=limit//24)
            dates = pd.date_range(start=start_date, end=end_date, freq='H')

        # Gera preços simulados
        np.random.seed(42)  # Para reprodutibilidade

        # Preço base baseado no símbolo
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'BNBUSDT': 400,
            'ADAUSDT': 0.5,
            'SOLUSDT': 100
        }
        base_price = base_prices.get(symbol.upper(), 100)

        # Simula caminhada aleatória com drift
        returns = np.random.normal(0.001, 0.02, len(dates))  # Retorno diário médio
        prices = base_price * np.exp(np.cumsum(returns))

        # Cria OHLCV
        high_mult = 1 + np.random.uniform(0, 0.05, len(dates))
        low_mult = 1 - np.random.uniform(0, 0.05, len(dates))

        df = pd.DataFrame({
            'open': prices,
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices * (1 + returns),
            'volume': np.random.uniform(1000, 100000, len(dates))
        }, index=dates)

        return df

    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def get_current_price(_self, symbol: str) -> Optional[float]:
        """
        Obtém preço atual do ativo via API pública.
        """
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
            response = requests.get(url)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None

    def get_multiple_assets(self, symbols: List[str], interval: str = '1d', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Carrega dados para múltiplos ativos.
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_historical_data(symbol, interval, limit)
            time.sleep(0.1)  # Rate limiting
        return data


# Instância global
data_loader = BinanceDataLoader()


def get_binance_data(symbol: str, interval: str = '1d', limit: int = 500) -> pd.DataFrame:
    """
    Função de conveniência para carregar dados da Binance.
    """
    return data_loader.get_historical_data(symbol, interval, limit)


def get_current_price(symbol: str) -> Optional[float]:
    """
    Função de conveniência para obter preço atual.
    """
    return data_loader.get_current_price(symbol)


def get_portfolio_data(symbols, interval="1h", limit=500):
    """
    Carrega dados para múltiplos ativos e normaliza para o mesmo range de tempo.
    """
    portfolio_data = {}
    for symbol in symbols:
        df = get_binance_klines(symbol, interval, limit)
        portfolio_data[symbol] = df

    # Normaliza timestamps para o mesmo range
    if portfolio_data:
        min_timestamp = min(df['timestamp'].min() for df in portfolio_data.values())
        for symbol in portfolio_data:
            portfolio_data[symbol] = portfolio_data[symbol][portfolio_data[symbol]['timestamp'] >= min_timestamp]

    return portfolio_data


__all__ = ['get_binance_data', 'get_current_price', 'BinanceDataLoader', 'get_portfolio_data']
