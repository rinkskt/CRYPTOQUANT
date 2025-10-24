"""
Funções auxiliares para obter preços e dados de mercado.
"""
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import requests
import ccxt
import streamlit as st

API_BASE_URL = "http://localhost:8000"

def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Obtém os preços atuais dos ativos usando a API local ou Binance como fallback.
    """
    prices = {}
    
    # Função auxiliar para adicionar USDT ao símbolo se necessário
    def format_symbol(s: str) -> str:
        return s if s.endswith('USDT') else f"{s}USDT"
    
    # Prepara símbolos para API local
    api_symbols = [format_symbol(s) for s in symbols]
    
    try:
        # Tenta primeiro via API local
        response = requests.get(f"{API_BASE_URL}/api/v1/crypto/prices", 
                              params={"symbols": ",".join(api_symbols)})
        
        if response.status_code == 200:
            prices_data = response.json()
            return {s.replace('USDT', ''): float(p) 
                   for s, p in prices_data.items()}
            
    except requests.exceptions.RequestException:
        pass
        
    # Fallback: Binance
    try:
        exchange = ccxt.binance()
        for symbol in symbols:
            try:
                binance_symbol = format_symbol(symbol)
                ticker = exchange.fetch_ticker(binance_symbol)
                prices[symbol] = float(ticker['last'])
            except:
                continue
                
    except Exception as e:
        st.error(f"Erro ao obter preços da Binance: {str(e)}")
        
    return prices

def get_historical_data(symbols: List[str], 
                       start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
    """
    Obtém dados históricos OHLCV dos ativos.
    """
    df = pd.DataFrame()
    
    # Função auxiliar para adicionar USDT ao símbolo se necessário
    def format_symbol(s: str) -> str:
        return s if s.endswith('USDT') else f"{s}USDT"
    
    # Prepara símbolos para API local
    api_symbols = [format_symbol(s) for s in symbols]
    
    try:
        # Tenta via API local
        params = {
            "symbols": ",".join(api_symbols),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        response = requests.get(f"{API_BASE_URL}/api/v1/ohlcv/historical",
                              params=params)
                              
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Remove USDT dos símbolos
            df['symbol'] = df['symbol'].str.replace('USDT', '')
            return df
            
    except requests.exceptions.RequestException:
        pass
        
    # Fallback: Binance
    try:
        exchange = ccxt.binance()
        all_data = []
        
        for symbol in symbols:
            try:
                binance_symbol = format_symbol(symbol)
                ohlcv = exchange.fetch_ohlcv(
                    binance_symbol,
                    timeframe='1d',
                    since=int(start_date.timestamp() * 1000),
                    limit=1000
                )
                
                symbol_df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                symbol_df['symbol'] = symbol  # Usa símbolo sem USDT
                symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'], unit='ms')
                all_data.append(symbol_df)
                
            except Exception as e:
                st.warning(f"Erro ao obter dados históricos para {symbol}: {str(e)}")
                continue
                
        if all_data:
            df = pd.concat(all_data)
            df = df[df['timestamp'].between(start_date, end_date)]
            return df
            
    except Exception as e:
        st.error(f"Erro ao obter dados históricos da Binance: {str(e)}")
        
    return df

from ...api_client import API_BASE_URL

def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Obtém preços atuais dos ativos, primeiro tentando a API local e depois a Binance.
    """
    prices = {}
    
    # Tenta obter preços da API local primeiro
    try:
        for symbol in symbols:
            response = requests.get(f"{API_BASE_URL}/assets/{symbol}/price", timeout=5)
            if response.status_code == 200:
                prices[symbol] = response.json()['price']
    except:
        pass
    
    # Para símbolos que faltam, tenta direto da Binance
    missing_symbols = [s for s in symbols if s not in prices]
    if missing_symbols:
        try:
            exchange = ccxt.binance()
            for symbol in missing_symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker['last']
                except:
                    st.warning(f"Não foi possível obter preço para {symbol}")
                    prices[symbol] = 0.0
        except Exception as e:
            st.error(f"Erro ao conectar com a Binance: {e}")
    
    return prices

def get_historical_data(symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Obtém dados históricos dos ativos, primeiro tentando a API local e depois a Binance.
    """
    data = {}
    
    # Tenta obter dados da API local primeiro
    try:
        for symbol in symbols:
            response = requests.get(
                f"{API_BASE_URL}/ohlcv",
                params={
                    "symbol": symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "limit": 1000
                },
                timeout=10
            )
            if response.status_code == 200:
                df = pd.DataFrame(response.json())
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                data[symbol] = df['close']
    except:
        pass
    
    # Para símbolos que faltam, tenta direto da Binance
    missing_symbols = [s for s in symbols if s not in data]
    if missing_symbols:
        try:
            exchange = ccxt.binance()
            timeframe = '1d'
            since = int(start_date.timestamp() * 1000)
            
            for symbol in missing_symbols:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df['close']
                except Exception as e:
                    st.warning(f"Não foi possível obter dados históricos para {symbol}: {e}")
        except Exception as e:
            st.error(f"Erro ao conectar com a Binance: {e}")
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)