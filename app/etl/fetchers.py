import ccxt
import pandas as pd
import logging
from typing import Optional
from app.etl.retry import retry_with_backoff
from app.etl.validation import OhlcvData

logger = logging.getLogger(__name__)

@retry_with_backoff(
    retries=3,
    backoff_in_seconds=2,
    max_backoff_in_seconds=30,
    exceptions=(ccxt.NetworkError, ccxt.ExchangeError)
)
def fetch_binance_ohlcv(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    since: Optional[int] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Busca dados OHLCV da Binance usando ccxt com retry e validação.

    Args:
        symbol (str): Par de moedas, ex: 'BTC/USDT'.
        timeframe (str): Intervalo de tempo, ex: '1d'.
        since (int): Timestamp em ms para início (opcional).
        limit (int): Número máximo de candles.

    Returns:
        pd.DataFrame: DataFrame com colunas ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        
    Raises:
        ValueError: Se os dados não passarem na validação.
        ccxt.NetworkError: Em caso de erro de rede (com retry).
        ccxt.ExchangeError: Em caso de erro da exchange (com retry).
    """
    try:
        exchange = ccxt.binance()
        klines = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        
        df = pd.DataFrame(
            klines,
            columns=['ts', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Validar cada linha
        for _, row in df.iterrows():
            OhlcvData(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
        return df
        
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
        raise ValueError(f"Error processing data: {str(e)}")
