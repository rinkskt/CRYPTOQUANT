from app.workers.celery_app import celery
from app.etl.run_etl import run_all as run_etl
from app.etl.fetchers import fetch_binance_ohlcv
from app.db.engine import engine
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

@celery.task(
    bind=True,
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=600,
    rate_limit='60/m'
)
def fetch_all_ohlcv(self):
    """Tarefa Celery para buscar dados OHLCV para todos os ativos ativos"""
    try:
        run_etl()
        return {'status': 'success', 'message': 'ETL completed successfully'}
    except Exception as exc:
        logger.error(f'Error in fetch_all_ohlcv: {exc}')
        raise self.retry(exc=exc)

@celery.task(
    bind=True,
    max_retries=3,
    retry_backoff=True
)
def fetch_asset_ohlcv(self, asset_id: int, symbol: str):
    """Tarefa Celery para buscar dados OHLCV para um ativo específico"""
    try:
        df = fetch_binance_ohlcv(symbol=symbol, timeframe='1d', limit=1000)
        if not df.empty:
            with engine.connect() as conn:
                # Inserir ou atualizar dados
                for _, row in df.iterrows():
                    query = text("""
                        INSERT INTO ohlcv (asset_id, timestamp, open, high, low, close, volume)
                        VALUES (:asset_id, :timestamp, :open, :high, :low, :close, :volume)
                        ON CONFLICT (asset_id, timestamp) DO UPDATE
                        SET open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """)
                    conn.execute(query, {
                        'asset_id': asset_id,
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                conn.commit()
            return {
                'status': 'success',
                'message': f'Updated {len(df)} records for {symbol}'
            }
        return {
            'status': 'warning',
            'message': f'No data found for {symbol}'
        }
    except Exception as exc:
        logger.error(f'Error in fetch_asset_ohlcv for {symbol}: {exc}')
        raise self.retry(exc=exc)