import logging
from app.db.engine import engine
from app.etl.fetchers import fetch_binance_ohlcv
from app.etl.upsert import upsert_ohlcv
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all():
    """
    Executa ETL para todos os ativos ativos.
    """
    with engine.connect() as conn:
        # Carregar ativos ativos
        result = conn.execute(text("SELECT id, symbol FROM assets WHERE active = true"))
        assets = result.fetchall()

    for asset_id, symbol in assets:
        logger.info(f"Processando {symbol} (ID: {asset_id})")
        try:
            # Buscar dados OHLCV (últimos 1000 dias)
            # Garantir que o símbolo tenha o par USDT
            trading_symbol = symbol if "/USDT" in symbol else f"{symbol}/USDT"
            df = fetch_binance_ohlcv(symbol=trading_symbol, timeframe='1d', limit=1000)
            if not df.empty:
                upsert_ohlcv(df, asset_id, engine)
                logger.info(f"Inseridos {len(df)} registros para {symbol}")
            else:
                logger.warning(f"Nenhum dado encontrado para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao processar {symbol}: {e}")

if __name__ == '__main__':
    run_all()
