import pandas as pd
from sqlalchemy import text
from app.db.engine import engine

def upsert_ohlcv(df, asset_id, engine):
    """
    Faz upsert em lote na tabela ohlcv usando tabela temporária.

    Args:
        df (pd.DataFrame): DataFrame com colunas ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        asset_id (int): ID do ativo.
        engine: SQLAlchemy engine.
    """
    if df.empty:
        return

    # Adicionar asset_id ao DataFrame
    df = df.copy()
    df['asset_id'] = asset_id

    # Converter timestamp para string ISO format
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Inserir diretamente usando executemany para upsert
    with engine.connect() as conn:
        # Preparar dados
        data = df.to_dict('records')

        # Upsert usando INSERT OR REPLACE
        for row in data:
            conn.execute(text("""
                INSERT OR REPLACE INTO ohlcv (asset_id, timestamp, open, high, low, close, volume)
                VALUES (:asset_id, :timestamp, :open, :high, :low, :close, :volume)
            """), row)

        conn.commit()
