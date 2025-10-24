from app.db.engine import engine
from sqlalchemy import text

def recreate_symbols():
    with engine.connect() as conn:
        # Remove entradas duplicadas
        conn.execute(text("DELETE FROM assets WHERE symbol LIKE '%/USDT'"))
        
        # Adicionar novos símbolos
        symbols = [
            ('BTC/USDT', 'Bitcoin'),
            ('ETH/USDT', 'Ethereum'),
            ('ADA/USDT', 'Cardano'),
            ('SOL/USDT', 'Solana'),
            ('DOT/USDT', 'Polkadot')
        ]
        
        for symbol, name in symbols:
            conn.execute(
                text("INSERT INTO assets (symbol, name, exchange, active) VALUES (:symbol, :name, 'binance', true)"),
                {'symbol': symbol, 'name': name}
            )
        
        conn.commit()

if __name__ == '__main__':
    recreate_symbols()