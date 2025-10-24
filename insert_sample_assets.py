from app.db.engine import engine
from sqlalchemy import text

# Sample assets
assets = [
    ('BTC/USDT', 'Bitcoin', 'binance', True),
    ('ETH/USDT', 'Ethereum', 'binance', True),
    ('ADA/USDT', 'Cardano', 'binance', True),
    ('SOL/USDT', 'Solana', 'binance', True),
    ('DOT/USDT', 'Polkadot', 'binance', True),
]

with engine.connect() as conn:
    for symbol, name, exchange, active in assets:
        conn.execute(text("""
            INSERT OR IGNORE INTO assets (symbol, name, exchange, active)
            VALUES (:symbol, :name, :exchange, :active)
        """), {'symbol': symbol, 'name': name, 'exchange': exchange, 'active': active})
    conn.commit()

print("Sample assets inserted.")
