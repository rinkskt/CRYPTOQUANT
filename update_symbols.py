from app.db.engine import engine
from sqlalchemy import text

def update_symbols():
    with engine.connect() as conn:
        conn.execute(text("UPDATE assets SET symbol = symbol || '/USDT' WHERE symbol NOT LIKE '%/USDT' AND active = true"))
        conn.commit()

if __name__ == '__main__':
    update_symbols()