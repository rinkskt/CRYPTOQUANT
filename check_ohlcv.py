from app.db.engine import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM ohlcv"))
    count = result.fetchone()[0]
    print(f"Total OHLCV records: {count}")

    # Check a few records
    result = conn.execute(text("SELECT asset_id, timestamp, close FROM ohlcv LIMIT 5"))
    rows = result.fetchall()
    print("Sample records:")
    for row in rows:
        print(row)
