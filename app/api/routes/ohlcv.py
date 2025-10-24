from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.engine import engine
from app.db.models import Ohlcv, Asset
from pydantic import BaseModel

router = APIRouter()

class OHLCVResponse(BaseModel):
    asset_id: int
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def get_db():
    with Session(engine) as session:
        yield session

@router.get("/ohlcv", response_model=List[OHLCVResponse])
def get_ohlcv(
    asset_id: Optional[int] = Query(None, description="Asset ID to filter by"),
    symbol: Optional[str] = Query(None, description="Asset symbol to filter by"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """
    Get OHLCV data with optional filtering.
    """
    # Build query
    query_parts = [
        "SELECT o.asset_id, a.symbol, o.timestamp, o.open, o.high, o.low, o.close, o.volume",
        "FROM ohlcv o",
        "JOIN assets a ON o.asset_id = a.id",
        "WHERE 1=1"
    ]
    params = {}

    if asset_id:
        query_parts.append("AND o.asset_id = :asset_id")
        params['asset_id'] = asset_id

    if symbol:
        query_parts.append("AND a.symbol = :symbol")
        params['symbol'] = symbol

    if start_date:
        query_parts.append("AND o.timestamp >= :start_date")
        params['start_date'] = start_date

    if end_date:
        query_parts.append("AND o.timestamp <= :end_date")
        params['end_date'] = end_date

    query_parts.append("ORDER BY o.timestamp DESC")
    query_parts.append("LIMIT :limit")
    params['limit'] = limit

    query = " ".join(query_parts)

    try:
        result = db.execute(text(query), params)
        rows = result.fetchall()

        return [
            OHLCVResponse(
                asset_id=row[0],
                symbol=row[1],
                timestamp=row[2],
                open=row[3],
                high=row[4],
                low=row[5],
                close=row[6],
                volume=row[7]
            )
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/ohlcv/latest/{asset_id}")
def get_latest_ohlcv(asset_id: int, db: Session = Depends(get_db)):
    """
    Get the latest OHLCV record for a specific asset.
    """
    result = db.execute(
        text("""
            SELECT o.asset_id, a.symbol, o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv o
            JOIN assets a ON o.asset_id = a.id
            WHERE o.asset_id = :asset_id
            ORDER BY o.timestamp DESC
            LIMIT 1
        """),
        {'asset_id': asset_id}
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail=f"No OHLCV data found for asset {asset_id}")

    return OHLCVResponse(
        asset_id=result[0],
        symbol=result[1],
        timestamp=result[2],
        open=result[3],
        high=result[4],
        low=result[5],
        close=result[6],
        volume=result[7]
    )

@router.get("/ohlcv/symbol/{symbol}/latest")
def get_latest_ohlcv_by_symbol(symbol: str, db: Session = Depends(get_db)):
    """
    Get the latest OHLCV record for a specific asset by symbol.
    """
    result = db.execute(
        text("""
            SELECT o.asset_id, a.symbol, o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv o
            JOIN assets a ON o.asset_id = a.id
            WHERE a.symbol = :symbol
            ORDER BY o.timestamp DESC
            LIMIT 1
        """),
        {'symbol': symbol}
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail=f"No OHLCV data found for symbol {symbol}")

    return OHLCVResponse(
        asset_id=result[0],
        symbol=result[1],
        timestamp=result[2],
        open=result[3],
        high=result[4],
        low=result[5],
        close=result[6],
        volume=result[7]
    )
