from fastapi import APIRouter, HTTPException
from typing import List
import ccxt

router = APIRouter()

@router.get("/exchanges")
def get_exchanges():
    """List all available exchanges"""
    return {"exchanges": ccxt.exchanges}

@router.get("/tickers/{exchange}")
def get_tickers(exchange: str):
    """Get tickers from a specific exchange"""
    try:
        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class()
        tickers = ex.fetch_tickers()
        return {"tickers": list(tickers.keys())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
